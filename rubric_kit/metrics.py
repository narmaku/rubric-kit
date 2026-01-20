"""LLM inference metrics tracking for rubric-kit.

This module provides:
- LLMCallMetrics: Dataclass for individual call metrics
- MetricsAggregator: Thread-safe collector and aggregator for metrics
- Token/cost estimation functions for dry-run mode
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import litellm


@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM API call.
    
    Attributes:
        call_id: Unique identifier for this call
        call_type: Type of call (e.g., 'generate_dimensions', 'evaluate_criterion')
        model: Model name used for the call
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens (prompt + completion)
        cost_usd: Estimated cost in USD
        latency_seconds: Time taken for the call in seconds
        context_id: Optional context identifier (e.g., rubric_id, criterion_name)
        timestamp: When the call was made
    """
    call_id: str
    call_type: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_seconds: float
    context_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "call_id": self.call_id,
            "call_type": self.call_type,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_seconds": round(self.latency_seconds, 3),
            "context_id": self.context_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MetricsSummary:
    """Aggregated metrics summary.
    
    Attributes:
        total_calls: Total number of LLM calls
        prompt_tokens: Total prompt tokens across all calls
        completion_tokens: Total completion tokens across all calls
        total_tokens: Total tokens (prompt + completion)
        cost_usd: Total estimated cost in USD
        latency_seconds: Total time spent on LLM calls
    """
    total_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_calls": self.total_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.cost_usd, 6),
            "total_time_seconds": round(self.latency_seconds, 3),
        }


class MetricsAggregator:
    """Thread-safe aggregator for LLM call metrics.
    
    Collects metrics from individual LLM calls and provides aggregation
    by call type, context, and overall summary.
    
    Thread-safe for use with parallel judge execution.
    
    Example:
        aggregator = MetricsAggregator()
        
        # Record a call
        aggregator.record_call(
            call_type="evaluate_criterion",
            model="gpt-4",
            usage=response.usage,
            latency=2.5,
            context_id="criterion_1"
        )
        
        # Get summary
        summary = aggregator.get_summary()
        print(f"Total cost: ${summary.cost_usd:.4f}")
    """
    
    def __init__(self, include_call_log: bool = False):
        """Initialize the aggregator.
        
        Args:
            include_call_log: If True, include detailed call log in output.
                             Default False to keep output compact.
        """
        self._calls: List[LLMCallMetrics] = []
        self._lock = threading.Lock()
        self.include_call_log = include_call_log
    
    def record_call(
        self,
        call_type: str,
        model: str,
        usage: Any,
        latency: float,
        context_id: Optional[str] = None,
        response: Optional[Any] = None
    ) -> LLMCallMetrics:
        """Record metrics from an LLM call.
        
        Args:
            call_type: Type of call (e.g., 'generate_dimensions', 'evaluate_criterion')
            model: Model name used
            usage: Usage object from LiteLLM response (has prompt_tokens, completion_tokens)
            latency: Time taken for the call in seconds
            context_id: Optional context identifier
            response: Optional full response for cost calculation
            
        Returns:
            The recorded LLMCallMetrics object
        """
        prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
        completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
        total_tokens = getattr(usage, 'total_tokens', 0) or (prompt_tokens + completion_tokens)
        
        # Calculate cost using LiteLLM
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens, response)
        
        metrics = LLMCallMetrics(
            call_id=str(uuid.uuid4()),
            call_type=call_type,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_seconds=latency,
            context_id=context_id,
        )
        
        with self._lock:
            self._calls.append(metrics)
        
        return metrics
    
    def _calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        response: Optional[Any] = None
    ) -> float:
        """Calculate cost for an LLM call.
        
        Uses LiteLLM's completion_cost function when possible.
        Falls back to token-based estimation if that fails.
        """
        try:
            if response is not None:
                # Try using the full response for accurate cost
                return litellm.completion_cost(completion_response=response)
            else:
                # Calculate cost from token counts using LiteLLM's pricing database
                prompt_cost, completion_cost = litellm.cost_per_token(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
                return prompt_cost + completion_cost
        except Exception:
            # Fallback: rough estimation based on common pricing
            # GPT-4: ~$0.03/1K input, ~$0.06/1K output
            # This is a conservative estimate
            return (prompt_tokens * 0.00003) + (completion_tokens * 0.00006)
    
    def get_summary(self) -> MetricsSummary:
        """Get aggregated summary of all recorded calls.
        
        Returns:
            MetricsSummary with totals across all calls
        """
        with self._lock:
            calls = list(self._calls)
        
        summary = MetricsSummary()
        for call in calls:
            summary.total_calls += 1
            summary.prompt_tokens += call.prompt_tokens
            summary.completion_tokens += call.completion_tokens
            summary.total_tokens += call.total_tokens
            summary.cost_usd += call.cost_usd
            summary.latency_seconds += call.latency_seconds
        
        return summary
    
    def get_by_type(self) -> Dict[str, MetricsSummary]:
        """Get metrics aggregated by call type.
        
        Returns:
            Dictionary mapping call_type to MetricsSummary
        """
        with self._lock:
            calls = list(self._calls)
        
        by_type: Dict[str, MetricsSummary] = {}
        
        for call in calls:
            if call.call_type not in by_type:
                by_type[call.call_type] = MetricsSummary()
            
            summary = by_type[call.call_type]
            summary.total_calls += 1
            summary.prompt_tokens += call.prompt_tokens
            summary.completion_tokens += call.completion_tokens
            summary.total_tokens += call.total_tokens
            summary.cost_usd += call.cost_usd
            summary.latency_seconds += call.latency_seconds
        
        return by_type
    
    def get_by_context(self) -> Dict[str, MetricsSummary]:
        """Get metrics aggregated by context_id.
        
        Returns:
            Dictionary mapping context_id to MetricsSummary
        """
        with self._lock:
            calls = list(self._calls)
        
        by_context: Dict[str, MetricsSummary] = {}
        
        for call in calls:
            ctx = call.context_id or "_no_context"
            if ctx not in by_context:
                by_context[ctx] = MetricsSummary()
            
            summary = by_context[ctx]
            summary.total_calls += 1
            summary.prompt_tokens += call.prompt_tokens
            summary.completion_tokens += call.completion_tokens
            summary.total_tokens += call.total_tokens
            summary.cost_usd += call.cost_usd
            summary.latency_seconds += call.latency_seconds
        
        return by_context
    
    def get_calls(self) -> List[LLMCallMetrics]:
        """Get a copy of all recorded calls.
        
        Returns:
            List of LLMCallMetrics (copy, not reference)
        """
        with self._lock:
            return list(self._calls)
    
    def clear(self) -> None:
        """Clear all recorded metrics."""
        with self._lock:
            self._calls.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to dictionary for YAML/JSON serialization.
        
        Returns:
            Dictionary with 'summary', 'by_type', and optionally 'calls' keys
        """
        result: Dict[str, Any] = {
            "summary": self.get_summary().to_dict(),
            "by_type": {
                call_type: summary.to_dict()
                for call_type, summary in self.get_by_type().items()
            }
        }
        
        if self.include_call_log:
            result["calls"] = [call.to_dict() for call in self.get_calls()]
        
        return result


def estimate_tokens(
    model: str,
    messages: List[Dict[str, str]],
) -> int:
    """Estimate token count for messages using LiteLLM.
    
    Args:
        model: Model name (e.g., 'gpt-4', 'claude-3-sonnet')
        messages: List of message dicts with 'role' and 'content'
        
    Returns:
        Estimated token count
    """
    try:
        return litellm.token_counter(model=model, messages=messages)
    except Exception:
        # Fallback: rough estimation (~4 chars per token)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // 4


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Estimate cost for given token counts.
    
    Args:
        model: Model name
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens (use max_tokens for upper bound)
        
    Returns:
        Estimated cost in USD
    """
    try:
        # Use LiteLLM's cost calculation with explicit token counts
        # This uses the model's pricing from LiteLLM's pricing database
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        return prompt_cost + completion_cost
    except Exception:
        # Fallback estimation (GPT-4 pricing as conservative default)
        return (prompt_tokens * 0.00003) + (completion_tokens * 0.00006)


@dataclass
class DryRunEstimate:
    """Estimate for dry-run mode.
    
    Attributes:
        total_calls: Estimated number of LLM calls
        prompt_tokens: Estimated prompt tokens
        completion_tokens_max: Maximum possible completion tokens
        cost_min_usd: Minimum estimated cost (if completions are short)
        cost_max_usd: Maximum estimated cost (if completions hit max_tokens)
    """
    total_calls: int
    prompt_tokens: int
    completion_tokens_max: int
    cost_min_usd: float
    cost_max_usd: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "estimated_calls": self.total_calls,
            "estimated_prompt_tokens": self.prompt_tokens,
            "estimated_completion_tokens_max": self.completion_tokens_max,
            "estimated_cost_min_usd": round(self.cost_min_usd, 4),
            "estimated_cost_max_usd": round(self.cost_max_usd, 4),
        }
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"Dry-run cost estimate:\n"
            f"  Estimated LLM calls: {self.total_calls}\n"
            f"  Estimated prompt tokens: ~{self.prompt_tokens:,}\n"
            f"  Estimated completion tokens (max): ~{self.completion_tokens_max:,}\n"
            f"  Estimated cost: ${self.cost_min_usd:.4f} - ${self.cost_max_usd:.4f}"
        )


class DryRunEstimator:
    """Estimator for dry-run mode cost calculation.
    
    Accumulates prompt estimates and calculates total estimated costs
    without making actual LLM calls.
    """
    
    def __init__(self, model: str, max_tokens: int = 8192):
        """Initialize the estimator.
        
        Args:
            model: Model name for cost calculation
            max_tokens: Maximum tokens for completion (from LLM config)
        """
        self.model = model
        self.max_tokens = max_tokens
        self._estimates: List[Dict[str, int]] = []
        self._lock = threading.Lock()
    
    def add_call(
        self,
        messages: List[Dict[str, str]],
        call_type: str = "unknown"
    ) -> int:
        """Add an estimated call.
        
        Args:
            messages: Messages that would be sent to the LLM
            call_type: Type of call for categorization
            
        Returns:
            Estimated prompt token count
        """
        prompt_tokens = estimate_tokens(self.model, messages)
        
        with self._lock:
            self._estimates.append({
                "call_type": call_type,
                "prompt_tokens": prompt_tokens,
            })
        
        return prompt_tokens
    
    def get_estimate(self) -> DryRunEstimate:
        """Get the final dry-run estimate.
        
        Returns:
            DryRunEstimate with cost range
        """
        with self._lock:
            estimates = list(self._estimates)
        
        total_calls = len(estimates)
        total_prompt_tokens = sum(e["prompt_tokens"] for e in estimates)
        total_completion_max = total_calls * self.max_tokens
        
        # Minimum cost assumes ~10% of max completion tokens used
        completion_min = int(total_completion_max * 0.1)
        cost_min = estimate_cost(self.model, total_prompt_tokens, completion_min)
        cost_max = estimate_cost(self.model, total_prompt_tokens, total_completion_max)
        
        return DryRunEstimate(
            total_calls=total_calls,
            prompt_tokens=total_prompt_tokens,
            completion_tokens_max=total_completion_max,
            cost_min_usd=cost_min,
            cost_max_usd=cost_max,
        )
    
    def clear(self) -> None:
        """Clear all estimates."""
        with self._lock:
            self._estimates.clear()
