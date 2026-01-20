"""Tests for the metrics module."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from rubric_kit.metrics import (
    DryRunEstimate,
    DryRunEstimator,
    LLMCallMetrics,
    MetricsAggregator,
    MetricsSummary,
    estimate_cost,
    estimate_tokens,
)


class TestLLMCallMetrics:
    """Tests for LLMCallMetrics dataclass."""

    def test_basic_creation(self):
        """Test creating a basic LLMCallMetrics instance."""
        metrics = LLMCallMetrics(
            call_id="test-123",
            call_type="evaluate_criterion",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
            latency_seconds=1.5,
        )

        assert metrics.call_id == "test-123"
        assert metrics.call_type == "evaluate_criterion"
        assert metrics.model == "gpt-4"
        assert metrics.prompt_tokens == 100
        assert metrics.completion_tokens == 50
        assert metrics.total_tokens == 150
        assert metrics.cost_usd == 0.01
        assert metrics.latency_seconds == 1.5
        assert metrics.context_id is None

    def test_with_context_id(self):
        """Test creating metrics with context_id."""
        metrics = LLMCallMetrics(
            call_id="test-456",
            call_type="generate_dimensions",
            model="gpt-4o",
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            cost_usd=0.02,
            latency_seconds=2.0,
            context_id="rubric_001",
        )

        assert metrics.context_id == "rubric_001"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = LLMCallMetrics(
            call_id="test-789",
            call_type="evaluate_criterion",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.012345,
            latency_seconds=1.5678,
            context_id="criterion_1",
        )

        result = metrics.to_dict()

        assert result["call_id"] == "test-789"
        assert result["call_type"] == "evaluate_criterion"
        assert result["model"] == "gpt-4"
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["total_tokens"] == 150
        assert result["cost_usd"] == 0.012345  # Rounded to 6 decimals
        assert result["latency_seconds"] == 1.568  # Rounded to 3 decimals
        assert result["context_id"] == "criterion_1"
        assert "timestamp" in result


class TestMetricsSummary:
    """Tests for MetricsSummary dataclass."""

    def test_default_values(self):
        """Test default values."""
        summary = MetricsSummary()

        assert summary.total_calls == 0
        assert summary.prompt_tokens == 0
        assert summary.completion_tokens == 0
        assert summary.total_tokens == 0
        assert summary.cost_usd == 0.0
        assert summary.latency_seconds == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        summary = MetricsSummary(
            total_calls=5,
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            cost_usd=0.123456,
            latency_seconds=10.1234,
        )

        result = summary.to_dict()

        assert result["total_calls"] == 5
        assert result["prompt_tokens"] == 1000
        assert result["completion_tokens"] == 500
        assert result["total_tokens"] == 1500
        assert result["estimated_cost_usd"] == 0.123456
        assert result["total_time_seconds"] == 10.123


class TestMetricsAggregator:
    """Tests for MetricsAggregator class."""

    def test_empty_aggregator(self):
        """Test empty aggregator."""
        aggregator = MetricsAggregator()

        summary = aggregator.get_summary()
        assert summary.total_calls == 0
        assert summary.total_tokens == 0

        by_type = aggregator.get_by_type()
        assert by_type == {}

        calls = aggregator.get_calls()
        assert calls == []

    def test_record_call(self):
        """Test recording a call."""
        aggregator = MetricsAggregator()

        # Mock usage object
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150

        with patch("rubric_kit.metrics.litellm.completion_cost", return_value=0.01):
            metrics = aggregator.record_call(
                call_type="evaluate_criterion",
                model="gpt-4",
                usage=usage,
                latency=1.5,
                context_id="criterion_1",
            )

        assert metrics.prompt_tokens == 100
        assert metrics.completion_tokens == 50
        assert metrics.total_tokens == 150
        assert metrics.call_type == "evaluate_criterion"
        assert metrics.context_id == "criterion_1"

        summary = aggregator.get_summary()
        assert summary.total_calls == 1
        assert summary.prompt_tokens == 100

    def test_multiple_calls_aggregation(self):
        """Test aggregation of multiple calls."""
        aggregator = MetricsAggregator()

        usage1 = MagicMock()
        usage1.prompt_tokens = 100
        usage1.completion_tokens = 50
        usage1.total_tokens = 150

        usage2 = MagicMock()
        usage2.prompt_tokens = 200
        usage2.completion_tokens = 100
        usage2.total_tokens = 300

        with patch("rubric_kit.metrics.litellm.completion_cost", return_value=0.01):
            aggregator.record_call("type_a", "gpt-4", usage1, 1.0)
            aggregator.record_call("type_b", "gpt-4", usage2, 2.0)

        summary = aggregator.get_summary()
        assert summary.total_calls == 2
        assert summary.prompt_tokens == 300
        assert summary.completion_tokens == 150
        assert summary.total_tokens == 450
        assert summary.latency_seconds == 3.0

    def test_get_by_type(self):
        """Test aggregation by call type."""
        aggregator = MetricsAggregator()

        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150

        with patch("rubric_kit.metrics.litellm.completion_cost", return_value=0.01):
            aggregator.record_call("evaluate_criterion", "gpt-4", usage, 1.0)
            aggregator.record_call("evaluate_criterion", "gpt-4", usage, 1.0)
            aggregator.record_call("generate_dimensions", "gpt-4", usage, 2.0)

        by_type = aggregator.get_by_type()

        assert "evaluate_criterion" in by_type
        assert "generate_dimensions" in by_type
        assert by_type["evaluate_criterion"].total_calls == 2
        assert by_type["generate_dimensions"].total_calls == 1

    def test_get_by_context(self):
        """Test aggregation by context."""
        aggregator = MetricsAggregator()

        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150

        with patch("rubric_kit.metrics.litellm.completion_cost", return_value=0.01):
            aggregator.record_call("eval", "gpt-4", usage, 1.0, context_id="rubric_a")
            aggregator.record_call("eval", "gpt-4", usage, 1.0, context_id="rubric_a")
            aggregator.record_call("eval", "gpt-4", usage, 1.0, context_id="rubric_b")

        by_context = aggregator.get_by_context()

        assert "rubric_a" in by_context
        assert "rubric_b" in by_context
        assert by_context["rubric_a"].total_calls == 2
        assert by_context["rubric_b"].total_calls == 1

    def test_clear(self):
        """Test clearing the aggregator."""
        aggregator = MetricsAggregator()

        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150

        with patch("rubric_kit.metrics.litellm.completion_cost", return_value=0.01):
            aggregator.record_call("eval", "gpt-4", usage, 1.0)

        assert aggregator.get_summary().total_calls == 1

        aggregator.clear()

        assert aggregator.get_summary().total_calls == 0
        assert aggregator.get_calls() == []

    def test_to_dict_without_call_log(self):
        """Test to_dict without call log."""
        aggregator = MetricsAggregator(include_call_log=False)

        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150

        with patch("rubric_kit.metrics.litellm.completion_cost", return_value=0.01):
            aggregator.record_call("eval", "gpt-4", usage, 1.0)

        result = aggregator.to_dict()

        assert "summary" in result
        assert "by_type" in result
        assert "calls" not in result

    def test_to_dict_with_call_log(self):
        """Test to_dict with call log."""
        aggregator = MetricsAggregator(include_call_log=True)

        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150

        with patch("rubric_kit.metrics.litellm.completion_cost", return_value=0.01):
            aggregator.record_call("eval", "gpt-4", usage, 1.0)

        result = aggregator.to_dict()

        assert "summary" in result
        assert "by_type" in result
        assert "calls" in result
        assert len(result["calls"]) == 1

    def test_thread_safety(self):
        """Test thread safety of the aggregator."""
        aggregator = MetricsAggregator()
        num_threads = 10
        calls_per_thread = 100

        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        usage.total_tokens = 15

        def record_calls():
            for _ in range(calls_per_thread):
                with patch("rubric_kit.metrics.litellm.completion_cost", return_value=0.001):
                    aggregator.record_call("eval", "gpt-4", usage, 0.1)

        threads = [threading.Thread(target=record_calls) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        summary = aggregator.get_summary()
        assert summary.total_calls == num_threads * calls_per_thread

    def test_handle_missing_usage_attributes(self):
        """Test handling of missing usage attributes."""
        aggregator = MetricsAggregator()

        # Usage object with None values
        usage = MagicMock()
        usage.prompt_tokens = None
        usage.completion_tokens = None
        usage.total_tokens = None

        with patch("rubric_kit.metrics.litellm.completion_cost", return_value=0.01):
            metrics = aggregator.record_call("eval", "gpt-4", usage, 1.0)

        assert metrics.prompt_tokens == 0
        assert metrics.completion_tokens == 0
        assert metrics.total_tokens == 0


class TestDryRunEstimate:
    """Tests for DryRunEstimate dataclass."""

    def test_basic_creation(self):
        """Test creating a basic estimate."""
        estimate = DryRunEstimate(
            total_calls=10,
            prompt_tokens=5000,
            completion_tokens_max=8192,
            cost_min_usd=0.1,
            cost_max_usd=0.5,
        )

        assert estimate.total_calls == 10
        assert estimate.prompt_tokens == 5000
        assert estimate.completion_tokens_max == 8192
        assert estimate.cost_min_usd == 0.1
        assert estimate.cost_max_usd == 0.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        estimate = DryRunEstimate(
            total_calls=10,
            prompt_tokens=5000,
            completion_tokens_max=8192,
            cost_min_usd=0.12344,  # Use value that rounds cleanly
            cost_max_usd=0.56788,
        )

        result = estimate.to_dict()

        assert result["estimated_calls"] == 10
        assert result["estimated_prompt_tokens"] == 5000
        assert result["estimated_completion_tokens_max"] == 8192
        assert result["estimated_cost_min_usd"] == 0.1234
        assert result["estimated_cost_max_usd"] == 0.5679

    def test_str_representation(self):
        """Test string representation."""
        estimate = DryRunEstimate(
            total_calls=10,
            prompt_tokens=5000,
            completion_tokens_max=8192,
            cost_min_usd=0.10,
            cost_max_usd=0.50,
        )

        result = str(estimate)

        assert "Dry-run cost estimate:" in result
        assert "10" in result
        assert "5,000" in result
        assert "$0.10" in result


class TestDryRunEstimator:
    """Tests for DryRunEstimator class."""

    def test_basic_estimation(self):
        """Test basic cost estimation."""
        estimator = DryRunEstimator(model="gpt-4", max_tokens=1000)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"},
        ]

        with patch("rubric_kit.metrics.estimate_tokens", return_value=50):
            estimator.add_call(messages, call_type="test")

        estimate = estimator.get_estimate()

        assert estimate.total_calls == 1
        assert estimate.prompt_tokens == 50
        assert estimate.completion_tokens_max == 1000

    def test_multiple_calls(self):
        """Test estimation with multiple calls."""
        estimator = DryRunEstimator(model="gpt-4", max_tokens=500)

        messages = [{"role": "user", "content": "Test message"}]

        with patch("rubric_kit.metrics.estimate_tokens", return_value=10):
            estimator.add_call(messages, call_type="eval")
            estimator.add_call(messages, call_type="eval")
            estimator.add_call(messages, call_type="gen")

        estimate = estimator.get_estimate()

        assert estimate.total_calls == 3
        assert estimate.prompt_tokens == 30
        assert estimate.completion_tokens_max == 1500

    def test_clear(self):
        """Test clearing estimates."""
        estimator = DryRunEstimator(model="gpt-4", max_tokens=1000)

        messages = [{"role": "user", "content": "Test"}]

        with patch("rubric_kit.metrics.estimate_tokens", return_value=10):
            estimator.add_call(messages, call_type="test")

        assert estimator.get_estimate().total_calls == 1

        estimator.clear()

        assert estimator.get_estimate().total_calls == 0


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_with_litellm_success(self):
        """Test token estimation using LiteLLM."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        with patch("rubric_kit.metrics.litellm.token_counter", return_value=25):
            result = estimate_tokens("gpt-4", messages)

        assert result == 25

    def test_fallback_estimation(self):
        """Test fallback estimation when LiteLLM fails."""
        messages = [
            {"role": "system", "content": "X" * 400},  # ~100 tokens
            {"role": "user", "content": "Y" * 400},  # ~100 tokens
        ]

        with patch("rubric_kit.metrics.litellm.token_counter", side_effect=Exception("API error")):
            result = estimate_tokens("unknown-model", messages)

        # Fallback uses ~4 chars per token
        assert result == 200  # 800 chars / 4


class TestEstimateCost:
    """Tests for estimate_cost function."""

    def test_with_litellm_success(self):
        """Test cost estimation using LiteLLM."""
        # cost_per_token returns (prompt_cost, completion_cost)
        with patch("rubric_kit.metrics.litellm.cost_per_token", return_value=(0.03, 0.02)):
            result = estimate_cost("gpt-4", 1000, 500)

        assert result == 0.05  # 0.03 + 0.02

    def test_fallback_estimation(self):
        """Test fallback estimation when LiteLLM fails."""
        with patch("rubric_kit.metrics.litellm.cost_per_token", side_effect=Exception("API error")):
            result = estimate_cost("unknown-model", 1000, 500)

        # Fallback: (1000 * 0.00003) + (500 * 0.00006) = 0.03 + 0.03 = 0.06
        assert result == pytest.approx(0.06, abs=0.001)
