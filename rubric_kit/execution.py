"""Judge execution strategies: sequential, parallel, and batched."""

from typing import List, Dict, Any, Callable, Optional, Literal, TYPE_CHECKING
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from rubric_kit.schema import JudgeConfig, Criterion, Dimension

if TYPE_CHECKING:
    from rubric_kit.metrics import MetricsAggregator


def execute_judges(
    judges: List[JudgeConfig],
    judge_function: Callable,
    execution_mode: Literal["sequential", "parallel", "batched"],
    criterion: Optional[Criterion] = None,
    chat_content: str = "",
    dimension: Optional[Dimension] = None,
    parsed_session: Optional[Any] = None,
    batch_size: int = 2,
    timeout: int = 30,
    metrics: Optional["MetricsAggregator"] = None
) -> List[Dict[str, Any]]:
    """
    Execute judges using specified execution strategy.
    
    Args:
        judges: List of judge configurations
        judge_function: Function to call for each judge evaluation
                       Should have signature: (judge_config, criterion, chat_content, dimension, parsed_session, metrics) -> Dict
        execution_mode: Execution strategy ("sequential", "parallel", "batched")
        criterion: Criterion to evaluate (optional, passed to judge_function)
        chat_content: Chat session content to evaluate
        dimension: Dimension for score-based criteria (optional)
        parsed_session: Optional pre-parsed chat session (optional, passed to judge_function)
        batch_size: Batch size for batched mode
        timeout: Timeout per judge call in seconds
        metrics: Optional MetricsAggregator for tracking LLM calls
        
    Returns:
        List of evaluation results, one per judge, in same order as judges list.
        Each result has at minimum: {"judge": judge_name, ...}
        On error, result will be: {"judge": judge_name, "error": error_message}
        
    Raises:
        ValueError: If judges list is empty or execution_mode is invalid
    """
    if not judges:
        raise ValueError("No judges provided")
    
    execution_strategies = {
        "sequential": lambda: _execute_sequential(
            judges, judge_function, criterion, chat_content, dimension, parsed_session, timeout, metrics
        ),
        "parallel": lambda: _execute_parallel(
            judges, judge_function, criterion, chat_content, dimension, parsed_session, timeout, metrics
        ),
        "batched": lambda: _execute_batched(
            judges, judge_function, criterion, chat_content, dimension, parsed_session, batch_size, timeout, metrics
        ),
    }
    
    strategy = execution_strategies.get(execution_mode)
    if strategy is None:
        raise ValueError(f"Invalid execution mode: {execution_mode}")
    
    return strategy()


def _create_error_result(judge_name: str, error_message: str) -> Dict[str, Any]:
    """Create a standardized error result for a judge evaluation."""
    return {
        "judge": judge_name,
        "error": f"Judge evaluation failed: {error_message}"
    }


def _create_success_result(judge_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Add judge name to successful result."""
    result["judge"] = judge_name
    return result


def _execute_sequential(
    judges: List[JudgeConfig],
    judge_function: Callable,
    criterion: Optional[Criterion],
    chat_content: str,
    dimension: Optional[Dimension],
    parsed_session: Optional[Any],
    timeout: int,
    metrics: Optional["MetricsAggregator"] = None
) -> List[Dict[str, Any]]:
    """Execute judges one by one in sequence."""
    results = []
    
    for judge in judges:
        result = _call_judge_safe(
            judge_function, judge, criterion, chat_content, dimension, parsed_session, timeout, metrics
        )
        results.append(result)
    
    return results


def _execute_parallel(
    judges: List[JudgeConfig],
    judge_function: Callable,
    criterion: Optional[Criterion],
    chat_content: str,
    dimension: Optional[Dimension],
    parsed_session: Optional[Any],
    timeout: int,
    metrics: Optional["MetricsAggregator"] = None
) -> List[Dict[str, Any]]:
    """Execute all judges in parallel using asyncio."""
    return asyncio.run(_execute_parallel_async(
        judges, judge_function, criterion, chat_content, dimension, parsed_session, timeout, metrics
    ))


async def _execute_parallel_async(
    judges: List[JudgeConfig],
    judge_function: Callable,
    criterion: Optional[Criterion],
    chat_content: str,
    dimension: Optional[Dimension],
    parsed_session: Optional[Any],
    timeout: int,
    metrics: Optional["MetricsAggregator"] = None
) -> List[Dict[str, Any]]:
    """Async helper for parallel execution."""
    loop = asyncio.get_event_loop()
    
    tasks = [
        loop.run_in_executor(
            None,
            _call_judge_safe,
            judge_function,
            judge,
            criterion,
            chat_content,
            dimension,
            parsed_session,
            timeout,
            metrics
        )
        for judge in judges
    ]
    
    return await asyncio.gather(*tasks)


def _execute_batched(
    judges: List[JudgeConfig],
    judge_function: Callable,
    criterion: Optional[Criterion],
    chat_content: str,
    dimension: Optional[Dimension],
    parsed_session: Optional[Any],
    batch_size: int,
    timeout: int,
    metrics: Optional["MetricsAggregator"] = None
) -> List[Dict[str, Any]]:
    """Execute judges in batches."""
    results = []
    
    for i in range(0, len(judges), batch_size):
        batch = judges[i:i + batch_size]
        batch_results = asyncio.run(_execute_parallel_async(
            batch, judge_function, criterion, chat_content, dimension, parsed_session, timeout, metrics
        ))
        results.extend(batch_results)
    
    return results


def _call_judge_with_timeout(
    judge_function: Callable,
    judge: JudgeConfig,
    criterion: Optional[Criterion],
    chat_content: str,
    dimension: Optional[Dimension],
    parsed_session: Optional[Any],
    timeout: int,
    metrics: Optional["MetricsAggregator"] = None
) -> Dict[str, Any]:
    """Call judge function with timeout using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            judge_function, judge, criterion, chat_content, dimension, parsed_session, metrics
        )
        
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            raise TimeoutError(f"Judge {judge.name} evaluation timed out after {timeout}s")


def _call_judge_safe(
    judge_function: Callable,
    judge: JudgeConfig,
    criterion: Optional[Criterion],
    chat_content: str,
    dimension: Optional[Dimension],
    parsed_session: Optional[Any],
    timeout: int,
    metrics: Optional["MetricsAggregator"] = None
) -> Dict[str, Any]:
    """Safely call judge function, catching all errors and returning standardized result."""
    try:
        result = _call_judge_with_timeout(
            judge_function, judge, criterion, chat_content, dimension, parsed_session, timeout, metrics
        )
        return _create_success_result(judge.name, result)
    except Exception as e:
        return _create_error_result(judge.name, str(e))

