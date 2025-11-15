"""Judge execution strategies: sequential, parallel, and batched."""

from typing import List, Dict, Any, Callable, Optional, Literal
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from rubric_kit.schema import JudgeConfig, Criterion, Dimension


def execute_judges(
    judges: List[JudgeConfig],
    judge_function: Callable,
    execution_mode: Literal["sequential", "parallel", "batched"],
    criterion: Optional[Criterion] = None,
    chat_content: str = "",
    dimension: Optional[Dimension] = None,
    batch_size: int = 2,
    timeout: int = 30
) -> List[Dict[str, Any]]:
    """
    Execute judges using specified execution strategy.
    
    Args:
        judges: List of judge configurations
        judge_function: Function to call for each judge evaluation
                       Should have signature: (judge_config, criterion, chat_content, dimension) -> Dict
        execution_mode: Execution strategy ("sequential", "parallel", "batched")
        criterion: Criterion to evaluate (optional, passed to judge_function)
        chat_content: Chat session content to evaluate
        dimension: Dimension for score-based criteria (optional)
        batch_size: Batch size for batched mode
        timeout: Timeout per judge call in seconds
        
    Returns:
        List of evaluation results, one per judge, in same order as judges list.
        Each result has at minimum: {"judge": judge_name, ...}
        On error, result will be: {"judge": judge_name, "error": error_message}
        
    Raises:
        ValueError: If judges list is empty or execution_mode is invalid
    """
    if not judges:
        raise ValueError("No judges provided")
    
    if execution_mode == "sequential":
        return _execute_sequential(judges, judge_function, criterion, chat_content, dimension, timeout)
    elif execution_mode == "parallel":
        return _execute_parallel(judges, judge_function, criterion, chat_content, dimension, timeout)
    elif execution_mode == "batched":
        return _execute_batched(judges, judge_function, criterion, chat_content, dimension, batch_size, timeout)
    else:
        raise ValueError(f"Invalid execution mode: {execution_mode}")


def _execute_sequential(
    judges: List[JudgeConfig],
    judge_function: Callable,
    criterion: Optional[Criterion],
    chat_content: str,
    dimension: Optional[Dimension],
    timeout: int
) -> List[Dict[str, Any]]:
    """Execute judges one by one in sequence."""
    results = []
    
    for judge in judges:
        try:
            # Call judge function with timeout
            result = _call_judge_with_timeout(
                judge_function,
                judge,
                criterion,
                chat_content,
                dimension,
                timeout
            )
            # Add judge name to result
            result["judge"] = judge.name
            results.append(result)
        except Exception as e:
            # On error, add error result
            results.append({
                "judge": judge.name,
                "error": f"Judge evaluation failed: {str(e)}"
            })
    
    return results


def _execute_parallel(
    judges: List[JudgeConfig],
    judge_function: Callable,
    criterion: Optional[Criterion],
    chat_content: str,
    dimension: Optional[Dimension],
    timeout: int
) -> List[Dict[str, Any]]:
    """Execute all judges in parallel using asyncio."""
    # Run async execution in event loop
    return asyncio.run(_execute_parallel_async(
        judges,
        judge_function,
        criterion,
        chat_content,
        dimension,
        timeout
    ))


async def _execute_parallel_async(
    judges: List[JudgeConfig],
    judge_function: Callable,
    criterion: Optional[Criterion],
    chat_content: str,
    dimension: Optional[Dimension],
    timeout: int
) -> List[Dict[str, Any]]:
    """Async helper for parallel execution."""
    loop = asyncio.get_event_loop()
    
    # Create tasks for all judges
    tasks = []
    for judge in judges:
        task = loop.run_in_executor(
            None,
            _call_judge_safe,
            judge_function,
            judge,
            criterion,
            chat_content,
            dimension,
            timeout
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    return results


def _execute_batched(
    judges: List[JudgeConfig],
    judge_function: Callable,
    criterion: Optional[Criterion],
    chat_content: str,
    dimension: Optional[Dimension],
    batch_size: int,
    timeout: int
) -> List[Dict[str, Any]]:
    """Execute judges in batches."""
    results = []
    
    # Process judges in batches
    for i in range(0, len(judges), batch_size):
        batch = judges[i:i + batch_size]
        
        # Execute this batch in parallel
        batch_results = asyncio.run(_execute_parallel_async(
            batch,
            judge_function,
            criterion,
            chat_content,
            dimension,
            timeout
        ))
        
        results.extend(batch_results)
    
    return results


def _call_judge_with_timeout(
    judge_function: Callable,
    judge: JudgeConfig,
    criterion: Optional[Criterion],
    chat_content: str,
    dimension: Optional[Dimension],
    timeout: int
) -> Dict[str, Any]:
    """Call judge function with timeout using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            judge_function,
            judge,
            criterion,
            chat_content,
            dimension
        )
        try:
            result = future.result(timeout=timeout)
            return result
        except FuturesTimeoutError:
            raise TimeoutError(f"Judge {judge.name} evaluation timed out after {timeout}s")


def _call_judge_safe(
    judge_function: Callable,
    judge: JudgeConfig,
    criterion: Optional[Criterion],
    chat_content: str,
    dimension: Optional[Dimension],
    timeout: int
) -> Dict[str, Any]:
    """Safely call judge function, catching errors."""
    try:
        result = _call_judge_with_timeout(
            judge_function,
            judge,
            criterion,
            chat_content,
            dimension,
            timeout
        )
        # Add judge name to result
        result["judge"] = judge.name
        return result
    except Exception as e:
        # Return error result
        return {
            "judge": judge.name,
            "error": f"Judge evaluation failed: {str(e)}"
        }

