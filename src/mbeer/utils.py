import time
import functools
from typing import Callable, Any, Iterable


def timing_decorator(func: Callable) -> Callable:
    """
    A decorator that measures and prints the execution time of a function.
    
    Args:
        func: The function to be timed
        
    Returns:
        The wrapped function with timing functionality
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper


def batch_generator(iterable: Iterable, batch_size: int) -> Iterable:
    """
    A generator that yields batches of a given size from an iterable.
    
    Args:
        iterable: The iterable to be batched
        batch_size: The size of the batches
        
    Returns:
        A generator that yields batches of the given size
    """
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i+batch_size]