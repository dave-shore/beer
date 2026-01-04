import time
import functools
from typing import Callable, Any


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

