import time
import functools
from typing import Callable, Any, Iterable
from difflib import SequenceMatcher


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

def _find_tokens(s, query, encoding):
        result = []
        for match in SequenceMatcher(None, s, query).get_matching_blocks():
            char_start = match.a
            char_end = match.a + match.size
            ts = encoding.char_to_token(char_start)
            if ts is None:
                continue
            te = encoding.char_to_token(char_end - 1)
            if te is None:
                continue
            result.append((ts, te + 1))
        return result