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

def _find_tokens(s, query, encoding, min_match_ratio: float = 0.8):
    """Return (token_start, token_end_exclusive) spans where `query` occurs in `s`.

    Exact (case-insensitive) occurrences are preferred. If there is none, fall
    back to the single largest SequenceMatcher block, accepted only when it
    covers at least `min_match_ratio` of `query`. The previous implementation
    emitted one span per matching block (including single-character blocks),
    which flooded the gold spans with spurious partial matches.
    """
    result = []
    if not query:
        return result

    lowered_s, lowered_q = s.lower(), query.lower()
    start = lowered_s.find(lowered_q)
    while start != -1:
        ts = encoding.char_to_token(start)
        te = encoding.char_to_token(start + len(query) - 1)
        if ts is not None and te is not None:
            result.append((ts, te + 1))
        start = lowered_s.find(lowered_q, start + 1)

    if result:
        return sorted(set(result))

    match = SequenceMatcher(None, s, query).find_longest_match(
        0, len(s), 0, len(query)
    )
    if match.size >= max(1, int(min_match_ratio * len(query))):
        ts = encoding.char_to_token(match.a)
        te = encoding.char_to_token(match.a + match.size - 1)
        if ts is not None and te is not None:
            result.append((ts, te + 1))
    return result