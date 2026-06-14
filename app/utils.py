import logging
import random
import time
from functools import wraps

from .config import retry_settings

logger = logging.getLogger(__name__)


def retry(
    retries: int = retry_settings.max_attempts,
    initial_delay: float = retry_settings.initial_delay,
):
    """
    Decorator to retry a function with Exponential Backoff.
    Formula: delay = initial_delay * (2 ** attempt) + jitter
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < retries - 1:
                        # Calculate exponential backoff with jitter
                        delay = initial_delay * (2**attempt) + random.uniform(0, 1)
                        logger.warning(
                            "Attempt %d/%d failed for %s. Retrying in %.2fs... Error: %s",
                            attempt + 1,
                            retries,
                            func.__name__,
                            delay,
                            e,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"Function {func.__name__} failed after {retries} attempts.")
                        raise

        return wrapper

    return decorator
