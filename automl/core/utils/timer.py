import logging
import time
import uuid
from typing import Optional, Any

from automl.logger import get_logger

logger: logging.Logger = get_logger()


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 0:
        return "N/A"

    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        if remaining_seconds < 1:
            return f"{minutes}m"
        return f"{minutes}m {remaining_seconds:.0f}s"

    hours = minutes // 60
    remaining_minutes = minutes % 60

    if hours < 24:
        if remaining_minutes == 0:
            return f"{hours}h"
        return f"{hours}h {remaining_minutes}m"

    days = hours // 24
    remaining_hours = hours % 24

    if remaining_hours == 0:
        return f"{days}d"
    return f"{days}d {remaining_hours}h"


class Timer:
    """A context manager for timing code blocks with high precision and logging."""

    def __init__(self, name: str = "Block") -> None:
        self.name: str = name
        self.run_id: str = uuid.uuid4().hex[:8]
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self.execution_time: Optional[float] = None

    def __enter__(self) -> "Timer":
        logger.debug(f"[{self.run_id}] Starting code block '{self.name}'...")
        self._start_time = time.perf_counter()
        return self

    @property
    def formatted_execution_time(self):
        return format_duration(self.execution_time or 0)

    def __exit__(
        self, exc_type: type, exc_val: Exception, exc_tb: Any
    ) -> Optional[bool]:
        self._end_time = time.perf_counter()
        if self._start_time is not None:
            self.execution_time = self._end_time - self._start_time

        logger.debug(
            f"[{self.run_id}] Code block '{self.name}' finished in "
            f"{self.execution_time:.6f} seconds."
        )
        # Returning False ensures exceptions are not swallowed and still bubble up
        return False
