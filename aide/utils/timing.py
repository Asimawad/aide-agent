"""
Performance Timing Utilities for AIDE Agents

Provides decorators and context managers for automatic performance tracking
and logging.

Usage:
    from aide.utils.timing import Timer, log_timing
    
    # Context manager
    with Timer("code_execution") as t:
        execute_code()
    # Automatically logs: "code_execution: 1.23s"
    
    # Decorator
    @log_timing(operation="llm_query")
    def query_llm(...):
        pass
    # Automatically logs timing on every call
"""

import time
import functools
from typing import Any, Callable, Optional
from contextlib import contextmanager

from .contextual_logger import get_logger


class Timer:
    """
    Context manager for timing operations with automatic logging.
    
    Example:
        with Timer("training", logger=my_logger) as t:
            train_model()
        print(f"Took {t.duration:.2f} seconds")
    """
    
    def __init__(
        self,
        operation_name: str,
        logger: Optional[Any] = None,
        log_level: str = "debug",
        auto_log: bool = True,
        **context
    ):
        """
        Initialize timer.
        
        Args:
            operation_name: Name of the operation being timed
            logger: Optional ContextualLogger instance (or None to use default)
            log_level: Level to log at ("debug", "info", etc.)
            auto_log: If True, automatically log on exit
            **context: Additional context to include in logs
        """
        self.operation_name = operation_name
        self.logger = logger or get_logger("aide.timing")
        self.log_level = log_level
        self.auto_log = auto_log
        self.context = context
        
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.success: bool = True
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and optionally log."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = exc_type is None
        
        if self.auto_log:
            self._log_result()
        
        return False  # Don't suppress exceptions
    
    def _log_result(self):
        """Log the timing result."""
        log_func = getattr(self.logger, self.log_level, self.logger.info)
        
        status = "completed" if self.success else "failed"
        duration_ms = round(self.duration * 1000, 2)
        
        log_func(
            f"{self.operation_name} {status}",
            operation=self.operation_name,
            duration_ms=duration_ms,
            duration_sec=round(self.duration, 3),
            success=self.success,
            **self.context
        )
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.duration is None:
            return 0.0
        return self.duration * 1000


def log_timing(
    operation: Optional[str] = None,
    logger: Optional[Any] = None,
    log_level: str = "debug",
    include_args: bool = False,
):
    """
    Decorator that automatically times and logs function calls.
    
    Args:
        operation: Operation name (defaults to function name)
        logger: Optional ContextualLogger instance
        log_level: Level to log at ("debug", "info", etc.)
        include_args: If True, log function arguments
    
    Example:
        @log_timing(operation="llm_query")
        def query_model(prompt: str) -> str:
            return model.generate(prompt)
        
        # Automatically logs:
        # [DEBUG] llm_query completed | duration_ms=1234.56 success=True
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__
        func_logger = logger or get_logger("aide.timing")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build context
            context = {"function": func.__name__}
            if include_args:
                context["args"] = str(args)[:100]  # Truncate long args
                context["kwargs"] = str(kwargs)[:100]
            
            with Timer(op_name, logger=func_logger, log_level=log_level, **context):
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


class PerformanceTracker:
    """
    Accumulates timing statistics for repeated operations.
    
    Example:
        tracker = PerformanceTracker()
        
        for i in range(100):
            with tracker.measure("iteration"):
                do_work()
        
        tracker.print_stats()
        # Shows: min, max, mean, total time for "iteration"
    """
    
    def __init__(self, logger: Optional[Any] = None):
        """Initialize tracker."""
        self.logger = logger or get_logger("aide.performance")
        self.timings: dict[str, list[float]] = {}
    
    @contextmanager
    def measure(self, operation: str):
        """Context manager to measure an operation."""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            if operation not in self.timings:
                self.timings[operation] = []
            self.timings[operation].append(duration)
    
    def get_stats(self, operation: str) -> dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.timings or not self.timings[operation]:
            return {}
        
        times = self.timings[operation]
        return {
            "count": len(times),
            "total": sum(times),
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }
    
    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_stats(op) for op in self.timings}
    
    def print_stats(self):
        """Print statistics for all operations."""
        for operation, stats in self.get_all_stats().items():
            if not stats:
                continue
            
            self.logger.info(
                f"{operation} stats",
                operation=operation,
                count=stats["count"],
                total_sec=round(stats["total"], 2),
                mean_ms=round(stats["mean"] * 1000, 2),
                min_ms=round(stats["min"] * 1000, 2),
                max_ms=round(stats["max"] * 1000, 2),
            )
    
    def reset(self, operation: Optional[str] = None):
        """Reset timing data."""
        if operation is None:
            self.timings.clear()
        else:
            self.timings.pop(operation, None)


@contextmanager
def timed_operation(operation_name: str, logger: Optional[Any] = None, **context):
    """
    Convenience context manager combining Timer with logging context.
    
    Example:
        with timed_operation("draft_code", step=5):
            code = generate_code()
        # Logs with both timing and step context
    """
    from .logging_context import LoggingContext
    
    timer_logger = logger or get_logger("aide.timing")
    
    with LoggingContext.scope(**context):
        with Timer(operation_name, logger=timer_logger, **context):
            yield

