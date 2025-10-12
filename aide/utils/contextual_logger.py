"""
Contextual Logger for AIDE Agents

Provides a structured logging interface with automatic context injection,
performance tracking, and consistent formatting.

Usage:
    from aide.utils.contextual_logger import get_logger
    
    logger = get_logger("agent.codechain")
    
    # Set persistent context
    logger.set_context(agent_type="CodeChainAgent", run_id="exp_001")
    
    # Use scoped context
    with logger.context(step=5, operation="draft"):
        logger.info("Starting draft generation")
        logger.debug("Using model X")
"""

import logging
import time
from typing import Any, Dict, Optional
from contextlib import contextmanager

from .logging_context import LoggingContext, format_context_string


class ContextualLogger:
    """
    Wrapper around Python's logging.Logger that automatically injects context.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize with a standard Python logger.
        
        Args:
            logger: The underlying Python logger instance
        """
        self._logger = logger
        self._persistent_context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs) -> None:
        """
        Set persistent context for this logger.
        These values are added to LoggingContext for all log calls from this logger.
        """
        self._persistent_context.update(kwargs)
    
    def clear_context(self, key: Optional[str] = None) -> None:
        """Clear persistent context."""
        if key is None:
            self._persistent_context.clear()
        else:
            self._persistent_context.pop(key, None)
    
    @contextmanager
    def context(self, **kwargs):
        """
        Temporary context for a code block.
        
        Example:
            with logger.context(step=5, operation="draft"):
                logger.info("This has step and operation context")
        """
        # Combine persistent and temporary context
        combined = {**self._persistent_context, **kwargs}
        with LoggingContext.scope(**combined):
            yield
    
    def _merge_context(self, extra_context: Dict[str, Any]) -> Dict[str, Any]:
        """Merge persistent context with extra context from the call."""
        merged = self._persistent_context.copy()
        merged.update(extra_context)
        return merged
    
    def _log(self, level: int, msg: str, exc_info=None, **extra_context):
        """
        Internal log method that applies context.
        
        Args:
            level: Logging level (logging.INFO, etc.)
            msg: Log message
            exc_info: Exception info for errors
            **extra_context: Additional context for this specific log
        """
        # Apply merged context temporarily for this log call
        merged_context = self._merge_context(extra_context)
        
        with LoggingContext.scope(**merged_context):
            # Log with the underlying logger
            extra = {"extra_data": extra_context} if extra_context else {}
            self._logger.log(level, msg, exc_info=exc_info, extra=extra)
    
    def debug(self, msg: str, **context):
        """Log at DEBUG level."""
        self._log(logging.DEBUG, msg, **context)
    
    def info(self, msg: str, **context):
        """Log at INFO level."""
        self._log(logging.INFO, msg, **context)
    
    def warning(self, msg: str, **context):
        """Log at WARNING level."""
        self._log(logging.WARNING, msg, **context)
    
    def error(self, msg: str, exc_info=True, **context):
        """Log at ERROR level with optional exception info."""
        self._log(logging.ERROR, msg, exc_info=exc_info, **context)
    
    def critical(self, msg: str, exc_info=True, **context):
        """Log at CRITICAL level."""
        self._log(logging.CRITICAL, msg, exc_info=exc_info, **context)
    
    # Convenience methods for common operations
    
    def operation_start(self, operation: str, **context):
        """Log the start of an operation."""
        self.info(f"Starting {operation}", operation=operation, **context)
    
    def operation_end(self, operation: str, duration: float, success: bool = True, **context):
        """Log the end of an operation with timing."""
        status = "completed" if success else "failed"
        self.info(
            f"{operation.capitalize()} {status}",
            operation=operation,
            duration_ms=round(duration * 1000, 2),
            success=success,
            **context
        )
    
    def step_info(self, step: int, total: int, msg: str, **context):
        """Log with step context (common pattern in agents)."""
        self.info(msg, step=step, total_steps=total, **context)
    
    def llm_query(self, model: str, operation: str, **context):
        """Log an LLM query start."""
        self.debug(f"Querying LLM: {model}", model=model, operation=operation, **context)
    
    def code_execution(self, node_id: str, **context):
        """Log code execution."""
        self.debug("Executing code", node_id=node_id, **context)
    
    @contextmanager
    def operation(self, operation_name: str, **context):
        """
        Context manager that automatically logs operation start/end with timing.
        
        Example:
            with logger.operation("draft_code", step=5):
                # code here
                pass
            # Automatically logs: "draft_code completed in X.XX ms"
        """
        start_time = time.time()
        merged_context = {**context, "operation": operation_name}
        
        with self.context(**merged_context):
            self.operation_start(operation_name, **context)
            success = True
            try:
                yield
            except Exception:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                self.operation_end(operation_name, duration, success, **context)


# Module-level logger cache
_logger_cache: Dict[str, ContextualLogger] = {}


def get_logger(name: str = "aide") -> ContextualLogger:
    """
    Get or create a contextual logger.
    
    Args:
        name: Logger name (hierarchical with dots, e.g., "aide.agent.codechain")
    
    Returns:
        ContextualLogger instance
    
    Example:
        logger = get_logger("agent.baseline")
        logger.info("Agent initialized")
    """
    if name not in _logger_cache:
        underlying_logger = logging.getLogger(name)
        _logger_cache[name] = ContextualLogger(underlying_logger)
    
    return _logger_cache[name]


class ContextualLogFormatter(logging.Formatter):
    """
    Custom formatter that includes context in log output.
    
    Format: [timestamp] LEVEL [logger:context] message
    Example: [15:30:45] INFO [agent.CodeChain:step=5:op=draft] Starting draft
    """
    
    def format(self, record):
        """Format the log record with context."""
        # Get context string if available
        context_str = getattr(record, "log_context", "")
        
        # Build logger name with context
        if context_str:
            logger_part = f"{record.name}:{context_str}"
        else:
            logger_part = record.name
        
        # Create a copy of the record with modified name
        record = logging.makeLogRecord(record.__dict__)
        record.name = logger_part
        
        # Use parent formatter
        return super().format(record)

