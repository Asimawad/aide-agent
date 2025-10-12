"""
Logging Context Manager for AIDE Agents

Provides thread-local context storage that automatically injects 
contextual information into all log records.

Usage:
    from aide.utils.logging_context import LoggingContext
    
    # Set global context
    LoggingContext.set("agent_type", "CodeChainAgent")
    
    # Use context manager for scoped context
    with LoggingContext(step=5, operation="draft"):
        logger.info("This log has step and operation context")
"""

import threading
from typing import Any, Dict, Optional
from contextlib import contextmanager


class LoggingContext:
    """Thread-local storage for logging context."""
    
    _local = threading.local()
    
    @classmethod
    def _get_context(cls) -> Dict[str, Any]:
        """Get the current context dictionary."""
        if not hasattr(cls._local, "context"):
            cls._local.context = {}
        return cls._local.context
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a value from the current context."""
        return cls._get_context().get(key, default)
    
    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get all context values as a dict."""
        return cls._get_context().copy()
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set a context value that persists until explicitly cleared."""
        cls._get_context()[key] = value
    
    @classmethod
    def update(cls, **kwargs) -> None:
        """Update multiple context values at once."""
        cls._get_context().update(kwargs)
    
    @classmethod
    def clear(cls, key: Optional[str] = None) -> None:
        """Clear a specific key or all context if key is None."""
        if key is None:
            cls._get_context().clear()
        else:
            cls._get_context().pop(key, None)
    
    @classmethod
    @contextmanager
    def scope(cls, **kwargs):
        """
        Context manager for temporary context values.
        
        Example:
            with LoggingContext.scope(step=5, operation="draft"):
                logger.info("This has step and operation context")
            # step and operation are automatically removed
        """
        # Save previous values
        previous = {}
        context = cls._get_context()
        
        for key, value in kwargs.items():
            if key in context:
                previous[key] = context[key]
            context[key] = value
        
        try:
            yield
        finally:
            # Restore previous values or remove if didn't exist
            for key in kwargs:
                if key in previous:
                    context[key] = previous[key]
                else:
                    context.pop(key, None)
    
    def __init__(self, **kwargs):
        """Initialize context manager with keyword arguments."""
        self.kwargs = kwargs
    
    def __enter__(self):
        """Enter context manager."""
        self._cm = self.scope(**self.kwargs)
        return self._cm.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        return self._cm.__exit__(exc_type, exc_val, exc_tb)


class LoggingContextFilter:
    """
    Logging filter that injects context into log records.
    
    Usage:
        logger = logging.getLogger("aide")
        logger.addFilter(LoggingContextFilter())
    """
    
    def filter(self, record):
        """Add context to the log record."""
        context = LoggingContext.get_all()
        
        # Add all context as record attributes
        for key, value in context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        
        # Create a formatted context string for easy display
        if context:
            context_parts = [f"{k}={v}" for k, v in context.items()]
            record.log_context = " | ".join(context_parts)
        else:
            record.log_context = ""
        
        return True


def format_context_string(prefix: str = "", separator: str = ":") -> str:
    """
    Format current context as a string for log prefixes.
    
    Args:
        prefix: Optional prefix before context
        separator: Separator between context items
    
    Returns:
        Formatted context string like "agent=CodeChain:step=5:op=draft"
    """
    context = LoggingContext.get_all()
    if not context:
        return prefix
    
    parts = [f"{k}={v}" for k, v in context.items()]
    context_str = separator.join(parts)
    
    if prefix:
        return f"{prefix}{separator}{context_str}"
    return context_str

