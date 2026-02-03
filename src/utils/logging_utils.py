"""
Structured Logging Utilities

Provides:
- JSON-formatted structured logging
- Correlation ID propagation
- Request context tracking
- Integration with OpenTelemetry
"""

import logging
import json
import sys
import uuid
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from functools import wraps
from contextvars import ContextVar

# Context variable for request correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
request_context_var: ContextVar[Dict[str, Any]] = ContextVar("request_context", default={})


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs JSON-structured log entries.
    
    Each log entry includes:
    - timestamp: ISO 8601 formatted time
    - level: Log level name
    - logger: Logger name
    - message: Log message
    - correlation_id: Request correlation ID (if set)
    - Additional context fields
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if present
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id
        
        # Add request context if present
        request_context = request_context_var.get()
        if request_context:
            log_entry["request_context"] = request_context
        
        # Add extra fields from record
        if hasattr(record, "error_details"):
            log_entry["error_details"] = record.error_details
        
        # Add any additional extra attributes
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "error_details"
            ):
                try:
                    json.dumps(value)  # Check if serializable
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for development environments.
    """
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        correlation_id = correlation_id_var.get()
        
        prefix = f"[{correlation_id[:8]}] " if correlation_id else ""
        
        formatted = (
            f"{color}{record.levelname:8}{self.RESET} "
            f"{datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]} "
            f"{prefix}"
            f"{record.name}:{record.funcName}:{record.lineno} - "
            f"{record.getMessage()}"
        )
        
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        
        return formatted


def setup_structured_logging(
    level: str = "INFO",
    use_json: bool = True,
    log_file: Optional[str] = None,
    service_name: str = "fraud-detection"
) -> logging.Logger:
    """
    Configure structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_json: Whether to use JSON format (True for production)
        log_file: Optional file path for logging
        service_name: Service name for log identification
    
    Returns:
        Configured root logger
    
    Example:
        logger = setup_structured_logging(
            level="INFO",
            use_json=True,
            log_file="logs/fraud_detection.log"
        )
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Choose formatter based on environment
    if use_json:
        formatter = StructuredFormatter()
    else:
        formatter = HumanReadableFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())  # Always JSON for files
        root_logger.addHandler(file_handler)
    
    # Add service name to all logs
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.service = service_name
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation ID for the current context.
    
    Args:
        correlation_id: ID to set (generates UUID if not provided)
    
    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return correlation_id_var.get()


def set_request_context(context: Dict[str, Any]) -> None:
    """
    Set additional request context for logging.
    
    Args:
        context: Dictionary of context values
    """
    current = request_context_var.get()
    request_context_var.set({**current, **context})


def clear_request_context() -> None:
    """Clear the request context."""
    request_context_var.set({})
    correlation_id_var.set(None)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **kwargs
) -> None:
    """
    Log a message with additional context fields.
    
    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        **kwargs: Additional context to include
    """
    logger.log(level, message, extra=kwargs)


class CorrelationIdMiddleware:
    """
    ASGI middleware for correlation ID propagation.
    
    Extracts correlation ID from X-Correlation-ID header or generates new one.
    Adds correlation ID to response headers.
    
    Example:
        app = FastAPI()
        app.add_middleware(CorrelationIdMiddleware)
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Extract or generate correlation ID
        headers = dict(scope.get("headers", []))
        correlation_id = headers.get(
            b"x-correlation-id",
            headers.get(b"x-request-id")
        )
        
        if correlation_id:
            correlation_id = correlation_id.decode("utf-8")
        else:
            correlation_id = str(uuid.uuid4())
        
        # Set in context
        set_correlation_id(correlation_id)
        
        # Extract request context
        set_request_context({
            "method": scope.get("method"),
            "path": scope.get("path"),
            "client": scope.get("client"),
        })
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-correlation-id", correlation_id.encode()))
                message["headers"] = headers
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            clear_request_context()


def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function entry and exit.
    
    Args:
        logger: Logger to use (uses function's module logger if not provided)
    
    Example:
        @log_function_call()
        def process_transaction(transaction_id: str):
            ...
    """
    def decorator(func: Callable):
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_name = func.__qualname__
            logger.debug(f"Entering {func_name}", extra={"args": str(args)[:200]})
            
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Exiting {func_name} successfully")
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {e}", exc_info=True)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_name = func.__qualname__
            logger.debug(f"Entering {func_name}", extra={"args": str(args)[:200]})
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func_name} successfully")
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {e}", exc_info=True)
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
