"""
Centralized Error Handling Module

CRITICAL PRINCIPLE: Never silently fail
- Always log errors with full stack traces
- Provide context for debugging
- Handle errors appropriately at each layer
- Use custom exceptions for business logic errors
"""

import traceback
import logging
from typing import Optional, Dict, Any, Type
from functools import wraps
from datetime import datetime

# Configure module logger
logger = logging.getLogger(__name__)


class FraudDetectionError(Exception):
    """
    Base exception for all fraud detection system errors.
    
    Attributes:
        message: Human-readable error description
        context: Additional context for debugging
        timestamp: When the error occurred
        error_code: Optional error code for categorization
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.timestamp = datetime.utcnow().isoformat()
        self.error_code = error_code or "FRAUD_DETECTION_ERROR"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp,
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class DeviceFingerprintError(FraudDetectionError):
    """
    Raised when device fingerprinting fails.
    
    Common causes:
    - Missing critical device attributes
    - Network context collection failure
    - Fingerprint generation timeout
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        partial_attributes: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, context, error_code="DEVICE_FINGERPRINT_ERROR")
        self.partial_attributes = partial_attributes or {}


class PaymentValidationError(FraudDetectionError):
    """
    Raised when payment source validation fails.
    
    Common causes:
    - Device not recognized
    - Location mismatch
    - Transaction pattern anomaly
    - Authentication failure
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        validation_step: Optional[str] = None
    ):
        super().__init__(message, context, error_code="PAYMENT_VALIDATION_ERROR")
        self.validation_step = validation_step


class MLModelError(FraudDetectionError):
    """
    Raised when ML model inference fails.
    
    Common causes:
    - Model not loaded
    - Feature preprocessing failure
    - Inference timeout
    - Model version mismatch
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None
    ):
        super().__init__(message, context, error_code="ML_MODEL_ERROR")
        self.model_name = model_name
        self.model_version = model_version


class ConfigurationError(FraudDetectionError):
    """
    Raised when configuration is invalid or missing.
    
    Common causes:
    - Missing required environment variables
    - Invalid configuration values
    - Configuration file parsing errors
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        config_key: Optional[str] = None
    ):
        super().__init__(message, context, error_code="CONFIGURATION_ERROR")
        self.config_key = config_key


class RateLimitError(FraudDetectionError):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        context: Optional[Dict[str, Any]] = None,
        retry_after: Optional[int] = None
    ):
        super().__init__(message, context, error_code="RATE_LIMIT_ERROR")
        self.retry_after = retry_after


class ServiceUnavailableError(FraudDetectionError):
    """Raised when a required service is unavailable."""
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        service_name: Optional[str] = None
    ):
        super().__init__(message, context, error_code="SERVICE_UNAVAILABLE")
        self.service_name = service_name


def handle_error(
    error: Exception,
    context: Dict[str, Any],
    reraise: bool = True,
    log_level: int = logging.ERROR
) -> None:
    """
    Centralized error handling with comprehensive logging.
    
    Args:
        error: Exception that occurred
        context: Additional context for debugging
        reraise: Whether to re-raise the exception after logging
        log_level: Logging level (default: ERROR)
    
    Example:
        try:
            process_transaction(transaction)
        except DeviceFingerprintError as e:
            handle_error(e, {"transaction_id": transaction.id}, reraise=False)
            return fallback_result()
    """
    # Build comprehensive error context
    error_details = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stack_trace": traceback.format_exc(),
        "timestamp": datetime.utcnow().isoformat(),
        **context,
    }
    
    # Add FraudDetectionError-specific context
    if isinstance(error, FraudDetectionError):
        error_details["error_code"] = error.error_code
        error_details["error_context"] = error.context
    
    # Log with full context
    logger.log(
        log_level,
        f"Error occurred: {type(error).__name__}: {str(error)}",
        extra={"error_details": error_details},
        exc_info=True
    )
    
    # TODO: Send to error tracking service (Sentry, Rollbar)
    # sentry_sdk.capture_exception(error)
    
    if reraise:
        raise


def log_and_raise(
    exception_class: Type[FraudDetectionError],
    message: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Create, log, and raise an exception in one call.
    
    Args:
        exception_class: The exception class to raise
        message: Error message
        context: Additional context
        **kwargs: Additional arguments for the exception
    
    Example:
        log_and_raise(
            DeviceFingerprintError,
            "Failed to collect device attributes",
            context={"user_id": user_id},
            partial_attributes=collected
        )
    """
    error = exception_class(message, context=context, **kwargs)
    
    logger.error(
        f"{exception_class.__name__}: {message}",
        extra={
            "error_details": error.to_dict(),
            **kwargs
        },
        exc_info=True
    )
    
    raise error


def error_boundary(
    fallback_value: Any = None,
    exceptions: tuple = (Exception,),
    log_level: int = logging.ERROR
):
    """
    Decorator that wraps function in error boundary.
    
    Args:
        fallback_value: Value to return on error (or callable)
        exceptions: Tuple of exception types to catch
        log_level: Logging level for caught exceptions
    
    Example:
        @error_boundary(fallback_value={"risk_score": 1.0}, exceptions=(MLModelError,))
        def predict_fraud(features):
            return model.predict(features)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                handle_error(
                    e,
                    context={"function": func.__name__, "args": str(args)[:200]},
                    reraise=False,
                    log_level=log_level
                )
                
                if callable(fallback_value):
                    return fallback_value()
                return fallback_value
        
        return wrapper
    return decorator


def validate_or_raise(
    condition: bool,
    exception_class: Type[FraudDetectionError],
    message: str,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Validate a condition and raise if false.
    
    Args:
        condition: Boolean condition to check
        exception_class: Exception to raise if condition is False
        message: Error message
        context: Additional context
    
    Example:
        validate_or_raise(
            user_id is not None,
            PaymentValidationError,
            "User ID is required",
            context={"request_id": request_id}
        )
    """
    if not condition:
        log_and_raise(exception_class, message, context)
