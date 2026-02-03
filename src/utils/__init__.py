"""
Utils Layer - Generic utilities reusable across any project.
These have zero business logic and can be extracted to SDK for reuse.
"""

from .error_handling import (
    FraudDetectionError,
    DeviceFingerprintError,
    PaymentValidationError,
    MLModelError,
    ConfigurationError,
    handle_error,
    log_and_raise,
)
from .logging_utils import (
    setup_structured_logging,
    get_logger,
    log_with_context,
    CorrelationIdMiddleware,
)
from .data_utils import (
    safe_json_serialize,
    parse_datetime,
    calculate_hash,
    validate_required_fields,
)

__all__ = [
    # Error handling
    "FraudDetectionError",
    "DeviceFingerprintError", 
    "PaymentValidationError",
    "MLModelError",
    "ConfigurationError",
    "handle_error",
    "log_and_raise",
    # Logging
    "setup_structured_logging",
    "get_logger",
    "log_with_context",
    "CorrelationIdMiddleware",
    # Data utils
    "safe_json_serialize",
    "parse_datetime",
    "calculate_hash",
    "validate_required_fields",
]
