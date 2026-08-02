"""
Utils Layer - Generic utilities reusable across any project.
These have zero business logic and can be extracted to SDK for reuse.
"""

from .data_utils import (
    calculate_hash,
    parse_datetime,
    safe_json_serialize,
    validate_required_fields,
)
from .error_handling import (
    ConfigurationError,
    DeviceFingerprintError,
    FraudDetectionError,
    MLModelError,
    PaymentValidationError,
    handle_error,
    log_and_raise,
)
from .logging_utils import (
    CorrelationIdMiddleware,
    get_logger,
    log_with_context,
    setup_structured_logging,
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
