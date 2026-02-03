"""
Data Utilities Module

Generic data manipulation utilities reusable across any project.
These have zero business logic and can be extracted to SDK.
"""

import json
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
from decimal import Decimal
import re


def safe_json_serialize(obj: Any, default: Any = None) -> str:
    """
    Safely serialize any object to JSON string.
    
    Handles:
    - datetime objects
    - Decimal values
    - bytes
    - Sets
    - Custom objects with __dict__
    
    Args:
        obj: Object to serialize
        default: Default value for unserializable objects
    
    Returns:
        JSON string representation
    
    Example:
        data = {"timestamp": datetime.now(), "amount": Decimal("100.50")}
        json_str = safe_json_serialize(data)
    """
    def json_encoder(o):
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, date):
            return o.isoformat()
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="replace")
        if isinstance(o, set):
            return list(o)
        if hasattr(o, "__dict__"):
            return o.__dict__
        if default is not None:
            return default
        return str(o)
    
    return json.dumps(obj, default=json_encoder, ensure_ascii=False)


def safe_json_deserialize(json_str: str, default: Any = None) -> Any:
    """
    Safely deserialize JSON string.
    
    Args:
        json_str: JSON string to parse
        default: Value to return on parse error
    
    Returns:
        Parsed object or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def parse_datetime(
    value: Union[str, datetime, int, float],
    default: Optional[datetime] = None
) -> Optional[datetime]:
    """
    Parse various datetime formats to datetime object.
    
    Supports:
    - ISO 8601 strings
    - Unix timestamps (seconds)
    - Unix timestamps (milliseconds)
    - Common date formats
    
    Args:
        value: Value to parse
        default: Default if parsing fails
    
    Returns:
        datetime object or default
    
    Example:
        dt = parse_datetime("2026-02-03T11:30:00Z")
        dt = parse_datetime(1738571400)  # Unix timestamp
    """
    if isinstance(value, datetime):
        return value
    
    if isinstance(value, (int, float)):
        # Assume milliseconds if > 10 billion
        if value > 1e10:
            value = value / 1000
        try:
            return datetime.fromtimestamp(value)
        except (ValueError, OSError):
            return default
    
    if isinstance(value, str):
        # Try ISO format first
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y",
            "%m/%d/%Y",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        
        # Try ISO format with timezone
        try:
            # Remove timezone suffix for basic parsing
            clean_value = re.sub(r"[+-]\d{2}:?\d{2}$", "", value)
            return datetime.fromisoformat(clean_value.replace("Z", ""))
        except ValueError:
            pass
    
    return default


def calculate_hash(
    data: Union[str, bytes, Dict],
    algorithm: str = "sha256"
) -> str:
    """
    Calculate hash of data.
    
    Args:
        data: Data to hash (str, bytes, or dict)
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)
    
    Returns:
        Hexadecimal hash string
    
    Example:
        hash_value = calculate_hash({"user_id": 123, "amount": 100.0})
    """
    if isinstance(data, dict):
        # Sort keys for consistent hashing
        data = safe_json_serialize(data)
    
    if isinstance(data, str):
        data = data.encode("utf-8")
    
    hash_func = getattr(hashlib, algorithm)
    return hash_func(data).hexdigest()


def validate_required_fields(
    data: Dict[str, Any],
    required_fields: List[str],
    raise_on_missing: bool = True
) -> List[str]:
    """
    Validate that required fields are present in data.
    
    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        raise_on_missing: Whether to raise exception on missing fields
    
    Returns:
        List of missing field names
    
    Raises:
        ValueError: If raise_on_missing is True and fields are missing
    
    Example:
        missing = validate_required_fields(
            {"name": "John"},
            ["name", "email", "phone"]
        )
        # Returns ["email", "phone"] or raises ValueError
    """
    missing = [
        field for field in required_fields
        if field not in data or data[field] is None
    ]
    
    if missing and raise_on_missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    
    return missing


def deep_get(
    data: Dict[str, Any],
    path: str,
    default: Any = None,
    separator: str = "."
) -> Any:
    """
    Get nested value from dictionary using dot notation.
    
    Args:
        data: Dictionary to search
        path: Dot-separated path (e.g., "user.address.city")
        default: Default value if not found
        separator: Path separator (default ".")
    
    Returns:
        Value at path or default
    
    Example:
        city = deep_get(data, "user.address.city", default="Unknown")
    """
    keys = path.split(separator)
    result = data
    
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
        elif isinstance(result, (list, tuple)) and key.isdigit():
            index = int(key)
            if 0 <= index < len(result):
                result = result[index]
            else:
                return default
        else:
            return default
        
        if result is None:
            return default
    
    return result


def deep_set(
    data: Dict[str, Any],
    path: str,
    value: Any,
    separator: str = "."
) -> Dict[str, Any]:
    """
    Set nested value in dictionary using dot notation.
    
    Args:
        data: Dictionary to modify
        path: Dot-separated path
        value: Value to set
        separator: Path separator
    
    Returns:
        Modified dictionary
    
    Example:
        data = {}
        deep_set(data, "user.address.city", "New York")
        # Result: {"user": {"address": {"city": "New York"}}}
    """
    keys = path.split(separator)
    current = data
    
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
    return data


def mask_sensitive_data(
    data: Dict[str, Any],
    sensitive_fields: List[str] = None,
    mask_char: str = "*",
    visible_chars: int = 4
) -> Dict[str, Any]:
    """
    Mask sensitive fields in dictionary for logging.
    
    Args:
        data: Dictionary containing sensitive data
        sensitive_fields: Fields to mask (default: common sensitive fields)
        mask_char: Character to use for masking
        visible_chars: Number of characters to keep visible at end
    
    Returns:
        Copy of dictionary with masked values
    
    Example:
        masked = mask_sensitive_data({"card_number": "1234567890123456"})
        # Result: {"card_number": "************3456"}
    """
    if sensitive_fields is None:
        sensitive_fields = [
            "password", "secret", "token", "api_key", "apikey",
            "card_number", "cvv", "ssn", "credit_card", "account_number"
        ]
    
    result = {}
    
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = mask_sensitive_data(
                value, sensitive_fields, mask_char, visible_chars
            )
        elif isinstance(value, str) and any(
            sf.lower() in key.lower() for sf in sensitive_fields
        ):
            if len(value) > visible_chars:
                result[key] = mask_char * (len(value) - visible_chars) + value[-visible_chars:]
            else:
                result[key] = mask_char * len(value)
        else:
            result[key] = value
    
    return result


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: List to split
        chunk_size: Maximum size of each chunk
    
    Returns:
        List of chunks
    
    Example:
        chunks = chunk_list([1, 2, 3, 4, 5], 2)
        # Result: [[1, 2], [3, 4], [5]]
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(
    data: Dict[str, Any],
    separator: str = ".",
    parent_key: str = ""
) -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        data: Nested dictionary
        separator: Key separator
        parent_key: Parent key prefix
    
    Returns:
        Flattened dictionary
    
    Example:
        flat = flatten_dict({"a": {"b": {"c": 1}}})
        # Result: {"a.b.c": 1}
    """
    items = []
    
    for key, value in data.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(
                flatten_dict(value, separator, new_key).items()
            )
        else:
            items.append((new_key, value))
    
    return dict(items)
