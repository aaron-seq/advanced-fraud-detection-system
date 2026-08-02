"""
Authentication and token handling for the fraud detection system.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_application_settings

# JWT handling uses PyJWT rather than python-jose. python-jose 3.3.0 carries
# unpatched advisories (PYSEC-2024-232/233, PYSEC-2025-185) and drags in ecdsa
# and rsa, which are only needed for asymmetric algorithms this service does
# not use. PyJWT covers HS256 with no such transitive dependencies.
#
# Password hashing lived here via passlib's CryptContext. It had no callers -
# there is no login or registration endpoint - and passlib 1.7.4 (last released
# 2020) raises on every call against bcrypt >= 4.1, so it was dead code that
# could not have worked. Add hashing back with the `bcrypt` package directly
# when an endpoint actually needs it; passlib is no longer maintained.

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def create_access_token(
    subject: str | Any,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT access token for ``subject``."""
    settings = get_application_settings()

    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )

    return jwt.encode(
        {"exp": expire, "sub": str(subject)},
        settings.secret_key,
        algorithm=settings.algorithm,
    )


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT access token.

    Raises HTTPException(401) on any failure rather than returning None, so a
    caller that forgets to check the result cannot accidentally treat an
    invalid token as valid.
    """
    settings = get_application_settings()

    try:
        return jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
    except jwt.PyJWTError as exc:
        logger.warning("Rejected token: %s", exc)
        raise CREDENTIALS_EXCEPTION from exc


async def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    """
    Authenticate the caller from a bearer token.

    Every environment is authenticated identically. This previously returned a
    synthetic admin with full permissions whenever environment == "development"
    - the default - which meant an unauthenticated caller was handed admin
    rights on any deployment that had not explicitly set ENVIRONMENT.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)

    user_id = payload.get("sub")
    if not user_id:
        raise CREDENTIALS_EXCEPTION

    return {"user_id": user_id, "permissions": ["read", "write"]}
