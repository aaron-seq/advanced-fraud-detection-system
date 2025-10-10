"""
Security and authentication for the fraud detection system
"""

from datetime import datetime, timedelta
from typing import Any, Union, Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import get_application_settings
import logging

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token scheme
security = HTTPBearer(auto_error=False)

def create_access_token(
    subject: Union[str, Any], 
    expires_delta: timedelta = None
) -> str:
    """
    Create JWT access token
    """
    settings = get_application_settings()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    
    to_encode = {"exp": expire, "sub": str(subject)}
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.secret_key, 
        algorithm=settings.algorithm
    )
    
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Get password hash
    """
    return pwd_context.hash(password)

def decode_access_token(token: str) -> dict:
    """
    Decode JWT access token
    """
    settings = get_application_settings()
    
    try:
        payload = jwt.decode(
            token, 
            settings.secret_key, 
            algorithms=[settings.algorithm]
        )
        return payload
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        return None

async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    """
    Verify JWT token from request
    """
    # For development/demo, allow requests without authentication
    settings = get_application_settings()
    
    if settings.environment == "development":
        # Return mock user for development
        return {
            "user_id": "dev_user",
            "username": "developer",
            "role": "admin",
            "permissions": ["read", "write", "admin"]
        }
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    try:
        payload = decode_access_token(token)
        
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # In a real application, you would fetch user data from database
        return {
            "user_id": user_id,
            "username": "authenticated_user",
            "role": "user",
            "permissions": ["read", "write"]
        }
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

class SecurityUtils:
    """
    Security utility functions
    """
    
    @staticmethod
    def create_api_key() -> str:
        """
        Generate API key for external integrations
        """
        import secrets
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """
        Validate API key
        In production, this would check against a database
        """
        # Mock validation for demo
        valid_keys = [
            "demo_api_key_123456789",
            "test_api_key_987654321"
        ]
        return api_key in valid_keys
    
    @staticmethod
    def sanitize_input(user_input: str) -> str:
        """
        Sanitize user input to prevent injection attacks
        """
        import html
        import re
        
        # HTML escape
        sanitized = html.escape(user_input)
        
        # Remove potential SQL injection patterns
        sql_patterns = [
            r"('|(\-\-)|(;)|(\|)|(\*)|(%))",
            r"((\%27)|(\')|(\-\-)|(\%3B)|(;))",
            r"((\%27)|(\'))(union|select|insert|delete|update|drop|create)"
        ]
        
        for pattern in sql_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    @staticmethod
    def check_rate_limit(user_id: str, endpoint: str, limit: int = 100) -> bool:
        """
        Check if user has exceeded rate limit
        In production, this would use Redis or database
        """
        # Mock rate limiting for demo
        return True
    
    @staticmethod
    def log_security_event(event_type: str, details: dict, user_id: str = None):
        """
        Log security events for monitoring
        """
        logger.warning(f"Security Event: {event_type} - {details} - User: {user_id}")