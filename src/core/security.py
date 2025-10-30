# src/core/security.py

from datetime import datetime, timedelta
from typing import Any, Union, Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from app.core.config import get_application_settings

# Password hashing configuration
password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token-based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class AuthManager:
    """
    Manages authentication, including password hashing and JWT creation/validation.
    """

    def __init__(self, settings):
        self.settings = settings

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verifies a plain-text password against a hashed password."""
        return password_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hashes a plain-text password."""
        return password_context.hash(password)

    def create_access_token(self, subject: Union[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Creates a new JWT access token."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.settings.access_token_expire_minutes)

        to_encode = {"exp": expire, "sub": str(subject)}
        return jwt.encode(to_encode, self.settings.secret_key, algorithm=self.settings.algorithm)

def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Dependency to get the current user from a token, raising an exception if invalid.
    """
    settings = get_application_settings()
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # In a real application, you would fetch the user from the database here
    return {"username": username}

# Global instance for application-wide use
auth_manager = AuthManager(get_application_settings())
