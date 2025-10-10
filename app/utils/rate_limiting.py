"""
Rate limiting middleware for API protection
"""

import time
from typing import Callable, Dict, Optional
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging

logger = logging.getLogger(__name__)

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware to prevent API abuse
    """
    
    def __init__(
        self, 
        app: ASGIApp, 
        requests_per_minute: int = 60,
        burst_limit: int = 10
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.rate_limiter = InMemoryRateLimiter()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Check rate limits before processing request
        """
        # Get client identifier
        client_ip = self._get_client_ip(request)
        
        # Check rate limit
        if not self.rate_limiter.is_allowed(client_ip, self.requests_per_minute):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self.rate_limiter.get_remaining(client_ip, self.requests_per_minute)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address from request
        """
        # Check for forwarded headers (behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"

class InMemoryRateLimiter:
    """
    In-memory rate limiter using sliding window algorithm
    """
    
    def __init__(self):
        self.requests: Dict[str, list] = {}
    
    def is_allowed(self, identifier: str, limit_per_minute: int) -> bool:
        """
        Check if request is allowed based on rate limit
        """
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        # Initialize or clean old requests
        if identifier not in self.requests:
            self.requests[identifier] = []
        else:
            # Remove requests older than 1 minute
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
        
        # Check if limit is exceeded
        if len(self.requests[identifier]) >= limit_per_minute:
            return False
        
        # Add current request
        self.requests[identifier].append(current_time)
        return True
    
    def get_remaining(self, identifier: str, limit_per_minute: int) -> int:
        """
        Get remaining requests for the current window
        """
        if identifier not in self.requests:
            return limit_per_minute
        
        current_requests = len(self.requests[identifier])
        return max(0, limit_per_minute - current_requests)
    
    def reset(self, identifier: str):
        """
        Reset rate limit for identifier
        """
        if identifier in self.requests:
            del self.requests[identifier]

class AdvancedRateLimiter:
    """
    Advanced rate limiter with different limits for different endpoints
    """
    
    def __init__(self):
        self.limiters = {}
        self.endpoint_limits = {
            "/api/v1/detect-fraud": 100,  # transactions per minute
            "/api/v1/detect-fraud/batch": 10,  # batch requests per minute
            "/api/v1/analytics/dashboard-data": 30,  # dashboard requests per minute
        }
    
    def get_limiter(self, endpoint: str) -> InMemoryRateLimiter:
        """
        Get rate limiter for specific endpoint
        """
        if endpoint not in self.limiters:
            self.limiters[endpoint] = InMemoryRateLimiter()
        return self.limiters[endpoint]
    
    def is_allowed(self, identifier: str, endpoint: str) -> bool:
        """
        Check if request is allowed for specific endpoint
        """
        limit = self.endpoint_limits.get(endpoint, 60)  # Default 60 per minute
        limiter = self.get_limiter(endpoint)
        return limiter.is_allowed(identifier, limit)
    
    def get_remaining(self, identifier: str, endpoint: str) -> int:
        """
        Get remaining requests for endpoint
        """
        limit = self.endpoint_limits.get(endpoint, 60)
        limiter = self.get_limiter(endpoint)
        return limiter.get_remaining(identifier, limit)