# src/utils/rate_limiting.py

import time
from typing import Callable, Dict
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    An in-memory rate limiter using a sliding window algorithm to control request frequency.
    """

    def __init__(self):
        self.requests: Dict[str, list] = {}

    def is_allowed(self, identifier: str, limit: int, window: int) -> bool:
        """
        Checks if a request from a given identifier is allowed within the rate limit.
        """
        now = time.time()
        window_start = now - window

        request_times = self.requests.get(identifier, [])
        valid_requests = [t for t in request_times if t > window_start]

        if len(valid_requests) >= limit:
            return False

        valid_requests.append(now)
        self.requests[identifier] = valid_requests
        return True

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    A FastAPI middleware to enforce rate limiting on incoming requests.
    """

    def __init__(self, app: ASGIApp, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.rate_limiter = RateLimiter()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Processes a request, checking it against the rate limit before proceeding.
        """
        client_ip = request.client.host if request.client else "unknown"

        if not self.rate_limiter.is_allowed(client_ip, self.requests_per_minute, 60):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(status_code=429, detail="Rate limit exceeded.")

        return await call_next(request)
