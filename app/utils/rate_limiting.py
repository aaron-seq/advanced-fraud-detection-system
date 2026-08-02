"""
Rate limiting middleware for API protection.
"""

import logging
import time
from collections import OrderedDict, deque
from collections.abc import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import get_application_settings

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60


class InMemoryRateLimiter:
    """
    Sliding-window rate limiter backed by a bounded LRU of client identifiers.

    ``max_identifiers`` caps memory. Without it, every distinct client IP
    allocates a list that is never reclaimed, so traffic from many sources
    grows the process until it is killed - the limiter itself becomes the
    denial-of-service vector it exists to prevent.

    Per-process only: with several workers each holds its own counts, so the
    effective limit is ``limit x workers``. Move to Redis if that matters.
    """

    def __init__(self, max_identifiers: int = 10_000):
        self.max_identifiers = max_identifiers
        self._requests: OrderedDict[str, deque[float]] = OrderedDict()

    def _window(self, identifier: str, now: float) -> deque[float]:
        timestamps = self._requests.get(identifier)

        if timestamps is None:
            timestamps = deque()
            self._requests[identifier] = timestamps
            while len(self._requests) > self.max_identifiers:
                self._requests.popitem(last=False)  # evict least recently used
        else:
            self._requests.move_to_end(identifier)

        cutoff = now - WINDOW_SECONDS
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()

        return timestamps

    def is_allowed(self, identifier: str, limit_per_minute: int) -> bool:
        """Record a request and report whether it is within the limit."""
        now = time.monotonic()
        timestamps = self._window(identifier, now)

        if len(timestamps) >= limit_per_minute:
            return False

        timestamps.append(now)
        return True

    def get_remaining(self, identifier: str, limit_per_minute: int) -> int:
        """Requests still available to ``identifier`` in the current window."""
        timestamps = self._window(identifier, time.monotonic())
        return max(0, limit_per_minute - len(timestamps))

    def reset(self, identifier: str) -> None:
        """Forget all requests recorded for ``identifier``."""
        self._requests.pop(identifier, None)


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Reject clients that exceed the configured request rate."""

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        trust_proxy_headers: bool | None = None,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.rate_limiter = InMemoryRateLimiter()

        if trust_proxy_headers is None:
            trust_proxy_headers = get_application_settings().trust_proxy_headers
        self.trust_proxy_headers = trust_proxy_headers

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = self._get_client_ip(request)

        if not self.rate_limiter.is_allowed(client_ip, self.requests_per_minute):
            logger.warning("Rate limit exceeded for %s", client_ip)
            # Returned, not raised. HTTPException raised inside middleware runs
            # outside the router, so FastAPI's handler never sees it and the
            # client gets a 500 instead of the 429 this is meant to signal.
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please try again later."},
                headers={
                    "Retry-After": str(WINDOW_SECONDS),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)

        remaining = self.rate_limiter.get_remaining(client_ip, self.requests_per_minute)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + WINDOW_SECONDS)

        return response

    def _get_client_ip(self, request: Request) -> str:
        """
        Resolve the client identifier used for rate limiting.

        Proxy headers are only consulted when TRUST_PROXY_HEADERS is enabled.
        They are attacker-controlled otherwise: a client that sets a fresh
        X-Forwarded-For on each request gets a fresh bucket every time and is
        never limited at all.
        """
        if self.trust_proxy_headers:
            forwarded_for = request.headers.get("X-Forwarded-For")
            if forwarded_for:
                return forwarded_for.split(",")[0].strip()

            real_ip = request.headers.get("X-Real-IP")
            if real_ip:
                return real_ip

        return request.client.host if request.client else "unknown"
