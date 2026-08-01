"""
Request monitoring middleware.

Application metrics are collected by the Prometheus instrumentation in
src/telemetry. A second in-process metrics collector lived here and was never
read by anything - two sources of truth for the same numbers, one of them
always wrong.
"""

import logging
import time
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

SLOW_REQUEST_SECONDS = 1.0


class RequestMonitoringMiddleware(BaseHTTPMiddleware):
    """Time each request, expose the duration, and flag slow ones."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # perf_counter, not time(): it is monotonic, so an NTP correction
        # mid-request cannot produce a negative or wildly inflated duration.
        start = time.perf_counter()

        response = await call_next(request)

        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"

        log = logger.warning if elapsed > SLOW_REQUEST_SECONDS else logger.info
        log(
            "%s %s -> %s in %.4fs",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )

        return response
