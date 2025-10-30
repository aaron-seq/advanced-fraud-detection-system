# src/utils/monitoring.py

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    A FastAPI middleware for logging incoming requests and their processing time.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Processes a request, logs its details, and adds a processing time header.
        """
        start_time = time.time()

        response = await call_next(request)

        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        self.log_request(request, response, process_time)

        return response

    def log_request(self, request: Request, response: Response, process_time: float):
        """
        Logs the details of a request and its corresponding response.
        """
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} "
            f"({process_time:.4f}s)"
        )
        if process_time > 1.0:
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path} "
                f"took {process_time:.4f}s"
            )

class MetricsCollector:
    """
    Collects and stores various application metrics for monitoring and analysis.
    """

    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "fraud_detections_total": 0
        }

    def record_request(self):
        """
        Increments the total number of requests.
        """
        self.metrics["requests_total"] += 1

    def record_fraud_detection(self, is_fraud: bool):
        """
        Records a fraud detection event, incrementing the total if fraud is detected.
        """
        if is_fraud:
            self.metrics["fraud_detections_total"] += 1

    def get_metrics(self) -> dict:
        """
        Returns a snapshot of the current metrics.
        """
        return self.metrics.copy()

# Global instance for application-wide use
metrics_collector = MetricsCollector()
