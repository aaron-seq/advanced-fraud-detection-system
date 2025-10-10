"""
Monitoring and observability middleware
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import json

logger = logging.getLogger(__name__)

class RequestMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for monitoring API requests and responses
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and add monitoring
        """
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Time: {process_time:.4f}s"
        )
        
        # Log slow requests
        if process_time > 1.0:
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path} - "
                f"{process_time:.4f}s"
            )
        
        return response

class MetricsCollector:
    """
    Collect and store application metrics
    """
    
    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "requests_by_method": {},
            "requests_by_status": {},
            "response_times": [],
            "fraud_detections_total": 0,
            "model_predictions": 0
        }
    
    def record_request(self, method: str, status_code: int, response_time: float):
        """
        Record request metrics
        """
        self.metrics["requests_total"] += 1
        
        # By method
        if method not in self.metrics["requests_by_method"]:
            self.metrics["requests_by_method"][method] = 0
        self.metrics["requests_by_method"][method] += 1
        
        # By status
        if status_code not in self.metrics["requests_by_status"]:
            self.metrics["requests_by_status"][status_code] = 0
        self.metrics["requests_by_status"][status_code] += 1
        
        # Response times (keep last 1000)
        self.metrics["response_times"].append(response_time)
        if len(self.metrics["response_times"]) > 1000:
            self.metrics["response_times"] = self.metrics["response_times"][-1000:]
    
    def record_fraud_detection(self, is_fraud: bool):
        """
        Record fraud detection metrics
        """
        self.metrics["model_predictions"] += 1
        if is_fraud:
            self.metrics["fraud_detections_total"] += 1
    
    def get_metrics(self) -> dict:
        """
        Get current metrics snapshot
        """
        response_times = self.metrics["response_times"]
        
        metrics_snapshot = self.metrics.copy()
        
        if response_times:
            metrics_snapshot["avg_response_time"] = sum(response_times) / len(response_times)
            metrics_snapshot["max_response_time"] = max(response_times)
            metrics_snapshot["min_response_time"] = min(response_times)
        else:
            metrics_snapshot["avg_response_time"] = 0
            metrics_snapshot["max_response_time"] = 0
            metrics_snapshot["min_response_time"] = 0
        
        return metrics_snapshot

# Global metrics collector
metrics_collector = MetricsCollector()