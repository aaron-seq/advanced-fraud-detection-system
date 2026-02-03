"""
Prometheus Metrics for Fraud Detection System

Telemetry (System Health) vs Analytics (Business Value):

TELEMETRY (this module):
- How fast is the system?
- How many errors occurred?
- Is the system healthy?
- What's the resource utilization?

ANALYTICS (separate module):
- How much fraud did we prevent?
- What's our detection accuracy?
- What's the business impact?

Metrics defined here:
- fraud_detection_checks_total: Counter of all fraud checks
- fraud_detection_check_duration_seconds: Histogram of check latency
- ml_model_inference_duration_seconds: Histogram of ML model latency
- fraud_detection_active_transactions: Gauge of in-flight transactions
- fraud_detection_errors_total: Counter of errors by type
"""

from typing import Dict, Any, Optional
from contextlib import contextmanager
from datetime import datetime
import time
import logging

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default registry for metrics
_registry: Optional["CollectorRegistry"] = None

# Metric definitions (initialized lazily)
FRAUD_CHECKS_TOTAL: Optional["Counter"] = None
FRAUD_CHECK_LATENCY: Optional["Histogram"] = None
ML_MODEL_LATENCY: Optional["Histogram"] = None
ACTIVE_TRANSACTIONS: Optional["Gauge"] = None
FRAUD_DECISIONS: Optional["Counter"] = None
FRAUD_DETECTION_ERRORS: Optional["Counter"] = None
DEVICE_FINGERPRINT_LATENCY: Optional["Histogram"] = None
PAYMENT_VALIDATION_LATENCY: Optional["Histogram"] = None
SYSTEM_INFO: Optional["Info"] = None

# Latency buckets optimized for fraud detection (sub-ms to seconds)
LATENCY_BUCKETS = (
    0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05,
    0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0
)


def setup_metrics(
    registry: Optional["CollectorRegistry"] = None,
    prefix: str = "fraud_detection"
) -> None:
    """
    Initialize Prometheus metrics.
    
    Args:
        registry: Optional custom registry (uses default if None)
        prefix: Metric name prefix
    """
    global _registry, FRAUD_CHECKS_TOTAL, FRAUD_CHECK_LATENCY
    global ML_MODEL_LATENCY, ACTIVE_TRANSACTIONS, FRAUD_DECISIONS
    global FRAUD_DETECTION_ERRORS, DEVICE_FINGERPRINT_LATENCY
    global PAYMENT_VALIDATION_LATENCY, SYSTEM_INFO
    
    if not PROMETHEUS_AVAILABLE:
        logger.warning("prometheus_client not available, metrics disabled")
        return
    
    _registry = registry or CollectorRegistry()
    
    # Total fraud checks counter
    FRAUD_CHECKS_TOTAL = Counter(
        f"{prefix}_checks_total",
        "Total number of fraud detection checks",
        labelnames=["status", "risk_level"],
        registry=_registry
    )
    
    # Fraud check latency histogram
    FRAUD_CHECK_LATENCY = Histogram(
        f"{prefix}_check_duration_seconds",
        "Time spent processing fraud check",
        labelnames=["decision"],
        buckets=LATENCY_BUCKETS,
        registry=_registry
    )
    
    # ML model inference latency
    ML_MODEL_LATENCY = Histogram(
        f"{prefix}_ml_inference_duration_seconds",
        "Time spent on ML model inference",
        labelnames=["model_name"],
        buckets=LATENCY_BUCKETS,
        registry=_registry
    )
    
    # Active transactions gauge
    ACTIVE_TRANSACTIONS = Gauge(
        f"{prefix}_active_transactions",
        "Number of transactions currently being processed",
        registry=_registry
    )
    
    # Fraud decisions counter
    FRAUD_DECISIONS = Counter(
        f"{prefix}_decisions_total",
        "Total fraud detection decisions",
        labelnames=["decision", "risk_level"],
        registry=_registry
    )
    
    # Error counter
    FRAUD_DETECTION_ERRORS = Counter(
        f"{prefix}_errors_total",
        "Total errors in fraud detection",
        labelnames=["error_type", "component"],
        registry=_registry
    )
    
    # Device fingerprint latency
    DEVICE_FINGERPRINT_LATENCY = Histogram(
        f"{prefix}_device_fingerprint_duration_seconds",
        "Time spent generating device fingerprint",
        buckets=LATENCY_BUCKETS,
        registry=_registry
    )
    
    # Payment validation latency
    PAYMENT_VALIDATION_LATENCY = Histogram(
        f"{prefix}_payment_validation_duration_seconds",
        "Time spent validating payment source",
        buckets=LATENCY_BUCKETS,
        registry=_registry
    )
    
    # System info
    SYSTEM_INFO = Info(
        f"{prefix}_system",
        "Fraud detection system information",
        registry=_registry
    )
    
    SYSTEM_INFO.info({
        "version": "2.9.0",
        "environment": "production",
    })
    
    logger.info("Prometheus metrics initialized", extra={"prefix": prefix})


def get_metrics() -> "FraudDetectionMetrics":
    """Get metrics instance."""
    return FraudDetectionMetrics()


class FraudDetectionMetrics:
    """
    Wrapper class for fraud detection metrics.
    
    Provides type-safe, context manager-based metric recording.
    Gracefully handles missing prometheus_client.
    
    Usage:
        metrics = FraudDetectionMetrics()
        
        with metrics.timer("fraud_check", labels={"decision": "approve"}):
            result = process_fraud_check(transaction)
        
        metrics.counter("fraud_checks", labels={"status": "success"})
    """
    
    def __init__(self):
        self._enabled = PROMETHEUS_AVAILABLE and FRAUD_CHECKS_TOTAL is not None
    
    @contextmanager
    def timer(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.
        
        Args:
            metric_name: Name of the metric (maps to histogram)
            labels: Optional labels
            
        Example:
            with metrics.timer("fraud_check", {"decision": "approve"}):
                process_transaction()
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self._record_histogram(metric_name, duration, labels)
    
    def counter(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
        value: int = 1
    ) -> None:
        """
        Increment a counter metric.
        
        Args:
            metric_name: Name of the counter
            labels: Optional labels
            value: Amount to increment by
        """
        if not self._enabled:
            return
        
        metric = self._get_counter(metric_name)
        if metric:
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)
    
    def gauge_set(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge value."""
        if not self._enabled:
            return
        
        metric = self._get_gauge(metric_name)
        if metric:
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
    
    def gauge_inc(self, metric_name: str, value: float = 1) -> None:
        """Increment a gauge."""
        if not self._enabled:
            return
        
        metric = self._get_gauge(metric_name)
        if metric:
            metric.inc(value)
    
    def gauge_dec(self, metric_name: str, value: float = 1) -> None:
        """Decrement a gauge."""
        if not self._enabled:
            return
        
        metric = self._get_gauge(metric_name)
        if metric:
            metric.dec(value)
    
    def record_error(
        self,
        error_type: str,
        component: str
    ) -> None:
        """Record an error occurrence."""
        if not self._enabled or not FRAUD_DETECTION_ERRORS:
            return
        
        FRAUD_DETECTION_ERRORS.labels(
            error_type=error_type,
            component=component
        ).inc()
    
    def record_decision(
        self,
        decision: str,
        risk_level: str
    ) -> None:
        """Record a fraud detection decision."""
        if not self._enabled:
            return
        
        if FRAUD_DECISIONS:
            FRAUD_DECISIONS.labels(
                decision=decision,
                risk_level=risk_level
            ).inc()
        
        if FRAUD_CHECKS_TOTAL:
            FRAUD_CHECKS_TOTAL.labels(
                status="completed",
                risk_level=risk_level
            ).inc()
    
    def record_latency(
        self,
        metric_name: str,
        duration_seconds: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a latency observation."""
        self._record_histogram(metric_name, duration_seconds, labels)
    
    def _record_histogram(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record value to histogram."""
        if not self._enabled:
            return
        
        metric = self._get_histogram(metric_name)
        if metric:
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
    
    def _get_counter(self, name: str) -> Optional["Counter"]:
        """Get counter by name."""
        counters = {
            "fraud_checks": FRAUD_CHECKS_TOTAL,
            "decisions": FRAUD_DECISIONS,
            "errors": FRAUD_DETECTION_ERRORS,
        }
        return counters.get(name)
    
    def _get_histogram(self, name: str) -> Optional["Histogram"]:
        """Get histogram by name."""
        histograms = {
            "fraud_check": FRAUD_CHECK_LATENCY,
            "ml_inference": ML_MODEL_LATENCY,
            "device_fingerprint": DEVICE_FINGERPRINT_LATENCY,
            "payment_validation": PAYMENT_VALIDATION_LATENCY,
        }
        return histograms.get(name)
    
    def _get_gauge(self, name: str) -> Optional["Gauge"]:
        """Get gauge by name."""
        gauges = {
            "active_transactions": ACTIVE_TRANSACTIONS,
        }
        return gauges.get(name)
    
    @contextmanager
    def track_active_transaction(self):
        """
        Track active transaction count.
        
        Usage:
            with metrics.track_active_transaction():
                process_transaction()
        """
        if self._enabled and ACTIVE_TRANSACTIONS:
            ACTIVE_TRANSACTIONS.inc()
        try:
            yield
        finally:
            if self._enabled and ACTIVE_TRANSACTIONS:
                ACTIVE_TRANSACTIONS.dec()


def generate_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    if not PROMETHEUS_AVAILABLE or not _registry:
        return b""
    
    return generate_latest(_registry)


def get_content_type() -> str:
    """Get content type for metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        return "text/plain"
    
    return CONTENT_TYPE_LATEST
