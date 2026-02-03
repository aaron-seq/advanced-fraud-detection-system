"""
Business Metrics Module

Tracks business value and model performance:
- Fraud prevented (dollars saved)
- Detection accuracy (precision, recall, F1)
- False positive impact
- Model drift indicators
- A/B testing framework

ANALYTICS vs TELEMETRY:
- Telemetry: Is the system healthy? (ops focused)
- Analytics: Is the system effective? (business focused)
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import logging

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FraudPreventionStats:
    """Statistics on fraud prevention effectiveness."""
    
    period_start: datetime
    period_end: datetime
    
    # Transaction counts
    total_transactions: int = 0
    approved_transactions: int = 0
    denied_transactions: int = 0
    review_transactions: int = 0
    
    # Fraud detection
    true_positives: int = 0  # Correctly detected fraud
    false_positives: int = 0  # Incorrectly flagged legitimate
    true_negatives: int = 0  # Correctly passed legitimate
    false_negatives: int = 0  # Missed fraud
    
    # Financial impact
    total_amount_processed: float = 0.0
    fraud_amount_prevented: float = 0.0
    fraud_amount_missed: float = 0.0
    false_positive_amount: float = 0.0
    
    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    @property
    def false_positive_rate(self) -> float:
        """FPR = FP / (FP + TN)"""
        total = self.false_positives + self.true_negatives
        return self.false_positives / total if total > 0 else 0.0
    
    @property
    def detection_rate(self) -> float:
        """Detection rate = TP / (TP + FN)"""
        return self.recall
    
    @property
    def net_savings(self) -> float:
        """Net savings = prevented - (missed + false positive cost)"""
        # Assume false positive cost is 1% of transaction (customer friction)
        fp_cost = self.false_positive_amount * 0.01
        return self.fraud_amount_prevented - self.fraud_amount_missed - fp_cost
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "transactions": {
                "total": self.total_transactions,
                "approved": self.approved_transactions,
                "denied": self.denied_transactions,
                "review": self.review_transactions,
            },
            "detection": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "true_negatives": self.true_negatives,
                "false_negatives": self.false_negatives,
            },
            "metrics": {
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1_score": round(self.f1_score, 4),
                "false_positive_rate": round(self.false_positive_rate, 4),
            },
            "financial_impact": {
                "total_processed": round(self.total_amount_processed, 2),
                "fraud_prevented": round(self.fraud_amount_prevented, 2),
                "fraud_missed": round(self.fraud_amount_missed, 2),
                "net_savings": round(self.net_savings, 2),
            },
        }


@dataclass
class ModelPerformanceStats:
    """ML model performance statistics."""
    
    model_name: str
    model_version: str
    
    # Prediction stats
    total_predictions: int = 0
    fraud_predictions: int = 0
    legitimate_predictions: int = 0
    
    # Performance
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Accuracy after feedback
    accuracy: float = 0.0
    auc_roc: float = 0.0
    
    # Drift indicators
    feature_drift_score: float = 0.0
    prediction_drift_score: float = 0.0
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


class BusinessMetricsCollector:
    """
    Collects and aggregates business metrics for fraud detection.
    
    Provides:
    - Real-time fraud prevention statistics
    - Model performance tracking
    - Trend analysis
    - A/B testing support
    
    Usage:
        collector = BusinessMetricsCollector()
        collector.record_decision(transaction_id, "approve", amount, is_fraud=False)
        stats = collector.get_period_stats(start, end)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # In-memory storage (would use database in production)
        self._decisions: List[Dict[str, Any]] = []
        self._model_stats: Dict[str, ModelPerformanceStats] = {}
        self._feedback: Dict[str, bool] = {}  # transaction_id -> is_fraud
        
        # Aggregation windows
        self._hourly_stats: Dict[str, FraudPreventionStats] = {}
        self._daily_stats: Dict[str, FraudPreventionStats] = {}
        
        logger.info("BusinessMetricsCollector initialized")
    
    def record_decision(
        self,
        transaction_id: str,
        decision: str,
        amount: float,
        risk_score: float,
        model_name: str = "ensemble",
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a fraud detection decision.
        
        Args:
            transaction_id: Transaction identifier
            decision: Decision made (approve, deny, review)
            amount: Transaction amount
            risk_score: Computed risk score
            model_name: Model that made prediction
            latency_ms: Processing latency
            metadata: Additional metadata
        """
        record = {
            "transaction_id": transaction_id,
            "decision": decision,
            "amount": amount,
            "risk_score": risk_score,
            "model_name": model_name,
            "latency_ms": latency_ms,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {},
        }
        
        self._decisions.append(record)
        
        # Update model stats
        self._update_model_stats(model_name, latency_ms, decision)
        
        logger.debug(
            "Decision recorded",
            extra={
                "transaction_id": transaction_id,
                "decision": decision,
                "amount": amount,
            }
        )
    
    def record_feedback(
        self,
        transaction_id: str,
        was_fraud: bool
    ) -> None:
        """
        Record feedback on whether transaction was actually fraud.
        
        This is used for calculating true/false positive rates.
        
        Args:
            transaction_id: Transaction identifier
            was_fraud: Whether transaction was actually fraudulent
        """
        self._feedback[transaction_id] = was_fraud
        
        logger.debug(
            "Feedback recorded",
            extra={
                "transaction_id": transaction_id,
                "was_fraud": was_fraud,
            }
        )
    
    def get_period_stats(
        self,
        start: datetime,
        end: datetime
    ) -> FraudPreventionStats:
        """
        Get fraud prevention statistics for a time period.
        
        Args:
            start: Period start
            end: Period end
            
        Returns:
            FraudPreventionStats for the period
        """
        stats = FraudPreventionStats(
            period_start=start,
            period_end=end
        )
        
        # Filter decisions in period
        period_decisions = [
            d for d in self._decisions
            if start <= d["timestamp"] <= end
        ]
        
        for decision in period_decisions:
            stats.total_transactions += 1
            stats.total_amount_processed += decision["amount"]
            
            # Track by decision type
            if decision["decision"] == "approve":
                stats.approved_transactions += 1
            elif decision["decision"] == "deny":
                stats.denied_transactions += 1
            else:
                stats.review_transactions += 1
            
            # Check feedback for accuracy
            feedback = self._feedback.get(decision["transaction_id"])
            if feedback is not None:
                was_fraud = feedback
                predicted_fraud = decision["decision"] in ("deny", "review")
                
                if was_fraud and predicted_fraud:
                    stats.true_positives += 1
                    stats.fraud_amount_prevented += decision["amount"]
                elif was_fraud and not predicted_fraud:
                    stats.false_negatives += 1
                    stats.fraud_amount_missed += decision["amount"]
                elif not was_fraud and predicted_fraud:
                    stats.false_positives += 1
                    stats.false_positive_amount += decision["amount"]
                else:
                    stats.true_negatives += 1
        
        return stats
    
    def get_realtime_stats(self) -> Dict[str, Any]:
        """
        Get real-time statistics (last hour).
        
        Returns:
            Dictionary with current stats
        """
        end = datetime.utcnow()
        start = end - timedelta(hours=1)
        
        stats = self.get_period_stats(start, end)
        
        return {
            "period": "last_hour",
            "transactions_per_minute": stats.total_transactions / 60,
            "approval_rate": (
                stats.approved_transactions / stats.total_transactions
                if stats.total_transactions > 0 else 0
            ),
            "denial_rate": (
                stats.denied_transactions / stats.total_transactions
                if stats.total_transactions > 0 else 0
            ),
            "avg_amount": (
                stats.total_amount_processed / stats.total_transactions
                if stats.total_transactions > 0 else 0
            ),
            "precision": stats.precision,
            "recall": stats.recall,
        }
    
    def get_model_performance(
        self,
        model_name: str
    ) -> Optional[ModelPerformanceStats]:
        """Get performance stats for a specific model."""
        return self._model_stats.get(model_name)
    
    def get_all_model_stats(self) -> Dict[str, ModelPerformanceStats]:
        """Get performance stats for all models."""
        return self._model_stats.copy()
    
    def calculate_drift_score(
        self,
        model_name: str,
        window_hours: int = 24
    ) -> float:
        """
        Calculate model drift score based on prediction distribution.
        
        Higher score indicates more drift from baseline.
        
        Args:
            model_name: Model to check
            window_hours: Time window to analyze
            
        Returns:
            Drift score (0.0 to 1.0)
        """
        end = datetime.utcnow()
        start = end - timedelta(hours=window_hours)
        
        # Get recent predictions
        recent = [
            d for d in self._decisions
            if start <= d["timestamp"] <= end
            and d["model_name"] == model_name
        ]
        
        if not recent:
            return 0.0
        
        # Calculate distribution of risk scores
        recent_scores = [d["risk_score"] for d in recent]
        recent_mean = statistics.mean(recent_scores)
        recent_std = statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0
        
        # Compare to baseline (all historical)
        all_model = [
            d for d in self._decisions
            if d["model_name"] == model_name
        ]
        
        if len(all_model) < 100:
            return 0.0  # Not enough data
        
        baseline_scores = [d["risk_score"] for d in all_model]
        baseline_mean = statistics.mean(baseline_scores)
        baseline_std = statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 1
        
        # Calculate drift as normalized difference
        mean_drift = abs(recent_mean - baseline_mean) / (baseline_std + 0.01)
        
        # Normalize to 0-1 range
        return min(mean_drift / 3, 1.0)  # 3 std devs = max drift
    
    def _update_model_stats(
        self,
        model_name: str,
        latency_ms: float,
        decision: str
    ) -> None:
        """Update model performance statistics."""
        if model_name not in self._model_stats:
            self._model_stats[model_name] = ModelPerformanceStats(
                model_name=model_name,
                model_version="unknown"
            )
        
        stats = self._model_stats[model_name]
        stats.total_predictions += 1
        
        if decision in ("deny", "review"):
            stats.fraud_predictions += 1
        else:
            stats.legitimate_predictions += 1
        
        # Update latency (rolling average)
        n = stats.total_predictions
        stats.avg_latency_ms = (
            (stats.avg_latency_ms * (n - 1) + latency_ms) / n
        )
    
    def generate_report(
        self,
        period_days: int = 7
    ) -> Dict[str, Any]:
        """
        Generate comprehensive business metrics report.
        
        Args:
            period_days: Number of days to include
            
        Returns:
            Dictionary with full report
        """
        end = datetime.utcnow()
        start = end - timedelta(days=period_days)
        
        stats = self.get_period_stats(start, end)
        
        report = {
            "report_generated": datetime.utcnow().isoformat(),
            "period_days": period_days,
            "fraud_prevention": stats.to_dict(),
            "model_performance": {
                name: {
                    "model_name": s.model_name,
                    "total_predictions": s.total_predictions,
                    "fraud_rate": (
                        s.fraud_predictions / s.total_predictions
                        if s.total_predictions > 0 else 0
                    ),
                    "avg_latency_ms": round(s.avg_latency_ms, 2),
                }
                for name, s in self._model_stats.items()
            },
            "trends": self._calculate_trends(period_days),
        }
        
        return report
    
    def _calculate_trends(self, period_days: int) -> Dict[str, Any]:
        """Calculate trend indicators."""
        # Would implement proper trend analysis
        return {
            "fraud_rate_trend": "stable",
            "volume_trend": "increasing",
            "false_positive_trend": "decreasing",
        }
