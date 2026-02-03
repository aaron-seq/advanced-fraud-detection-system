"""
Spending Pattern Analyzer

Analyzes user spending patterns for anomaly detection:
- Transaction frequency patterns
- Amount deviation analysis
- Merchant category patterns
- Time-of-day patterns
- Geographic patterns
- Velocity checks
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import logging

from ...block.models import Transaction, UserProfile
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SpendingPattern:
    """User's historical spending pattern."""
    user_id: str
    avg_amount: float = 0.0
    std_amount: float = 0.0
    median_amount: float = 0.0
    max_amount: float = 0.0
    transactions_per_day: float = 0.0
    transactions_per_hour: Dict[int, float] = field(default_factory=dict)
    category_distribution: Dict[str, float] = field(default_factory=dict)
    location_distribution: Dict[str, float] = field(default_factory=dict)
    weekday_distribution: Dict[int, float] = field(default_factory=dict)


@dataclass
class AnomalyScore:
    """Anomaly detection result."""
    overall_score: float
    amount_anomaly: float
    frequency_anomaly: float
    time_anomaly: float
    category_anomaly: float
    location_anomaly: float
    velocity_anomaly: float
    factors: List[str] = field(default_factory=list)
    
    @property
    def is_anomalous(self) -> bool:
        return self.overall_score >= 0.6


class SpendingPatternAnalyzer:
    """
    Analyzes spending patterns and detects anomalies.
    
    Detection methods:
    1. Statistical deviation (z-score based)
    2. Temporal pattern analysis
    3. Category distribution matching
    4. Geographic clustering
    5. Velocity detection
    
    Usage:
        analyzer = SpendingPatternAnalyzer()
        pattern = analyzer.build_pattern(user_id, transaction_history)
        anomaly = analyzer.detect_anomaly(transaction, pattern)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Thresholds
        self.z_score_threshold = self.config.get("z_score_threshold", 2.5)
        self.anomaly_threshold = self.config.get("anomaly_threshold", 0.6)
        
        # Velocity limits
        self.max_transactions_per_hour = self.config.get("max_hourly_transactions", 10)
        self.max_transactions_per_day = self.config.get("max_daily_transactions", 50)
        
        logger.info("SpendingPatternAnalyzer initialized")
    
    def build_pattern(
        self,
        user_id: str,
        transaction_history: List[Dict[str, Any]]
    ) -> SpendingPattern:
        """
        Build spending pattern from transaction history.
        
        Args:
            user_id: User identifier
            transaction_history: List of historical transactions
            
        Returns:
            SpendingPattern object with aggregated patterns
        """
        if not transaction_history:
            return SpendingPattern(user_id=user_id)
        
        # Extract amounts
        amounts = [t.get("amount", 0.0) for t in transaction_history]
        
        # Calculate amount statistics
        avg_amount = statistics.mean(amounts) if amounts else 0.0
        std_amount = statistics.stdev(amounts) if len(amounts) > 1 else 0.0
        median_amount = statistics.median(amounts) if amounts else 0.0
        max_amount = max(amounts) if amounts else 0.0
        
        # Calculate frequency
        if len(transaction_history) >= 2:
            timestamps = [
                self._parse_timestamp(t.get("timestamp"))
                for t in transaction_history
                if t.get("timestamp")
            ]
            timestamps = [t for t in timestamps if t]
            
            if len(timestamps) >= 2:
                timestamps.sort()
                date_range = (timestamps[-1] - timestamps[0]).days or 1
                transactions_per_day = len(transaction_history) / date_range
            else:
                transactions_per_day = 1.0
        else:
            transactions_per_day = 1.0
        
        # Hourly distribution
        hour_counts: Dict[int, int] = defaultdict(int)
        for t in transaction_history:
            ts = self._parse_timestamp(t.get("timestamp"))
            if ts:
                hour_counts[ts.hour] += 1
        
        total_transactions = len(transaction_history)
        hourly_dist = {
            hour: count / total_transactions
            for hour, count in hour_counts.items()
        }
        
        # Category distribution
        category_counts: Dict[str, int] = defaultdict(int)
        for t in transaction_history:
            category = t.get("merchant_category", "unknown")
            category_counts[category] += 1
        
        category_dist = {
            cat: count / total_transactions
            for cat, count in category_counts.items()
        }
        
        # Location distribution
        location_counts: Dict[str, int] = defaultdict(int)
        for t in transaction_history:
            location = t.get("location", {})
            city = location.get("city", "unknown") if location else "unknown"
            location_counts[city] += 1
        
        location_dist = {
            loc: count / total_transactions
            for loc, count in location_counts.items()
        }
        
        # Weekday distribution
        weekday_counts: Dict[int, int] = defaultdict(int)
        for t in transaction_history:
            ts = self._parse_timestamp(t.get("timestamp"))
            if ts:
                weekday_counts[ts.weekday()] += 1
        
        weekday_dist = {
            day: count / total_transactions
            for day, count in weekday_counts.items()
        }
        
        pattern = SpendingPattern(
            user_id=user_id,
            avg_amount=avg_amount,
            std_amount=std_amount,
            median_amount=median_amount,
            max_amount=max_amount,
            transactions_per_day=transactions_per_day,
            transactions_per_hour=hourly_dist,
            category_distribution=category_dist,
            location_distribution=location_dist,
            weekday_distribution=weekday_dist,
        )
        
        logger.debug(
            "Built spending pattern",
            extra={
                "user_id": user_id,
                "avg_amount": avg_amount,
                "transactions_per_day": transactions_per_day,
                "history_size": len(transaction_history),
            }
        )
        
        return pattern
    
    def detect_anomaly(
        self,
        transaction: Transaction,
        pattern: SpendingPattern
    ) -> AnomalyScore:
        """
        Detect anomalies in transaction compared to user pattern.
        
        Args:
            transaction: Transaction to analyze
            pattern: User's historical pattern
            
        Returns:
            AnomalyScore with detailed breakdown
        """
        factors = []
        
        # Amount anomaly
        amount_anomaly = self._detect_amount_anomaly(
            transaction.amount,
            pattern,
            factors
        )
        
        # Frequency anomaly
        frequency_anomaly = self._detect_frequency_anomaly(
            transaction,
            pattern,
            factors
        )
        
        # Time anomaly
        time_anomaly = self._detect_time_anomaly(
            transaction.timestamp,
            pattern,
            factors
        )
        
        # Category anomaly
        category_anomaly = self._detect_category_anomaly(
            transaction.merchant_category,
            pattern,
            factors
        )
        
        # Location anomaly
        location_anomaly = self._detect_location_anomaly(
            transaction.location,
            pattern,
            factors
        )
        
        # Velocity anomaly
        velocity_anomaly = self._detect_velocity_anomaly(
            transaction,
            pattern,
            factors
        )
        
        # Calculate overall score (weighted average)
        weights = {
            "amount": 0.25,
            "frequency": 0.15,
            "time": 0.15,
            "category": 0.15,
            "location": 0.20,
            "velocity": 0.10,
        }
        
        overall_score = (
            amount_anomaly * weights["amount"] +
            frequency_anomaly * weights["frequency"] +
            time_anomaly * weights["time"] +
            category_anomaly * weights["category"] +
            location_anomaly * weights["location"] +
            velocity_anomaly * weights["velocity"]
        )
        
        anomaly = AnomalyScore(
            overall_score=overall_score,
            amount_anomaly=amount_anomaly,
            frequency_anomaly=frequency_anomaly,
            time_anomaly=time_anomaly,
            category_anomaly=category_anomaly,
            location_anomaly=location_anomaly,
            velocity_anomaly=velocity_anomaly,
            factors=factors,
        )
        
        logger.info(
            "Anomaly detection completed",
            extra={
                "transaction_id": transaction.id,
                "overall_score": overall_score,
                "is_anomalous": anomaly.is_anomalous,
                "factors": factors,
            }
        )
        
        return anomaly
    
    def _detect_amount_anomaly(
        self,
        amount: float,
        pattern: SpendingPattern,
        factors: List[str]
    ) -> float:
        """Detect anomaly in transaction amount."""
        
        if pattern.avg_amount == 0:
            return 0.3  # Unknown user, moderate baseline
        
        # Z-score calculation
        if pattern.std_amount > 0:
            z_score = abs(amount - pattern.avg_amount) / pattern.std_amount
        else:
            z_score = 0.0
        
        # Amount ratio to historical max
        if pattern.max_amount > 0:
            max_ratio = amount / pattern.max_amount
        else:
            max_ratio = 1.0
        
        # Calculate anomaly score
        if z_score >= self.z_score_threshold * 2:
            factors.append(f"Amount {z_score:.1f} standard deviations from mean")
            return 0.9
        elif z_score >= self.z_score_threshold:
            factors.append(f"Amount {z_score:.1f} standard deviations from mean")
            return 0.6
        elif max_ratio > 2.0:
            factors.append(f"Amount {max_ratio:.1f}x higher than historical max")
            return 0.5
        elif max_ratio > 1.5:
            return 0.3
        
        return 0.1
    
    def _detect_frequency_anomaly(
        self,
        transaction: Transaction,
        pattern: SpendingPattern,
        factors: List[str]
    ) -> float:
        """Detect anomaly in transaction frequency."""
        
        # Would need recent transaction count to calculate properly
        # This is a simplified version
        
        if pattern.transactions_per_day > self.max_transactions_per_day:
            factors.append("Excessive daily transaction frequency")
            return 0.8
        elif pattern.transactions_per_day > self.max_transactions_per_day / 2:
            return 0.4
        
        return 0.1
    
    def _detect_time_anomaly(
        self,
        timestamp: datetime,
        pattern: SpendingPattern,
        factors: List[str]
    ) -> float:
        """Detect anomaly in transaction time."""
        
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # Check hourly pattern
        hour_probability = pattern.transactions_per_hour.get(hour, 0.0)
        
        # Check weekday pattern
        weekday_probability = pattern.weekday_distribution.get(weekday, 0.0)
        
        # Late night/early morning (2-5 AM) is inherently suspicious
        if 2 <= hour <= 5:
            if hour_probability < 0.01:
                factors.append("Transaction at unusual hour (late night)")
                return 0.7
            return 0.4
        
        # Check if this hour is unusual for user
        if hour_probability < 0.01 and len(pattern.transactions_per_hour) > 0:
            factors.append(f"Transaction at unusual hour ({hour}:00)")
            return 0.5
        
        return 0.1
    
    def _detect_category_anomaly(
        self,
        category: str,
        pattern: SpendingPattern,
        factors: List[str]
    ) -> float:
        """Detect anomaly in merchant category."""
        
        if not pattern.category_distribution:
            return 0.2  # No history, slight uncertainty
        
        category_probability = pattern.category_distribution.get(category, 0.0)
        
        if category_probability == 0:
            factors.append(f"First transaction in category: {category}")
            return 0.5
        elif category_probability < 0.05:
            factors.append(f"Rare category for user: {category}")
            return 0.3
        
        return 0.1
    
    def _detect_location_anomaly(
        self,
        location: Optional[Dict[str, Any]],
        pattern: SpendingPattern,
        factors: List[str]
    ) -> float:
        """Detect anomaly in transaction location."""
        
        if not location:
            return 0.2  # Unknown location
        
        city = location.get("city", "unknown")
        
        if not pattern.location_distribution:
            return 0.2  # No history
        
        location_probability = pattern.location_distribution.get(city, 0.0)
        
        if location_probability == 0:
            # New location
            known_locations = list(pattern.location_distribution.keys())
            
            # Check if country is at least familiar
            country = location.get("country", "")
            same_country = any(
                country in loc for loc in known_locations
            )
            
            if same_country:
                factors.append(f"New city in familiar country: {city}")
                return 0.4
            else:
                factors.append(f"Transaction from unfamiliar location: {city}")
                return 0.7
        elif location_probability < 0.05:
            factors.append(f"Rare location: {city}")
            return 0.3
        
        return 0.1
    
    def _detect_velocity_anomaly(
        self,
        transaction: Transaction,
        pattern: SpendingPattern,
        factors: List[str]
    ) -> float:
        """Detect velocity-based anomalies."""
        
        # Would need access to recent transaction history
        # Simplified version based on pattern
        
        if pattern.transactions_per_day > 10:
            factors.append("High transaction velocity detected")
            return 0.6
        elif pattern.transactions_per_day > 5:
            return 0.3
        
        return 0.1
    
    def _parse_timestamp(
        self,
        value: Any
    ) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        
        if isinstance(value, datetime):
            return value
        
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
        
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value)
            except (ValueError, OSError):
                pass
        
        return None
    
    def analyze_transaction_batch(
        self,
        transactions: List[Transaction],
        pattern: SpendingPattern
    ) -> Dict[str, AnomalyScore]:
        """
        Analyze a batch of transactions for anomalies.
        
        Args:
            transactions: List of transactions
            pattern: User's spending pattern
            
        Returns:
            Dictionary mapping transaction IDs to anomaly scores
        """
        results = {}
        
        for transaction in transactions:
            results[transaction.id] = self.detect_anomaly(transaction, pattern)
        
        return results
