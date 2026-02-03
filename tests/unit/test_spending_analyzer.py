"""
Unit Tests for Spending Pattern Analyzer

Tests cover:
- Pattern building from history
- Anomaly detection
- Statistical analysis
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.fraud_detection.behavioral.spending_pattern_analyzer import (
    SpendingPatternAnalyzer,
    SpendingPattern,
    AnomalyScore,
)
from src.block.models import Transaction


def create_transaction_history(
    user_id: str,
    count: int = 100,
    avg_amount: float = 100.0,
    std_amount: float = 20.0,
) -> List[Dict[str, Any]]:
    """Create synthetic transaction history for testing."""
    import random
    random.seed(42)  # For reproducibility
    
    history = []
    now = datetime.utcnow()
    
    for i in range(count):
        history.append({
            "transaction_id": f"txn_{i}",
            "amount": max(10, random.gauss(avg_amount, std_amount)),
            "timestamp": (now - timedelta(days=random.randint(0, 90))).isoformat(),
            "merchant_category": random.choice(["retail", "grocery", "restaurant"]),
            "location": {"city": random.choice(["New York", "Boston", "Chicago"])},
        })
    
    return history


class TestSpendingPatternAnalyzer:
    """Tests for SpendingPatternAnalyzer."""
    
    def test_build_pattern_basic(self):
        """Test basic pattern building."""
        analyzer = SpendingPatternAnalyzer()
        
        history = create_transaction_history(
            user_id="user_001",
            count=50,
            avg_amount=100.0,
        )
        
        pattern = analyzer.build_pattern("user_001", history)
        
        assert isinstance(pattern, SpendingPattern)
        assert pattern.user_id == "user_001"
        assert 80 <= pattern.avg_amount <= 120  # Around 100
        assert pattern.transactions_per_day > 0
        assert len(pattern.category_distribution) > 0
    
    def test_build_pattern_empty_history(self):
        """Test pattern building with empty history."""
        analyzer = SpendingPatternAnalyzer()
        
        pattern = analyzer.build_pattern("user_new", [])
        
        assert pattern.user_id == "user_new"
        assert pattern.avg_amount == 0.0
        assert pattern.transactions_per_day == 0.0
    
    def test_detect_amount_anomaly(self):
        """Test detection of abnormal transaction amount."""
        analyzer = SpendingPatternAnalyzer()
        
        # Build pattern with avg = 100
        history = create_transaction_history(
            user_id="user_002",
            count=100,
            avg_amount=100.0,
            std_amount=20.0,
        )
        pattern = analyzer.build_pattern("user_002", history)
        
        # Transaction 10x higher than average
        anomalous_transaction = Transaction(
            id="txn_anomalous",
            user_id="user_002",
            amount=1000.0,  # 10x average
            timestamp=datetime.utcnow(),
        )
        
        anomaly = analyzer.detect_anomaly(anomalous_transaction, pattern)
        
        assert isinstance(anomaly, AnomalyScore)
        assert anomaly.amount_anomaly > 0.5  # High amount anomaly
        assert anomaly.overall_score > 0.3  # Elevated overall
    
    def test_detect_normal_transaction(self):
        """Test normal transaction is not flagged."""
        analyzer = SpendingPatternAnalyzer()
        
        history = create_transaction_history(
            user_id="user_003",
            count=100,
            avg_amount=100.0,
            std_amount=20.0,
        )
        pattern = analyzer.build_pattern("user_003", history)
        
        normal_transaction = Transaction(
            id="txn_normal",
            user_id="user_003",
            amount=95.0,  # Within normal range
            timestamp=datetime.utcnow(),
            merchant_category="retail",
            location={"city": "New York"},
        )
        
        anomaly = analyzer.detect_anomaly(normal_transaction, pattern)
        
        assert anomaly.amount_anomaly < 0.3
        assert anomaly.overall_score < 0.5
        assert anomaly.is_anomalous is False
    
    def test_detect_time_anomaly(self):
        """Test detection of unusual transaction time."""
        analyzer = SpendingPatternAnalyzer()
        
        # Create history with daytime transactions only
        history = []
        now = datetime.utcnow()
        for i in range(50):
            t = now - timedelta(days=i)
            t = t.replace(hour=14)  # Always at 2 PM
            history.append({
                "transaction_id": f"txn_{i}",
                "amount": 100.0,
                "timestamp": t.isoformat(),
                "merchant_category": "retail",
            })
        
        pattern = analyzer.build_pattern("user_004", history)
        
        # Transaction at 3 AM
        night_transaction = Transaction(
            id="txn_night",
            user_id="user_004",
            amount=100.0,
            timestamp=datetime.utcnow().replace(hour=3),
        )
        
        anomaly = analyzer.detect_anomaly(night_transaction, pattern)
        
        # Late night should be flagged
        assert anomaly.time_anomaly > 0.3
    
    def test_detect_location_anomaly(self):
        """Test detection of unusual location."""
        analyzer = SpendingPatternAnalyzer()
        
        # History with NYC transactions only
        history = [
            {
                "transaction_id": f"txn_{i}",
                "amount": 100.0,
                "timestamp": (datetime.utcnow() - timedelta(days=i)).isoformat(),
                "location": {"city": "New York", "country": "US"},
            }
            for i in range(50)
        ]
        
        pattern = analyzer.build_pattern("user_005", history)
        
        # Transaction from different city
        foreign_transaction = Transaction(
            id="txn_foreign",
            user_id="user_005",
            amount=100.0,
            timestamp=datetime.utcnow(),
            location={"city": "Tokyo", "country": "Japan"},
        )
        
        anomaly = analyzer.detect_anomaly(foreign_transaction, pattern)
        
        # New location should be flagged
        assert anomaly.location_anomaly > 0.3
    
    def test_detect_category_anomaly(self):
        """Test detection of unusual merchant category."""
        analyzer = SpendingPatternAnalyzer()
        
        # History with retail transactions only
        history = [
            {
                "transaction_id": f"txn_{i}",
                "amount": 100.0,
                "timestamp": (datetime.utcnow() - timedelta(days=i)).isoformat(),
                "merchant_category": "retail",
            }
            for i in range(50)
        ]
        
        pattern = analyzer.build_pattern("user_006", history)
        
        # Transaction in different category
        gambling_transaction = Transaction(
            id="txn_gambling",
            user_id="user_006",
            amount=100.0,
            timestamp=datetime.utcnow(),
            merchant_category="gambling",
        )
        
        anomaly = analyzer.detect_anomaly(gambling_transaction, pattern)
        
        # New category should be flagged
        assert anomaly.category_anomaly > 0.3
    
    def test_anomaly_score_is_anomalous_property(self):
        """Test is_anomalous property."""
        # High score = anomalous
        high_score = AnomalyScore(
            overall_score=0.7,
            amount_anomaly=0.5,
            frequency_anomaly=0.5,
            time_anomaly=0.5,
            category_anomaly=0.5,
            location_anomaly=0.5,
            velocity_anomaly=0.5,
        )
        assert high_score.is_anomalous is True
        
        # Low score = not anomalous
        low_score = AnomalyScore(
            overall_score=0.2,
            amount_anomaly=0.1,
            frequency_anomaly=0.1,
            time_anomaly=0.1,
            category_anomaly=0.1,
            location_anomaly=0.1,
            velocity_anomaly=0.1,
        )
        assert low_score.is_anomalous is False


class TestSpendingPattern:
    """Tests for SpendingPattern dataclass."""
    
    def test_default_values(self):
        """Test default values for SpendingPattern."""
        pattern = SpendingPattern(user_id="test")
        
        assert pattern.user_id == "test"
        assert pattern.avg_amount == 0.0
        assert pattern.std_amount == 0.0
        assert pattern.transactions_per_day == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
