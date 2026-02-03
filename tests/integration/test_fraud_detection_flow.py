"""
Integration Tests for Fraud Detection Flow

End-to-end tests for complete fraud detection pipeline.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from src.fraud_detection.device_fingerprinting import VisitorFingerprintEngine
from src.fraud_detection.payment_validation import PaymentSourceValidator
from src.fraud_detection.behavioral import SpendingPatternAnalyzer
from src.block.models import Transaction, UserProfile


class TestFraudDetectionFlow:
    """End-to-end fraud detection tests."""
    
    def test_complete_low_risk_flow(self):
        """Test complete flow for low-risk transaction."""
        # Step 1: Generate fingerprint
        fp_engine = VisitorFingerprintEngine()
        fingerprint = fp_engine.generate_fingerprint({
            "headers": {"user-agent": "Mozilla/5.0 Chrome/120"},
            "client": ("192.168.1.100", 54321),
            "client_data": {
                "screen_resolution": "1920x1080",
                "canvas_fingerprint": "canvas123",
            },
        })
        
        assert fingerprint.id
        assert fingerprint.confidence > 0
        
        # Step 2: Validate payment
        validator = PaymentSourceValidator()
        user_profile = UserProfile(
            user_id="user_001",
            known_devices=[fingerprint],
            frequent_locations=[{"country": "US", "city": "New York"}],
            avg_transaction_amount=100.0,
        )
        transaction = Transaction(
            id="txn_001",
            user_id="user_001",
            amount=50.0,
            location={"country": "US", "city": "New York"},
        )
        
        validation = validator.validate(transaction, fingerprint, user_profile)
        
        assert validation.approved is True or validation.auth_requirement is not None
        
        # Step 3: Analyze spending pattern
        analyzer = SpendingPatternAnalyzer()
        pattern = analyzer.build_pattern("user_001", [
            {"amount": 80.0, "timestamp": datetime.utcnow().isoformat()},
            {"amount": 100.0, "timestamp": datetime.utcnow().isoformat()},
        ])
        anomaly = analyzer.detect_anomaly(transaction, pattern)
        
        assert anomaly.overall_score < 0.6  # Should not be anomalous


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
