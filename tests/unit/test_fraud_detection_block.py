"""
Unit Tests for Fraud Detection Block (Business Logic Layer)

Tests cover:
- Core fraud detection flow
- Error handling and fallbacks
- Risk score calculation
- Decision logic
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.block.fraud_detection_block import (
    FraudDetectionBlock,
    FraudDetectionConfig,
)
from src.block.models import (
    Transaction,
    DeviceFingerprint,
    FraudDetectionResult,
    PaymentValidationResult,
    MLPrediction,
    UserProfile,
    Decision,
    RiskLevel,
    DeviceAttributes,
    NetworkAttributes,
    BehavioralAttributes,
)
from src.utils.error_handling import DeviceFingerprintError, MLModelError


def create_mock_dependencies():
    """Create mock dependencies for FraudDetectionBlock."""
    # Mock device fingerprint service
    device_fp_service = Mock()
    device_fp_service.generate.return_value = DeviceFingerprint(
        id="fp_test_123",
        confidence=0.9,
        risk_score=0.1,
        collision_risk=0.1,
        device_attrs=DeviceAttributes(platform="Windows"),
        network_attrs=NetworkAttributes(ip_address="192.168.1.1"),
        behavioral_attrs=BehavioralAttributes(),
        is_known_device=True,
    )
    
    # Mock payment validator
    payment_validator = Mock()
    payment_validator.validate.return_value = PaymentValidationResult(
        approved=True,
        risk_score=0.1,
        validation_factors={},
    )
    
    # Mock ML model service
    ml_model_service = Mock()
    ml_model_service.predict.return_value = {
        "xgboost": MLPrediction(
            model_name="xgboost",
            model_version="1.0",
            fraud_probability=0.1,
            confidence=0.9,
        ),
        "lightgbm": MLPrediction(
            model_name="lightgbm",
            model_version="1.0",
            fraud_probability=0.08,
            confidence=0.85,
        ),
    }
    
    # Mock telemetry
    telemetry = Mock()
    telemetry.timer.return_value.__enter__ = Mock()
    telemetry.timer.return_value.__exit__ = Mock(return_value=False)
    telemetry.counter = Mock()
    
    # Mock user profile repository
    user_profile_repo = Mock()
    user_profile_repo.get_by_user_id.return_value = UserProfile(
        user_id="user_123",
        known_devices=[],
        frequent_locations=[{"country": "US", "city": "New York"}],
        avg_transaction_amount=100.0,
    )
    
    return {
        "device_fingerprint_service": device_fp_service,
        "payment_validator": payment_validator,
        "ml_model_service": ml_model_service,
        "telemetry": telemetry,
        "user_profile_repository": user_profile_repo,
    }


class TestFraudDetectionBlock:
    """Tests for FraudDetectionBlock business logic."""
    
    def test_detect_fraud_approve_low_risk(self):
        """Test fraud detection approves low-risk transaction."""
        deps = create_mock_dependencies()
        block = FraudDetectionBlock(**deps)
        
        transaction = Transaction(
            id="txn_001",
            user_id="user_123",
            amount=100.0,
            currency="USD",
            features={"V1": 0.5, "V2": -0.3},
        )
        
        result = block.detect_fraud(transaction, request_id="req_001")
        
        assert isinstance(result, FraudDetectionResult)
        assert result.decision == Decision.APPROVE
        assert result.risk_level == RiskLevel.LOW
        assert result.risk_score < 30.0
        assert result.request_id == "req_001"
    
    def test_detect_fraud_deny_high_risk(self):
        """Test fraud detection denies high-risk transaction."""
        deps = create_mock_dependencies()
        
        # Configure high fraud probability
        deps["ml_model_service"].predict.return_value = {
            "xgboost": MLPrediction("xgboost", "1.0", fraud_probability=0.95, confidence=0.9),
            "lightgbm": MLPrediction("lightgbm", "1.0", fraud_probability=0.92, confidence=0.88),
        }
        deps["device_fingerprint_service"].generate.return_value = DeviceFingerprint(
            id="fp_risky",
            confidence=0.5,
            risk_score=0.8,
            collision_risk=0.3,
            is_known_device=False,
            device_attrs=DeviceAttributes(),
            network_attrs=NetworkAttributes(is_vpn=True, is_tor=True),
            behavioral_attrs=BehavioralAttributes(),
        )
        deps["payment_validator"].validate.return_value = PaymentValidationResult(
            approved=False,
            risk_score=0.9,
        )
        
        block = FraudDetectionBlock(**deps)
        
        transaction = Transaction(
            id="txn_002",
            user_id="user_456",
            amount=50000.0,
            features={"V1": 2.0, "V2": 3.0},
        )
        
        result = block.detect_fraud(transaction)
        
        assert result.decision == Decision.DENY
        assert result.risk_level == RiskLevel.CRITICAL
        assert result.risk_score >= 90.0
    
    def test_detect_fraud_review_medium_risk(self):
        """Test fraud detection flags medium-risk for review."""
        deps = create_mock_dependencies()
        
        # Configure medium fraud probability
        deps["ml_model_service"].predict.return_value = {
            "xgboost": MLPrediction("xgboost", "1.0", fraud_probability=0.6, confidence=0.7),
            "lightgbm": MLPrediction("lightgbm", "1.0", fraud_probability=0.65, confidence=0.72),
        }
        deps["device_fingerprint_service"].generate.return_value = DeviceFingerprint(
            id="fp_medium",
            confidence=0.7,
            risk_score=0.5,
            is_known_device=False,
            device_attrs=DeviceAttributes(),
            network_attrs=NetworkAttributes(),
            behavioral_attrs=BehavioralAttributes(),
            collision_risk=0.2,
        )
        deps["payment_validator"].validate.return_value = PaymentValidationResult(
            approved=True,
            risk_score=0.5,
        )
        
        block = FraudDetectionBlock(**deps)
        
        transaction = Transaction(
            id="txn_003",
            user_id="user_789",
            amount=5000.0,
            features={},
        )
        
        result = block.detect_fraud(transaction)
        
        # Should be review or additional auth
        assert result.decision in (Decision.REVIEW, Decision.ADDITIONAL_AUTH_REQUIRED)
        assert result.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)
    
    def test_detect_fraud_device_fingerprint_failure_fallback(self):
        """Test fallback when device fingerprinting fails."""
        deps = create_mock_dependencies()
        
        # Simulate device fingerprint failure
        deps["device_fingerprint_service"].generate.side_effect = Exception("Fingerprint service unavailable")
        
        block = FraudDetectionBlock(**deps)
        
        transaction = Transaction(
            id="txn_004",
            user_id="user_000",
            amount=200.0,
            features={},
        )
        
        result = block.detect_fraud(transaction, request_id="req_fail_fp")
        
        # Should use fallback (additional auth required)
        assert result.decision == Decision.ADDITIONAL_AUTH_REQUIRED
        assert result.risk_level == RiskLevel.MEDIUM
        assert "fallback" in str(result.explanation).lower() or "limited" in str(result.explanation).lower()
    
    def test_detect_fraud_ml_model_failure_fallback(self):
        """Test fallback when ML models fail."""
        deps = create_mock_dependencies()
        
        # Simulate ML model failure
        deps["ml_model_service"].predict.side_effect = MLModelError("Model inference failed")
        
        block = FraudDetectionBlock(**deps)
        
        transaction = Transaction(
            id="txn_005",
            user_id="user_111",
            amount=300.0,
            features={},
        )
        
        result = block.detect_fraud(transaction, request_id="req_fail_ml")
        
        # Should use rule-based fallback
        assert result.decision in (Decision.APPROVE, Decision.ADDITIONAL_AUTH_REQUIRED)
        assert "rule" in str(result.explanation).lower() or "fallback" in str(result.explanation).lower()
    
    def test_detect_fraud_generates_explanation(self):
        """Test that fraud detection generates explanation."""
        deps = create_mock_dependencies()
        block = FraudDetectionBlock(**deps)
        
        transaction = Transaction(
            id="txn_006",
            user_id="user_222",
            amount=500.0,
            features={},
        )
        
        result = block.detect_fraud(transaction)
        
        assert result.explanation
        assert "summary" in result.explanation
        assert result.explanation["summary"]  # Not empty


class TestFraudDetectionConfig:
    """Tests for FraudDetectionConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FraudDetectionConfig()
        
        assert config.device_weight == 0.25
        assert config.ml_weight == 0.40
        assert config.approve_threshold == 30.0
        assert config.deny_threshold == 90.0
        assert config.model_version == "2.9.0"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = FraudDetectionConfig(
            device_weight=0.30,
            ml_weight=0.50,
            approve_threshold=20.0,
            model_version="3.0.0",
        )
        
        assert config.device_weight == 0.30
        assert config.ml_weight == 0.50
        assert config.approve_threshold == 20.0
        assert config.model_version == "3.0.0"


class TestRiskScoreCalculation:
    """Tests for risk score calculation logic."""
    
    def test_weighted_average_calculation(self):
        """Test weighted average risk calculation."""
        deps = create_mock_dependencies()
        config = FraudDetectionConfig(
            device_weight=0.25,
            payment_weight=0.25,
            ml_weight=0.40,
            behavioral_weight=0.10,
        )
        block = FraudDetectionBlock(**deps, config=config)
        
        # All components report 50% risk
        deps["device_fingerprint_service"].generate.return_value = DeviceFingerprint(
            id="fp_mid",
            confidence=0.5,
            risk_score=0.5,
            is_known_device=False,
            device_attrs=DeviceAttributes(),
            network_attrs=NetworkAttributes(),
            behavioral_attrs=BehavioralAttributes(),
            collision_risk=0.2,
        )
        deps["payment_validator"].validate.return_value = PaymentValidationResult(
            approved=True,
            risk_score=0.5,
        )
        deps["ml_model_service"].predict.return_value = {
            "xgboost": MLPrediction("xgboost", "1.0", fraud_probability=0.5, confidence=0.8),
        }
        
        transaction = Transaction(
            id="txn_weighted",
            user_id="user_weight",
            amount=1000.0,
            features={},
        )
        
        result = block.detect_fraud(transaction)
        
        # Should be around 50 given equal risk from all components
        assert 30 <= result.risk_score <= 70


class TestFromErrorFallback:
    """Tests for FraudDetectionResult.from_error fallback."""
    
    def test_from_error_creates_review_decision(self):
        """Test from_error creates safe review decision."""
        result = FraudDetectionResult.from_error(
            error="Unexpected error occurred",
            request_id="req_error"
        )
        
        assert result.decision == Decision.REVIEW
        assert result.risk_score == 100.0
        assert result.risk_level == RiskLevel.CRITICAL
        assert result.request_id == "req_error"
        assert "error" in result.explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
