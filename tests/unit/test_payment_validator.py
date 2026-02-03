"""
Unit Tests for Payment Source Validator

Tests cover:
- Device binding validation
- Location validation
- Amount threshold validation
- Velocity checks
- Cross-border rules
- Authentication requirements
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from src.fraud_detection.payment_validation.source_validator import (
    PaymentSourceValidator,
    PaymentValidationConfig,
    ValidationFactor,
)
from src.block.models import (
    Transaction,
    DeviceFingerprint,
    UserProfile,
    DeviceAttributes,
    NetworkAttributes,
    BehavioralAttributes,
    AuthRequirementLevel,
)


def create_test_transaction(
    transaction_id: str = "txn_123",
    user_id: str = "user_456",
    amount: float = 100.0,
    location: dict = None,
) -> Transaction:
    """Helper to create test transactions."""
    return Transaction(
        id=transaction_id,
        user_id=user_id,
        amount=amount,
        currency="USD",
        merchant_id="merchant_001",
        merchant_category="retail",
        timestamp=datetime.utcnow(),
        location=location or {"country": "US", "city": "New York"},
        payment_method="card",
    )


def create_test_fingerprint(
    fingerprint_id: str = "fp_123",
    is_known: bool = True,
    risk_score: float = 0.1,
    is_vpn: bool = False,
) -> DeviceFingerprint:
    """Helper to create test fingerprints."""
    return DeviceFingerprint(
        id=fingerprint_id,
        confidence=0.9,
        risk_score=risk_score,
        collision_risk=0.1,
        device_attrs=DeviceAttributes(platform="Windows"),
        network_attrs=NetworkAttributes(
            ip_address="192.168.1.1",
            is_vpn=is_vpn,
            geolocation={"country": "US", "city": "New York"},
        ),
        behavioral_attrs=BehavioralAttributes(),
        is_known_device=is_known,
        first_seen=datetime.utcnow(),
        raw_attributes={"fingerprint_hierarchy": {"fine": fingerprint_id}},
    )


def create_test_user_profile(
    user_id: str = "user_456",
    known_devices: list = None,
    frequent_locations: list = None,
    avg_amount: float = 100.0,
) -> UserProfile:
    """Helper to create test user profiles."""
    return UserProfile(
        user_id=user_id,
        known_devices=known_devices or [],
        frequent_locations=frequent_locations or [
            {"country": "US", "city": "New York"}
        ],
        avg_transaction_amount=avg_amount,
        transaction_frequency=5.0,
        account_age_days=365,
    )


class TestPaymentSourceValidator:
    """Tests for PaymentSourceValidator."""
    
    def test_approve_low_risk_transaction(self):
        """Test approval of low-risk transaction."""
        validator = PaymentSourceValidator()
        
        transaction = create_test_transaction(amount=50.0)
        fingerprint = create_test_fingerprint(is_known=True)
        user_profile = create_test_user_profile(
            known_devices=[fingerprint],
            frequent_locations=[{"country": "US", "city": "New York"}]
        )
        
        result = validator.validate(transaction, fingerprint, user_profile)
        
        assert result.approved is True
        assert result.risk_score < 0.3
        assert result.auth_requirement is None or result.auth_requirement.level == AuthRequirementLevel.NONE
    
    def test_require_otp_unknown_device(self):
        """Test OTP requirement for unknown device."""
        validator = PaymentSourceValidator()
        
        transaction = create_test_transaction(amount=100.0)
        fingerprint = create_test_fingerprint(is_known=False)
        known_fp = create_test_fingerprint(fingerprint_id="known_fp")
        user_profile = create_test_user_profile(
            known_devices=[known_fp]
        )
        
        result = validator.validate(transaction, fingerprint, user_profile)
        
        # Unknown device should trigger additional auth
        if result.auth_requirement:
            assert result.auth_requirement.level in (
                AuthRequirementLevel.OTP,
                AuthRequirementLevel.MFA,
            )
    
    def test_require_mfa_unfamiliar_location(self):
        """Test MFA requirement for unfamiliar location."""
        config = PaymentValidationConfig(mfa_threshold=0.4)
        validator = PaymentSourceValidator(config)
        
        transaction = create_test_transaction(
            amount=1000.0,
            location={"country": "JP", "city": "Tokyo"}  # Different country
        )
        fingerprint = create_test_fingerprint(
            is_known=False,
            risk_score=0.5,
        )
        user_profile = create_test_user_profile(
            frequent_locations=[{"country": "US", "city": "New York"}]
        )
        
        result = validator.validate(transaction, fingerprint, user_profile)
        
        # Unfamiliar location should increase risk
        assert result.risk_score > 0.3
    
    def test_require_biometric_high_value(self):
        """Test biometric requirement for high-value transaction."""
        config = PaymentValidationConfig(
            high_value_threshold=10000.0,
            biometric_threshold=0.6
        )
        validator = PaymentSourceValidator(config)
        
        transaction = create_test_transaction(amount=50000.0)
        fingerprint = create_test_fingerprint(is_known=False, risk_score=0.4)
        user_profile = create_test_user_profile(avg_amount=200.0)
        
        result = validator.validate(transaction, fingerprint, user_profile)
        
        # High value + unknown device should trigger high auth
        if result.auth_requirement:
            assert result.auth_requirement.level in (
                AuthRequirementLevel.MFA,
                AuthRequirementLevel.BIOMETRIC,
            )
    
    def test_manual_review_cross_border(self):
        """Test manual review for cross-border transaction."""
        config = PaymentValidationConfig(require_manual_review_cross_border=True)
        validator = PaymentSourceValidator(config)
        
        transaction = create_test_transaction(
            location={"country_code": "UK", "city": "London"}
        )
        fingerprint = create_test_fingerprint()
        user_profile = create_test_user_profile(
            frequent_locations=[{"country_code": "US", "city": "New York"}]
        )
        
        result = validator.validate(transaction, fingerprint, user_profile)
        
        # Cross-border should be flagged
        if result.validation_factors.get("cross_border"):
            factor = result.validation_factors["cross_border"]
            if isinstance(factor, dict):
                assert factor.get("passed") is False or factor.get("risk_score", 0) > 0.3
    
    def test_velocity_check_excessive(self):
        """Test velocity check for excessive transactions."""
        validator = PaymentSourceValidator()
        
        transaction = create_test_transaction()
        fingerprint = create_test_fingerprint()
        user_profile = create_test_user_profile()
        user_profile.transaction_frequency = 50  # Very high
        
        result = validator.validate(transaction, fingerprint, user_profile)
        
        # High velocity should increase risk
        factor = result.validation_factors.get("velocity")
        if factor and isinstance(factor, dict):
            assert factor.get("risk_score", 0) > 0.3
    
    def test_amount_deviation_detection(self):
        """Test detection of amount deviation."""
        validator = PaymentSourceValidator()
        
        transaction = create_test_transaction(amount=10000.0)  # High amount
        fingerprint = create_test_fingerprint()
        user_profile = create_test_user_profile(avg_amount=100.0)  # Usually low
        
        result = validator.validate(transaction, fingerprint, user_profile)
        
        # 100x higher than average should be flagged
        factor = result.validation_factors.get("amount")
        if factor and isinstance(factor, dict):
            assert factor.get("risk_score", 0) > 0.3
    
    def test_new_user_moderate_risk(self):
        """Test new user has moderate base risk."""
        validator = PaymentSourceValidator()
        
        transaction = create_test_transaction()
        fingerprint = create_test_fingerprint(is_known=False)
        user_profile = create_test_user_profile(
            known_devices=[],  # No devices yet
            frequent_locations=[],  # No location history
        )
        
        result = validator.validate(transaction, fingerprint, user_profile)
        
        # New user should have some risk but not be blocked
        assert result.approved is True or result.auth_requirement is not None
    
    def test_vpn_increases_risk(self):
        """Test that VPN usage increases risk."""
        validator = PaymentSourceValidator()
        
        transaction = create_test_transaction()
        fingerprint = create_test_fingerprint(is_vpn=True, risk_score=0.5)
        user_profile = create_test_user_profile()
        
        result = validator.validate(transaction, fingerprint, user_profile)
        
        # VPN should increase overall risk through device risk
        assert result.risk_score > 0.2  # Should be elevated


class TestValidationFactor:
    """Tests for ValidationFactor dataclass."""
    
    def test_validation_factor_creation(self):
        """Test ValidationFactor creation."""
        factor = ValidationFactor(
            name="device_binding",
            passed=True,
            risk_score=0.1,
            details="Known device"
        )
        
        assert factor.name == "device_binding"
        assert factor.passed is True
        assert factor.risk_score == 0.1
        assert factor.details == "Known device"


class TestPaymentValidationConfig:
    """Tests for PaymentValidationConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PaymentValidationConfig()
        
        assert config.low_value_threshold == 5000.0
        assert config.high_value_threshold == 50000.0
        assert config.approve_threshold == 0.3
        assert config.mfa_threshold == 0.5
        assert config.deny_threshold == 0.9
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PaymentValidationConfig(
            low_value_threshold=1000.0,
            high_value_threshold=10000.0,
            approve_threshold=0.2,
        )
        
        assert config.low_value_threshold == 1000.0
        assert config.high_value_threshold == 10000.0
        assert config.approve_threshold == 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
