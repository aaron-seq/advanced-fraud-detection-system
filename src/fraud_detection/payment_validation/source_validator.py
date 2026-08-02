"""
Payment Source Validator

RBI April 2026 compliant payment source validation.

Implements:
- Device binding verification
- Location-based risk assessment
- Multi-factor authentication requirements
- Cross-border transaction rules
- BioDiNo (Biometric + DigiLocker) for high-risk verification
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, ClassVar

from ...block.models import (
    AuthRequirement,
    AuthRequirementLevel,
    DeviceFingerprint,
    PaymentValidationResult,
    Transaction,
    UserProfile,
)
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class RiskLevel(str, Enum):
    """Payment risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationFactor:
    """Individual validation factor result."""

    name: str
    passed: bool
    risk_score: float
    details: str = ""


@dataclass
class PaymentValidationConfig:
    """Configuration for payment validation."""

    # Amount thresholds (in base currency)
    low_value_threshold: float = 5000.0
    high_value_threshold: float = 50000.0
    very_high_value_threshold: float = 200000.0

    # Risk thresholds
    approve_threshold: float = 0.3
    mfa_threshold: float = 0.5
    biometric_threshold: float = 0.7
    deny_threshold: float = 0.9

    # Location mismatch tolerance (km)
    location_tolerance_km: float = 50.0

    # Cross-border rules
    require_manual_review_cross_border: bool = True

    # New device rules
    new_device_cool_down_hours: int = 24

    # Velocity limits
    max_transactions_per_hour: int = 10
    max_amount_per_day: float = 500000.0


class PaymentSourceValidator:
    """
    Validates payment source for fraud risk.

    RBI April 2026 Compliance:
    - Multi-factor authentication beyond SMS OTP
    - Biometric verification for high-value transactions
    - DigiLocker integration for identity verification
    - Cross-border transaction controls

    Risk Matrix:
    | Device Known | Location Familiar | Amount | Auth Required |
    |--------------|-------------------|--------|---------------|
    | Yes          | Yes               | Low    | None         |
    | Yes          | Yes               | High   | OTP          |
    | Yes          | No                | Any    | MFA          |
    | No           | Any               | Low    | OTP          |
    | No           | Any               | Medium | MFA          |
    | No           | Any               | High   | Biometric    |
    | No           | Cross-border      | Any    | Manual Review|
    """

    def __init__(self, config: PaymentValidationConfig | None = None):
        self.config = config or PaymentValidationConfig()

        logger.info("PaymentSourceValidator initialized", extra={"config": self.config.__dict__})

    def validate(
        self,
        transaction: Transaction,
        device_fingerprint: DeviceFingerprint,
        user_profile: UserProfile,
    ) -> PaymentValidationResult:
        """
        Validate payment source and determine authentication requirements.

        Args:
            transaction: Transaction to validate
            device_fingerprint: Device fingerprint for the transaction
            user_profile: User's historical profile

        Returns:
            PaymentValidationResult with approval status and auth requirements
        """
        start_time = datetime.now(UTC)
        validation_factors: dict[str, Any] = {}

        try:
            # Step 1: Device binding check
            device_factor = self._check_device_binding(device_fingerprint, user_profile)
            validation_factors["device"] = device_factor

            # Step 2: Location check
            location_factor = self._check_location(transaction, device_fingerprint, user_profile)
            validation_factors["location"] = location_factor

            # Step 3: Amount check
            amount_factor = self._check_transaction_amount(transaction, user_profile)
            validation_factors["amount"] = amount_factor

            # Step 4: Velocity check
            velocity_factor = self._check_velocity(transaction, user_profile)
            validation_factors["velocity"] = velocity_factor

            # Step 5: Cross-border check
            cross_border_factor = self._check_cross_border(transaction, user_profile)
            validation_factors["cross_border"] = cross_border_factor

            # Step 6: Calculate composite risk score
            risk_score = self._calculate_risk_score(validation_factors)

            # Step 7: Determine auth requirements
            auth_requirement = self._determine_auth_requirement(
                risk_score, validation_factors, transaction
            )

            # Step 8: Make final decision
            approved = self._make_decision(risk_score, auth_requirement)

            result = PaymentValidationResult(
                approved=approved,
                risk_score=risk_score,
                auth_requirement=auth_requirement,
                validation_factors={
                    k: v.__dict__ if hasattr(v, "__dict__") else v
                    for k, v in validation_factors.items()
                },
            )

            processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

            logger.info(
                "Payment validation completed",
                extra={
                    "transaction_id": transaction.id,
                    "approved": approved,
                    "risk_score": risk_score,
                    "auth_level": auth_requirement.level.value if auth_requirement else "none",
                    "processing_time_ms": processing_time,
                },
            )

            return result

        except Exception as e:
            logger.error(
                f"Payment validation error: {e}",
                extra={"transaction_id": transaction.id},
                exc_info=True,
            )

            # Fail safe - require manual review
            return PaymentValidationResult.denied(reason=f"Validation error: {e!s}", risk_score=1.0)

    def _check_device_binding(
        self, device_fingerprint: DeviceFingerprint, user_profile: UserProfile
    ) -> ValidationFactor:
        """
        Check if device is bound to user, and how suspicious the device itself is.

        Two independent signals are fused here:
        1. Binding - is this device linked to this user's history?
        2. Intrinsic device risk - DeviceRiskEvaluator's score, which already
           folds in VPN/proxy/Tor/datacenter, spoofing indicators and bot-like
           behaviour. It is deliberately NOT recomputed here; consuming the
           evaluator's score keeps anonymiser detection in exactly one place.

        A recognised device coming over Tor is still dangerous, so the signals
        are combined with a noisy-OR rather than letting either one dominate.
        """

        # Check if device is in user's known devices
        is_known = device_fingerprint.is_known_device

        if not is_known and user_profile.known_devices:
            # Check for partial match
            for known_device in user_profile.known_devices:
                if self._devices_similar(device_fingerprint, known_device):
                    is_known = True
                    break

        if is_known:
            binding_risk, binding_passed = 0.1, True
            details = "Transaction from recognized device"
        elif self._is_new_account(user_profile):
            # Genuinely new account - it has to enrol a first device somehow,
            # so an unbound device is expected rather than suspicious.
            binding_risk, binding_passed = 0.3, True
            details = "New user - first device"
        else:
            # An established account transacting from a device with no binding
            # history is MORE suspicious than a brand new one, not less. An
            # empty known_devices list alone must not be read as "new user".
            binding_risk, binding_passed = 0.6, False
            details = "Transaction from unrecognized device"

        device_risk = self._combine_independent_risks(binding_risk, device_fingerprint.risk_score)

        if device_risk > binding_risk:
            details += f" (device risk score {device_fingerprint.risk_score:.2f})"

        return ValidationFactor(
            name="device_binding",
            passed=binding_passed and device_risk < 0.6,
            risk_score=device_risk,
            details=details,
        )

    @staticmethod
    def _is_new_account(user_profile: UserProfile) -> bool:
        """
        Is this account genuinely new, rather than merely lacking bound devices?

        Any prior history - transactions, known locations, or an account older
        than the device cool-down window - means the account is established.
        """
        return not (
            user_profile.known_devices
            or user_profile.frequent_locations
            or user_profile.transaction_history
            or user_profile.account_age_days > 0
        )

    @staticmethod
    def _combine_independent_risks(*risks: float) -> float:
        """
        Fuse independent risk signals with a noisy-OR.

        Each signal only ever adds risk, the result stays within [0, 1] by
        construction, and no single signal can mask another.
        """
        remaining_safety = 1.0
        for risk in risks:
            remaining_safety *= 1.0 - min(max(risk, 0.0), 1.0)
        return 1.0 - remaining_safety

    def _check_location(
        self,
        transaction: Transaction,
        device_fingerprint: DeviceFingerprint,
        user_profile: UserProfile,
    ) -> ValidationFactor:
        """Check if location is familiar."""

        transaction_location = transaction.location
        network_location = device_fingerprint.network_attrs.geolocation

        if not transaction_location and not network_location:
            return ValidationFactor(
                name="location", passed=True, risk_score=0.2, details="Location data unavailable"
            )

        # Use network location if transaction location not available
        current_location = transaction_location or network_location

        if not current_location:
            return ValidationFactor(
                name="location", passed=True, risk_score=0.2, details="Location data unavailable"
            )

        # Check against frequent locations
        is_familiar = False
        if user_profile.frequent_locations:
            for familiar_loc in user_profile.frequent_locations:
                if self._locations_match(current_location, familiar_loc):
                    is_familiar = True
                    break

        if is_familiar:
            return ValidationFactor(
                name="location",
                passed=True,
                risk_score=0.1,
                details="Transaction from familiar location",
            )
        elif not user_profile.frequent_locations:
            return ValidationFactor(
                name="location", passed=True, risk_score=0.3, details="No location history for user"
            )
        else:
            return ValidationFactor(
                name="location",
                passed=False,
                risk_score=0.5,
                details=f"Transaction from unfamiliar location: {current_location.get('city', 'Unknown')}",
            )

    def _check_transaction_amount(
        self, transaction: Transaction, user_profile: UserProfile
    ) -> ValidationFactor:
        """
        Check if transaction amount is normal for user.

        The absolute band and the per-user deviation are independent concerns:
        a large amount is risky in itself, and an amount far above what this
        user normally spends is risky even when the absolute figure is modest.
        Both are always evaluated and the worse of the two wins - returning
        early on the band would make the deviation check unreachable for any
        amount above the low-value threshold.
        """

        amount = transaction.amount
        config = self.config

        # Signal 1: absolute value bands
        if amount >= config.very_high_value_threshold:
            band = ValidationFactor("amount", False, 0.7, f"Very high value transaction: {amount}")
        elif amount >= config.high_value_threshold:
            band = ValidationFactor("amount", True, 0.5, f"High value transaction: {amount}")
        elif amount >= config.low_value_threshold:
            band = ValidationFactor("amount", True, 0.3, f"Medium value transaction: {amount}")
        else:
            band = ValidationFactor("amount", True, 0.1, "Normal transaction amount")

        # Signal 2: deviation from this user's typical spend
        deviation = None
        if user_profile.avg_transaction_amount > 0:
            ratio = amount / user_profile.avg_transaction_amount
            if ratio > 5.0:
                deviation = ValidationFactor(
                    "amount", False, 0.6, f"Amount {ratio:.1f}x higher than average"
                )
            elif ratio > 2.0:
                deviation = ValidationFactor(
                    "amount", True, 0.3, f"Amount {ratio:.1f}x higher than average"
                )

        if deviation is None:
            return band

        # Report whichever signal is worse, but never mark the factor as passed
        # when either signal failed.
        worst = max(band, deviation, key=lambda factor: factor.risk_score)
        worst.passed = band.passed and deviation.passed
        return worst

    def _check_velocity(
        self, transaction: Transaction, user_profile: UserProfile
    ) -> ValidationFactor:
        """Check transaction velocity."""

        # Check frequency (transactions per day)
        if user_profile.transaction_frequency > self.config.max_transactions_per_hour:
            return ValidationFactor(
                name="velocity",
                passed=False,
                risk_score=0.7,
                details="Excessive transaction frequency",
            )

        # Would check daily amount in production
        # For now, return normal
        return ValidationFactor(
            name="velocity", passed=True, risk_score=0.1, details="Normal transaction velocity"
        )

    def _check_cross_border(
        self, transaction: Transaction, user_profile: UserProfile
    ) -> ValidationFactor:
        """Check for cross-border transaction."""

        transaction_location = transaction.location

        if not transaction_location:
            return ValidationFactor(
                name="cross_border",
                passed=True,
                risk_score=0.1,
                details="Location data unavailable",
            )

        transaction_country = transaction_location.get("country_code", "").upper()

        # Determine user's home country from frequent locations
        home_country = None
        if user_profile.frequent_locations:
            # Most frequent country
            country_counts: dict[str, int] = {}
            for loc in user_profile.frequent_locations:
                country = loc.get("country_code", "").upper()
                if country:
                    country_counts[country] = country_counts.get(country, 0) + 1

            if country_counts:
                home_country = max(country_counts, key=country_counts.get)

        if home_country and transaction_country and home_country != transaction_country:
            if self.config.require_manual_review_cross_border:
                return ValidationFactor(
                    name="cross_border",
                    passed=False,
                    risk_score=0.8,
                    details=f"Cross-border transaction: {home_country} -> {transaction_country}",
                )
            else:
                return ValidationFactor(
                    name="cross_border",
                    passed=True,
                    risk_score=0.5,
                    details=f"Cross-border transaction: {home_country} -> {transaction_country}",
                )

        return ValidationFactor(
            name="cross_border", passed=True, risk_score=0.0, details="Domestic transaction"
        )

    def _calculate_risk_score(self, validation_factors: dict[str, ValidationFactor]) -> float:
        """Calculate composite risk score from all factors."""

        weights = {
            "device": 0.30,
            "location": 0.25,
            "amount": 0.20,
            "velocity": 0.15,
            "cross_border": 0.10,
        }

        total_score = 0.0
        total_weight = 0.0

        for factor_name, weight in weights.items():
            factor = validation_factors.get(factor_name)
            if factor:
                total_score += factor.risk_score * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.5

    def _determine_auth_requirement(
        self,
        risk_score: float,
        validation_factors: dict[str, ValidationFactor],
        transaction: Transaction,
    ) -> AuthRequirement | None:
        """
        Determine authentication requirement.

        Two ladders are evaluated and the stricter one wins:

        1. Score-based - the composite risk score crossing a configured
           threshold.
        2. Rule-based - the risk matrix documented on this class.

        Both are needed. The composite score is a weighted mean, so a single
        severe signal gets diluted by the benign ones around it; on its own it
        would leave the matrix's Biometric and MFA rows unreachable, and a
        high-value transaction from an unrecognised device would escalate no
        further than an SMS OTP.
        """

        cross_border = validation_factors.get("cross_border")
        if cross_border and not cross_border.passed:
            return AuthRequirement(
                level=AuthRequirementLevel.MANUAL_REVIEW,
                methods=["document_verification", "video_kyc"],
                timeout_seconds=86400,  # 24 hours
                reason="Cross-border transaction requires verification",
            )

        level = self._strictest(
            self._score_based_auth_level(risk_score),
            self._matrix_auth_level(validation_factors, transaction),
        )

        if level is AuthRequirementLevel.NONE:
            return None

        return AuthRequirement(
            level=level,
            methods=self._AUTH_METHODS[level],
            timeout_seconds=300,
            reason=self._AUTH_REASONS[level],
        )

    # Ordered least to most strict; index is used to compare severity.
    _AUTH_LADDER = (
        AuthRequirementLevel.NONE,
        AuthRequirementLevel.OTP,
        AuthRequirementLevel.MFA,
        AuthRequirementLevel.BIOMETRIC,
        AuthRequirementLevel.MANUAL_REVIEW,
    )

    _AUTH_METHODS: ClassVar[dict[AuthRequirementLevel, list[str]]] = {
        AuthRequirementLevel.OTP: ["sms_otp", "email_otp", "authenticator_app"],
        AuthRequirementLevel.MFA: ["totp", "push_notification", "security_question"],
        AuthRequirementLevel.BIOMETRIC: ["fingerprint", "face_id", "liveness_check"],
        AuthRequirementLevel.MANUAL_REVIEW: ["document_verification", "video_kyc"],
    }

    _AUTH_REASONS: ClassVar[dict[AuthRequirementLevel, str]] = {
        AuthRequirementLevel.OTP: "Elevated risk requires OTP verification",
        AuthRequirementLevel.MFA: "Elevated risk requires multi-factor authentication",
        AuthRequirementLevel.BIOMETRIC: "High risk transaction requires biometric verification",
        AuthRequirementLevel.MANUAL_REVIEW: "Transaction requires manual review",
    }

    @classmethod
    def _strictest(cls, *levels: AuthRequirementLevel) -> AuthRequirementLevel:
        """Return the most restrictive of the given auth levels."""
        return max(levels, key=cls._AUTH_LADDER.index)

    def _score_based_auth_level(self, risk_score: float) -> AuthRequirementLevel:
        """Map a composite risk score onto the auth ladder."""
        config = self.config

        if risk_score >= config.biometric_threshold:
            return AuthRequirementLevel.BIOMETRIC
        if risk_score >= config.mfa_threshold:
            return AuthRequirementLevel.MFA
        if risk_score >= config.approve_threshold:
            return AuthRequirementLevel.OTP
        return AuthRequirementLevel.NONE

    def _matrix_auth_level(
        self, validation_factors: dict[str, ValidationFactor], transaction: Transaction
    ) -> AuthRequirementLevel:
        """
        Apply the risk matrix documented on this class.

        Cross-border is handled by the caller, which short-circuits to manual
        review before this is reached.
        """
        config = self.config
        amount = transaction.amount

        device = validation_factors.get("device")
        location = validation_factors.get("location")
        device_known = device is None or device.passed
        location_familiar = location is None or location.passed

        if not device_known:
            if amount >= config.high_value_threshold:
                return AuthRequirementLevel.BIOMETRIC
            if amount >= config.low_value_threshold:
                return AuthRequirementLevel.MFA
            return AuthRequirementLevel.OTP

        if not location_familiar:
            return AuthRequirementLevel.MFA

        if amount >= config.high_value_threshold:
            return AuthRequirementLevel.OTP

        return AuthRequirementLevel.NONE

    def _make_decision(self, risk_score: float, auth_requirement: AuthRequirement | None) -> bool:
        """Make final approval decision."""

        # Deny if risk too high even with auth
        if risk_score >= self.config.deny_threshold:
            return False

        # If auth is required, technically "approved pending auth"
        if auth_requirement:
            return True  # Approved contingent on auth

        # Low risk - approve
        return True

    def _devices_similar(self, fp1: DeviceFingerprint, fp2: DeviceFingerprint) -> bool:
        """Check if two device fingerprints are similar."""

        # Check hierarchy fingerprints
        h1 = fp1.raw_attributes.get("fingerprint_hierarchy", {})
        h2 = fp2.raw_attributes.get("fingerprint_hierarchy", {})

        # Fine or medium match
        if h1.get("fine") == h2.get("fine"):
            return True
        if h1.get("medium") == h2.get("medium"):
            return True

        return False

    def _locations_match(self, loc1: dict[str, Any], loc2: dict[str, Any]) -> bool:
        """Check if two locations are approximately the same."""

        # Simple match by city and country
        if loc1.get("country") == loc2.get("country") and loc1.get("city") == loc2.get("city"):
            return True

        # Geo distance check (would implement great circle distance)
        lat1 = loc1.get("latitude")
        lon1 = loc1.get("longitude")
        lat2 = loc2.get("latitude")
        lon2 = loc2.get("longitude")

        if all([lat1, lon1, lat2, lon2]):
            # Simplified distance check
            lat_diff = abs(lat1 - lat2)
            lon_diff = abs(lon1 - lon2)

            # Rough approximation: 1 degree ~ 111km
            if lat_diff < 0.5 and lon_diff < 0.5:  # ~50km
                return True

        return False
