"""
Fraud Detection Block - Core Business Logic

PRINCIPLE: Business logic separate from UI and data fetching
WHY: Enables testing, reusability, and maintainability

This module contains the core fraud detection orchestration logic.
All dependencies are injected, making it fully testable.
"""

import logging
from typing import Dict, Any, Optional, Protocol, List
from datetime import datetime
from dataclasses import dataclass

from .models import (
    Transaction,
    DeviceFingerprint,
    FraudDetectionResult,
    ContributingFactors,
    MLPrediction,
    RiskLevel,
    Decision,
    UserProfile,
    PaymentValidationResult,
    AuthRequirement,
    AuthRequirementLevel,
)
from ..utils.error_handling import (
    FraudDetectionError,
    DeviceFingerprintError,
    MLModelError,
    handle_error,
)
from ..utils.logging_utils import get_logger, log_with_context

logger = get_logger(__name__)


# Protocol definitions for dependency injection
class DeviceFingerprintService(Protocol):
    """Interface for device fingerprinting service."""
    
    def generate(self, device_context: Dict[str, Any]) -> DeviceFingerprint:
        """Generate device fingerprint from context."""
        ...


class PaymentValidator(Protocol):
    """Interface for payment source validation."""
    
    def validate(
        self,
        transaction: Transaction,
        device_fingerprint: DeviceFingerprint,
        user_profile: UserProfile
    ) -> PaymentValidationResult:
        """Validate payment source."""
        ...


class MLModelService(Protocol):
    """Interface for ML model predictions."""
    
    def predict(self, features: Dict[str, float]) -> Dict[str, MLPrediction]:
        """Get predictions from all ML models."""
        ...


class TelemetryService(Protocol):
    """Interface for telemetry recording."""
    
    def timer(self, metric_name: str):
        """Context manager for timing operations."""
        ...
    
    def counter(self, metric_name: str, tags: Dict[str, str]) -> None:
        """Increment a counter metric."""
        ...


class UserProfileRepository(Protocol):
    """Interface for user profile data access."""
    
    def get_by_user_id(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID."""
        ...


@dataclass
class FraudDetectionConfig:
    """Configuration for fraud detection block."""
    
    # Risk score weights
    device_weight: float = 0.25
    payment_weight: float = 0.25
    ml_weight: float = 0.40
    behavioral_weight: float = 0.10
    
    # Thresholds
    approve_threshold: float = 30.0
    review_threshold: float = 70.0
    deny_threshold: float = 90.0
    
    # MFA thresholds
    mfa_threshold: float = 50.0
    biometric_threshold: float = 75.0
    
    # Model version
    model_version: str = "2.9.0"


class FraudDetectionBlock:
    """
    Business logic for fraud detection operations.
    
    Follows single responsibility principle.
    All dependencies are injected for testability.
    No direct database access (uses repositories).
    No UI rendering (returns data structures).
    """
    
    def __init__(
        self,
        device_fingerprint_service: DeviceFingerprintService,
        payment_validator: PaymentValidator,
        ml_model_service: MLModelService,
        telemetry: TelemetryService,
        user_profile_repository: UserProfileRepository,
        config: Optional[FraudDetectionConfig] = None,
    ):
        """
        Initialize fraud detection block with injected dependencies.
        
        Args:
            device_fingerprint_service: Service for device fingerprinting
            payment_validator: Service for payment validation
            ml_model_service: Service for ML predictions
            telemetry: Service for metrics and tracing
            user_profile_repository: Repository for user profile data
            config: Optional configuration overrides
        """
        self.device_fingerprint = device_fingerprint_service
        self.payment_validator = payment_validator
        self.ml_models = ml_model_service
        self.telemetry = telemetry
        self.user_profiles = user_profile_repository
        self.config = config or FraudDetectionConfig()
        
        logger.info(
            "FraudDetectionBlock initialized",
            extra={"config": self.config.__dict__}
        )
    
    def detect_fraud(
        self,
        transaction: Transaction,
        request_id: Optional[str] = None
    ) -> FraudDetectionResult:
        """
        Core fraud detection business logic.
        
        Orchestrates all fraud detection signals and combines them
        into a final decision.
        
        Args:
            transaction: Transaction to analyze
            request_id: Optional request ID for correlation
        
        Returns:
            FraudDetectionResult with decision and contributing factors
        
        Raises:
            FraudDetectionError: If critical error occurs during detection
        """
        start_time = datetime.utcnow()
        
        try:
            # Track active transaction
            with self.telemetry.timer("fraud_detection.detect_fraud.duration"):
                
                # Step 1: Get user profile
                user_profile = self._get_user_profile(transaction.user_id)
                
                # Step 2: Generate device fingerprint
                device_fp = self._generate_device_fingerprint(
                    transaction.device_context,
                    request_id
                )
                
                # Step 3: Validate payment source
                payment_validation = self._validate_payment_source(
                    transaction,
                    device_fp,
                    user_profile,
                    request_id
                )
                
                # Step 4: Get ML predictions
                ml_predictions = self._get_ml_predictions(
                    transaction.features,
                    request_id
                )
                
                # Step 5: Calculate behavioral score
                behavioral_score = self._calculate_behavioral_score(
                    transaction,
                    user_profile
                )
                
                # Step 6: Combine all signals into final risk score
                contributing_factors = ContributingFactors(
                    device_fingerprint=device_fp,
                    payment_validation=payment_validation,
                    ml_predictions=ml_predictions,
                    behavioral_score=behavioral_score,
                    velocity_score=self._calculate_velocity_score(transaction, user_profile)
                )
                
                final_risk_score = self._calculate_composite_risk_score(
                    contributing_factors
                )
                
                # Step 7: Determine decision based on risk score and rules
                decision, risk_level = self._apply_fraud_rules(
                    final_risk_score,
                    transaction,
                    contributing_factors
                )
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Create result
                result = FraudDetectionResult(
                    decision=decision,
                    risk_score=final_risk_score,
                    risk_level=risk_level,
                    fraud_probability=self._get_ensemble_fraud_probability(ml_predictions),
                    contributing_factors=contributing_factors,
                    explanation=self._generate_explanation(
                        final_risk_score,
                        contributing_factors,
                        decision
                    ),
                    processing_time_ms=processing_time,
                    model_version=self.config.model_version,
                    request_id=request_id,
                )
                
                # Record telemetry
                self.telemetry.counter(
                    "fraud_detection.decisions",
                    tags={
                        "decision": decision.value,
                        "risk_level": risk_level.value
                    }
                )
                
                logger.info(
                    "Fraud detection completed",
                    extra={
                        "transaction_id": transaction.id,
                        "decision": decision.value,
                        "risk_score": final_risk_score,
                        "processing_time_ms": processing_time,
                        "request_id": request_id,
                    }
                )
                
                return result
        
        except DeviceFingerprintError as e:
            # Device fingerprint failed - use fallback
            logger.warning(
                "Device fingerprint failed, using fallback",
                extra={"error": str(e), "request_id": request_id}
            )
            return self._fallback_fraud_check(transaction, request_id, str(e))
        
        except MLModelError as e:
            # ML model failed - use rule-based fallback
            logger.warning(
                "ML model failed, using rule-based fallback",
                extra={"error": str(e), "request_id": request_id}
            )
            return self._rule_based_fraud_check(transaction, request_id, str(e))
        
        except Exception as e:
            # Unexpected error - log and return safe default
            handle_error(
                e,
                context={
                    "transaction_id": transaction.id,
                    "request_id": request_id,
                    "unexpected_error": True
                },
                reraise=False
            )
            
            # Default to manual review for safety
            return FraudDetectionResult.from_error(
                error=str(e),
                request_id=request_id
            )
    
    def _get_user_profile(self, user_id: str) -> UserProfile:
        """Get user profile or create default for new users."""
        profile = self.user_profiles.get_by_user_id(user_id)
        
        if profile is None:
            logger.debug(f"No profile found for user {user_id}, creating default")
            return UserProfile(user_id=user_id)
        
        return profile
    
    def _generate_device_fingerprint(
        self,
        device_context: Dict[str, Any],
        request_id: Optional[str]
    ) -> DeviceFingerprint:
        """Generate device fingerprint with error handling."""
        try:
            return self.device_fingerprint.generate(device_context)
        except Exception as e:
            raise DeviceFingerprintError(
                "Failed to generate device fingerprint",
                context={"request_id": request_id}
            ) from e
    
    def _validate_payment_source(
        self,
        transaction: Transaction,
        device_fp: DeviceFingerprint,
        user_profile: UserProfile,
        request_id: Optional[str]
    ) -> PaymentValidationResult:
        """Validate payment source with error handling."""
        try:
            return self.payment_validator.validate(
                transaction,
                device_fp,
                user_profile
            )
        except Exception as e:
            logger.warning(
                "Payment validation failed",
                extra={"error": str(e), "request_id": request_id}
            )
            # Return high-risk result on validation failure
            return PaymentValidationResult.denied(
                reason=f"Payment validation error: {str(e)}"
            )
    
    def _get_ml_predictions(
        self,
        features: Dict[str, float],
        request_id: Optional[str]
    ) -> Dict[str, MLPrediction]:
        """Get ML predictions with error handling."""
        try:
            return self.ml_models.predict(features)
        except Exception as e:
            raise MLModelError(
                "Failed to get ML predictions",
                context={"request_id": request_id}
            ) from e
    
    def _calculate_behavioral_score(
        self,
        transaction: Transaction,
        user_profile: UserProfile
    ) -> float:
        """
        Calculate behavioral anomaly score.
        
        Compares transaction against user's historical patterns.
        Returns 0.0 (normal) to 1.0 (highly anomalous).
        """
        if not user_profile.transaction_history:
            # New user - moderate risk
            return 0.5
        
        score = 0.0
        factors = 0
        
        # Amount deviation
        if user_profile.avg_transaction_amount > 0:
            amount_ratio = transaction.amount / user_profile.avg_transaction_amount
            if amount_ratio > 5.0:
                score += 0.8
            elif amount_ratio > 3.0:
                score += 0.5
            elif amount_ratio > 2.0:
                score += 0.3
            factors += 1
        
        # Location check (simplified)
        if transaction.location and user_profile.frequent_locations:
            is_familiar_location = any(
                self._locations_match(transaction.location, loc)
                for loc in user_profile.frequent_locations
            )
            if not is_familiar_location:
                score += 0.4
            factors += 1
        
        # Time of day check (simplified)
        hour = transaction.timestamp.hour
        if 2 <= hour <= 5:  # Late night transactions
            score += 0.2
            factors += 1
        
        return score / max(factors, 1)
    
    def _calculate_velocity_score(
        self,
        transaction: Transaction,
        user_profile: UserProfile
    ) -> float:
        """
        Calculate velocity-based risk score.
        
        Checks for unusual transaction frequency.
        """
        # Simplified implementation - would check recent transaction count
        if user_profile.transaction_frequency > 10:  # More than 10 per day
            return 0.7
        elif user_profile.transaction_frequency > 5:
            return 0.4
        return 0.1
    
    def _locations_match(
        self,
        loc1: Dict[str, Any],
        loc2: Dict[str, Any]
    ) -> bool:
        """Check if two locations are approximately the same."""
        # Simplified - check country and city
        return (
            loc1.get("country") == loc2.get("country") and
            loc1.get("city") == loc2.get("city")
        )
    
    def _calculate_composite_risk_score(
        self,
        factors: ContributingFactors
    ) -> float:
        """
        Calculate weighted composite risk score (0-100).
        
        Combines all risk signals with configured weights.
        """
        config = self.config
        
        # Device risk
        device_risk = 0.0
        if factors.device_fingerprint:
            device_risk = factors.device_fingerprint.risk_score * 100
        
        # Payment risk
        payment_risk = 0.0
        if factors.payment_validation:
            payment_risk = factors.payment_validation.risk_score * 100
        
        # ML risk (average of all models)
        ml_risk = 0.0
        if factors.ml_predictions:
            ml_probs = [p.fraud_probability for p in factors.ml_predictions.values()]
            ml_risk = (sum(ml_probs) / len(ml_probs)) * 100
        
        # Behavioral risk
        behavioral_risk = factors.behavioral_score * 100
        
        # Weighted combination
        composite = (
            device_risk * config.device_weight +
            payment_risk * config.payment_weight +
            ml_risk * config.ml_weight +
            behavioral_risk * config.behavioral_weight
        )
        
        # Model agreement boost/penalty
        if factors.ml_predictions:
            std_dev = self._calculate_prediction_std(factors.ml_predictions)
            if std_dev < 0.1:  # High agreement
                composite *= 1.1  # Boost confidence
            elif std_dev > 0.3:  # Low agreement
                composite *= 0.9  # Reduce confidence
        
        return min(max(composite, 0.0), 100.0)
    
    def _calculate_prediction_std(
        self,
        predictions: Dict[str, MLPrediction]
    ) -> float:
        """Calculate standard deviation of model predictions."""
        probs = [p.fraud_probability for p in predictions.values()]
        if len(probs) < 2:
            return 0.0
        
        mean = sum(probs) / len(probs)
        variance = sum((p - mean) ** 2 for p in probs) / len(probs)
        return variance ** 0.5
    
    def _get_ensemble_fraud_probability(
        self,
        predictions: Dict[str, MLPrediction]
    ) -> float:
        """Calculate ensemble fraud probability."""
        if not predictions:
            return 0.5
        
        probs = [p.fraud_probability for p in predictions.values()]
        return sum(probs) / len(probs)
    
    def _apply_fraud_rules(
        self,
        risk_score: float,
        transaction: Transaction,
        factors: ContributingFactors
    ) -> tuple[Decision, RiskLevel]:
        """
        Apply business rules to determine final decision.
        
        Returns (Decision, RiskLevel) tuple.
        """
        config = self.config
        
        # Determine risk level
        if risk_score >= config.deny_threshold:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= config.review_threshold:
            risk_level = RiskLevel.HIGH
        elif risk_score >= config.approve_threshold:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Apply rules based on risk level
        if risk_level == RiskLevel.CRITICAL:
            return Decision.DENY, risk_level
        
        if risk_level == RiskLevel.HIGH:
            # Check if additional auth might clear it
            if factors.device_fingerprint and factors.device_fingerprint.is_known_device:
                return Decision.ADDITIONAL_AUTH_REQUIRED, risk_level
            return Decision.REVIEW, risk_level
        
        if risk_level == RiskLevel.MEDIUM:
            # Medium risk - may require additional auth
            if risk_score >= config.mfa_threshold:
                return Decision.ADDITIONAL_AUTH_REQUIRED, risk_level
            return Decision.APPROVE, risk_level
        
        # Low risk - approve
        return Decision.APPROVE, risk_level
    
    def _generate_explanation(
        self,
        risk_score: float,
        factors: ContributingFactors,
        decision: Decision
    ) -> Dict[str, Any]:
        """Generate human-readable explanation for the decision."""
        explanation = {
            "summary": self._get_decision_summary(risk_score, decision),
            "key_factors": [],
            "risk_indicators": [],
            "recommendations": [],
        }
        
        # Add device factors
        if factors.device_fingerprint:
            fp = factors.device_fingerprint
            if not fp.is_known_device:
                explanation["risk_indicators"].append("Transaction from unrecognized device")
            if fp.network_attrs.is_vpn:
                explanation["risk_indicators"].append("VPN or proxy detected")
            if fp.risk_score > 0.5:
                explanation["key_factors"].append({
                    "factor": "Device Risk",
                    "impact": "high",
                    "description": f"Device risk score: {fp.risk_score:.2f}"
                })
        
        # Add payment validation factors
        if factors.payment_validation:
            pv = factors.payment_validation
            if pv.error:
                explanation["risk_indicators"].append(f"Payment validation issue: {pv.error}")
        
        # Add behavioral factors
        if factors.behavioral_score > 0.5:
            explanation["risk_indicators"].append("Unusual transaction behavior detected")
            explanation["key_factors"].append({
                "factor": "Behavioral Anomaly",
                "impact": "medium",
                "description": "Transaction deviates from typical user patterns"
            })
        
        # Add recommendations
        if decision == Decision.ADDITIONAL_AUTH_REQUIRED:
            explanation["recommendations"].append("Request additional authentication")
        elif decision == Decision.REVIEW:
            explanation["recommendations"].append("Manual review recommended")
        
        return explanation
    
    def _get_decision_summary(
        self,
        risk_score: float,
        decision: Decision
    ) -> str:
        """Get human-readable summary for decision."""
        summaries = {
            Decision.APPROVE: "Transaction appears legitimate with low fraud risk",
            Decision.ADDITIONAL_AUTH_REQUIRED: "Additional authentication required due to elevated risk",
            Decision.REVIEW: "Transaction flagged for manual review due to suspicious indicators",
            Decision.DENY: "Transaction blocked due to high fraud probability",
        }
        return summaries.get(decision, "Unable to determine transaction status")
    
    def _fallback_fraud_check(
        self,
        transaction: Transaction,
        request_id: Optional[str],
        error: str
    ) -> FraudDetectionResult:
        """Fallback fraud check when device fingerprint fails."""
        # Use ML predictions and behavioral analysis only
        contributing_factors = ContributingFactors(
            behavioral_score=0.5  # Unknown without full context
        )
        
        return FraudDetectionResult(
            decision=Decision.ADDITIONAL_AUTH_REQUIRED,
            risk_score=60.0,
            risk_level=RiskLevel.MEDIUM,
            fraud_probability=0.5,
            contributing_factors=contributing_factors,
            explanation={
                "summary": "Limited validation available, additional authentication required",
                "fallback_reason": error,
            },
            model_version=self.config.model_version,
            request_id=request_id,
        )
    
    def _rule_based_fraud_check(
        self,
        transaction: Transaction,
        request_id: Optional[str],
        error: str
    ) -> FraudDetectionResult:
        """Rule-based fallback when ML models fail."""
        # Simple rules
        risk_score = 30.0
        
        if transaction.amount > 10000:
            risk_score += 30.0
        
        hour = transaction.timestamp.hour
        if 2 <= hour <= 5:
            risk_score += 20.0
        
        risk_level = RiskLevel.MEDIUM if risk_score > 50 else RiskLevel.LOW
        decision = Decision.ADDITIONAL_AUTH_REQUIRED if risk_score > 50 else Decision.APPROVE
        
        return FraudDetectionResult(
            decision=decision,
            risk_score=risk_score,
            risk_level=risk_level,
            fraud_probability=risk_score / 100,
            contributing_factors=ContributingFactors(),
            explanation={
                "summary": "Rule-based analysis due to model unavailability",
                "fallback_reason": error,
            },
            model_version=self.config.model_version,
            request_id=request_id,
        )
