"""
Domain Models for Fraud Detection

These models represent the core domain entities used throughout the fraud detection system.
They are plain data classes with no external dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Decision(str, Enum):
    """Fraud detection decision types."""
    APPROVE = "approve"
    DENY = "deny"
    REVIEW = "review"
    ADDITIONAL_AUTH_REQUIRED = "additional_auth_required"


class AuthRequirementLevel(str, Enum):
    """Authentication requirement levels."""
    NONE = "none"
    OTP = "otp"
    MFA = "mfa"
    BIOMETRIC = "biometric"
    MANUAL_REVIEW = "manual_review"


@dataclass
class DeviceAttributes:
    """Device-level attributes for fingerprinting."""
    user_agent: str = ""
    screen_resolution: str = ""
    timezone: str = ""
    language: str = ""
    platform: str = ""
    cpu_cores: Optional[int] = None
    memory_gb: Optional[float] = None
    gpu_renderer: str = ""
    canvas_fingerprint: str = ""
    webgl_fingerprint: str = ""
    audio_fingerprint: str = ""
    fonts_hash: str = ""
    plugins_hash: str = ""


@dataclass
class NetworkAttributes:
    """Network-level attributes."""
    ip_address: str = ""
    ip_version: str = "4"
    geolocation: Optional[Dict[str, Any]] = None
    isp: str = ""
    organization: str = ""
    is_vpn: bool = False
    is_proxy: bool = False
    is_tor: bool = False
    is_datacenter: bool = False
    threat_score: float = 0.0


@dataclass
class BehavioralAttributes:
    """Behavioral attributes for fingerprinting."""
    typing_speed_wpm: Optional[float] = None
    typing_cadence: Optional[List[float]] = None
    mouse_speed: Optional[float] = None
    mouse_acceleration: Optional[float] = None
    scroll_behavior: Optional[Dict[str, Any]] = None
    session_duration_seconds: Optional[float] = None
    pages_visited: int = 0
    interaction_pattern_hash: str = ""


@dataclass
class DeviceFingerprint:
    """
    Complete device fingerprint with confidence scoring.
    
    Attributes:
        id: Unique fingerprint identifier
        confidence: Confidence score (0.0 to 1.0)
        risk_score: Risk score (0.0 to 1.0)
        collision_risk: Probability of collision with another device
        device_attrs: Device-level attributes
        network_attrs: Network-level attributes
        behavioral_attrs: Behavioral attributes
        is_known_device: Whether device is in user's known devices
        first_seen: When this fingerprint was first observed
        last_seen: When this fingerprint was last seen
    """
    id: str
    confidence: float
    risk_score: float
    collision_risk: float = 0.0
    device_attrs: DeviceAttributes = field(default_factory=DeviceAttributes)
    network_attrs: NetworkAttributes = field(default_factory=NetworkAttributes)
    behavioral_attrs: BehavioralAttributes = field(default_factory=BehavioralAttributes)
    is_known_device: bool = False
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    raw_attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def attribute_count(self) -> int:
        """Count of non-empty attributes."""
        count = 0
        for attr_group in [self.device_attrs, self.network_attrs, self.behavioral_attrs]:
            for key, value in attr_group.__dict__.items():
                if value not in (None, "", [], {}):
                    count += 1
        return count


@dataclass
class Transaction:
    """
    Transaction data for fraud detection.
    
    Attributes:
        id: Unique transaction identifier
        user_id: User/customer identifier
        amount: Transaction amount
        currency: Currency code (e.g., "USD", "INR")
        merchant_id: Merchant identifier
        merchant_category: Merchant category code
        timestamp: Transaction timestamp
        location: Geographic location
        payment_method: Payment method type
        device_context: Device context for fingerprinting
        features: Pre-computed ML features (V1-V28, etc.)
    """
    id: str
    user_id: str
    amount: float
    currency: str = "USD"
    merchant_id: str = ""
    merchant_category: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    location: Optional[Dict[str, Any]] = None
    payment_method: str = ""
    device_context: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """User profile for behavioral analysis."""
    user_id: str
    known_devices: List[DeviceFingerprint] = field(default_factory=list)
    frequent_locations: List[Dict[str, Any]] = field(default_factory=list)
    transaction_history: List[Dict[str, Any]] = field(default_factory=list)
    avg_transaction_amount: float = 0.0
    transaction_frequency: float = 0.0  # per day
    risk_level: RiskLevel = RiskLevel.LOW
    account_age_days: int = 0
    is_verified: bool = False


@dataclass
class AuthRequirement:
    """Authentication requirement specification."""
    level: AuthRequirementLevel
    methods: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    reason: str = ""


@dataclass
class PaymentValidationResult:
    """Result of payment source validation."""
    approved: bool
    risk_score: float
    auth_requirement: Optional[AuthRequirement] = None
    validation_factors: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @classmethod
    def denied(cls, reason: str, risk_score: float = 1.0) -> "PaymentValidationResult":
        """Create a denied result."""
        return cls(
            approved=False,
            risk_score=risk_score,
            error=reason,
            auth_requirement=AuthRequirement(
                level=AuthRequirementLevel.MANUAL_REVIEW,
                reason=reason
            )
        )


@dataclass
class MLPrediction:
    """ML model prediction result."""
    model_name: str
    model_version: str
    fraud_probability: float
    confidence: float
    features_used: List[str] = field(default_factory=list)
    inference_time_ms: float = 0.0
    
    @property
    def is_fraud(self) -> bool:
        return self.fraud_probability >= 0.5


@dataclass
class ContributingFactors:
    """Factors contributing to fraud detection decision."""
    device_fingerprint: Optional[DeviceFingerprint] = None
    payment_validation: Optional[PaymentValidationResult] = None
    ml_predictions: Dict[str, MLPrediction] = field(default_factory=dict)
    behavioral_score: float = 0.0
    velocity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "device_risk": self.device_fingerprint.risk_score if self.device_fingerprint else 0.0,
            "payment_risk": self.payment_validation.risk_score if self.payment_validation else 0.0,
            "ml_predictions": {
                name: pred.fraud_probability
                for name, pred in self.ml_predictions.items()
            },
            "behavioral_score": self.behavioral_score,
            "velocity_score": self.velocity_score,
        }


@dataclass
class FraudDetectionResult:
    """
    Complete fraud detection result.
    
    Attributes:
        decision: Final decision (approve/deny/review)
        risk_score: Composite risk score (0.0 to 100.0)
        risk_level: Risk level classification
        fraud_probability: ML-predicted fraud probability
        contributing_factors: Factors that contributed to decision
        explanation: Human-readable explanation
        processing_time_ms: Time taken to process
        model_version: Version of the detection model used
    """
    decision: Decision
    risk_score: float
    risk_level: RiskLevel
    fraud_probability: float
    contributing_factors: ContributingFactors
    explanation: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    model_version: str = "2.9.0"
    request_id: Optional[str] = None
    
    @classmethod
    def from_error(
        cls,
        error: str,
        request_id: Optional[str] = None
    ) -> "FraudDetectionResult":
        """Create result for error case (defaults to manual review)."""
        return cls(
            decision=Decision.REVIEW,
            risk_score=100.0,
            risk_level=RiskLevel.CRITICAL,
            fraud_probability=1.0,
            contributing_factors=ContributingFactors(),
            explanation={"error": error, "summary": "Manual review required due to processing error"},
            request_id=request_id,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "decision": self.decision.value,
            "risk_score": round(self.risk_score, 2),
            "risk_level": self.risk_level.value,
            "fraud_probability": round(self.fraud_probability, 4),
            "contributing_factors": self.contributing_factors.to_dict(),
            "explanation": self.explanation,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "model_version": self.model_version,
            "request_id": self.request_id,
        }
