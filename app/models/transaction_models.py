"""
Pydantic models for transaction data validation and API responses.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from src.block.models import Transaction

MAX_TRANSACTION_AMOUNT = 1_000_000.0
MAX_BATCH_SIZE = 1000

# Rejects NaN and +/-inf at parse time.
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

# PCA components from the public credit-card fraud dataset.
PCA_FEATURE_NAMES = tuple(f"V{index}" for index in range(1, 29))


class TransactionType(str, Enum):
    """Transaction type enumeration."""

    PURCHASE = "purchase"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    PAYMENT = "payment"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TransactionRequest(BaseModel):
    """A single transaction submitted for fraud analysis."""

    transaction_id: str = Field(..., min_length=1, max_length=128)
    amount: float = Field(
        ...,
        gt=0,
        le=MAX_TRANSACTION_AMOUNT,
        description="Transaction amount; must be positive",
    )
    currency: str = Field("USD", min_length=3, max_length=3)
    transaction_type: TransactionType

    merchant_id: Optional[str] = Field(None, max_length=128)
    merchant_category: Optional[str] = Field(None, max_length=64)
    timestamp: datetime = Field(default_factory=_utcnow)

    transaction_country: Optional[str] = Field(None, max_length=64)
    transaction_city: Optional[str] = Field(None, max_length=128)

    card_type: Optional[str] = Field(None, max_length=32)
    user_id: Optional[str] = Field(None, max_length=128)

    # PCA-transformed features, kept flat for compatibility with the public
    # dataset's column layout.
    #
    # allow_inf_nan=False is load-bearing, not cosmetic: NaN propagates
    # silently through arithmetic and makes a risk score NaN, which compares
    # False against every threshold - so a transaction carrying one would slip
    # past every rule meant to catch it.
    features: Dict[str, FiniteFloat] = Field(
        default_factory=dict,
        description="Model features, e.g. {'V1': -1.36, 'V2': 0.07, 'Amount': 149.62}",
    )

    @field_validator("currency")
    @classmethod
    def _normalise_currency(cls, value: str) -> str:
        return value.upper()

    def to_domain(
        self,
        user_id: str,
        device_context: Optional[Dict[str, Any]] = None,
    ) -> Transaction:
        """Convert to the domain Transaction consumed by FraudDetectionBlock."""
        location = None
        if self.transaction_country or self.transaction_city:
            location = {
                "country": self.transaction_country,
                "country_code": self.transaction_country,
                "city": self.transaction_city,
            }

        return Transaction(
            id=self.transaction_id,
            user_id=user_id,
            amount=self.amount,
            currency=self.currency,
            merchant_id=self.merchant_id or "",
            merchant_category=self.merchant_category or "",
            timestamp=self.timestamp,
            location=location,
            payment_method=self.card_type or "",
            device_context=device_context or {},
            features=dict(self.features),
        )


class TransactionResponse(BaseModel):
    """Fraud analysis result for a single transaction."""

    transaction_id: str
    decision: str
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_level: str
    explanation: Dict[str, Any] = Field(default_factory=dict)
    contributing_factors: Dict[str, Any] = Field(default_factory=dict)
    model_version: str
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=_utcnow)

    @classmethod
    def from_result(
        cls, transaction_id: str, result: Dict[str, Any]
    ) -> "TransactionResponse":
        """Build a response from FraudDetectionResult.to_dict()."""
        return cls(
            transaction_id=transaction_id,
            decision=result["decision"],
            is_fraud=result["decision"] in ("deny", "review"),
            fraud_probability=result["fraud_probability"],
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            explanation=result.get("explanation", {}),
            contributing_factors=result.get("contributing_factors", {}),
            model_version=result["model_version"],
            processing_time_ms=result["processing_time_ms"],
        )


class BatchTransactionRequest(BaseModel):
    """A batch of transactions submitted for fraud analysis."""

    batch_id: str = Field(..., min_length=1, max_length=128)
    transactions: List[TransactionRequest] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE
    )


class BatchTransactionResponse(BaseModel):
    """Aggregated results for a batch."""

    batch_id: str
    total_transactions: int
    fraud_detected: int
    fraud_percentage: float
    average_risk_score: float
    results: List[TransactionResponse]
    timestamp: datetime = Field(default_factory=_utcnow)
