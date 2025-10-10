"""
Pydantic models for transaction data validation and API responses
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class TransactionType(str, Enum):
    """Transaction type enumeration"""
    PURCHASE = "purchase"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    PAYMENT = "payment"

class TransactionRequest(BaseModel):
    """Request model for fraud detection"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., ge=0.01, description="Transaction amount (must be positive)")
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")
    merchant_category: Optional[str] = Field(None, description="Merchant category code")
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    # Geographic features
    transaction_country: Optional[str] = Field(None, description="Transaction country code")
    transaction_city: Optional[str] = Field(None, description="Transaction city")
    
    # Card features
    card_type: Optional[str] = Field(None, description="Type of card used")
    
    # User behavioral features
    user_id: Optional[str] = Field(None, description="User identifier")
    
    # PCA transformed features (V1-V28) - for compatibility with original dataset
    V1: Optional[float] = Field(None, description="PCA feature V1")
    V2: Optional[float] = Field(None, description="PCA feature V2")
    V3: Optional[float] = Field(None, description="PCA feature V3")
    V4: Optional[float] = Field(None, description="PCA feature V4")
    V5: Optional[float] = Field(None, description="PCA feature V5")
    V6: Optional[float] = Field(None, description="PCA feature V6")
    V7: Optional[float] = Field(None, description="PCA feature V7")
    V8: Optional[float] = Field(None, description="PCA feature V8")
    V9: Optional[float] = Field(None, description="PCA feature V9")
    V10: Optional[float] = Field(None, description="PCA feature V10")
    V11: Optional[float] = Field(None, description="PCA feature V11")
    V12: Optional[float] = Field(None, description="PCA feature V12")
    V13: Optional[float] = Field(None, description="PCA feature V13")
    V14: Optional[float] = Field(None, description="PCA feature V14")
    V15: Optional[float] = Field(None, description="PCA feature V15")
    V16: Optional[float] = Field(None, description="PCA feature V16")
    V17: Optional[float] = Field(None, description="PCA feature V17")
    V18: Optional[float] = Field(None, description="PCA feature V18")
    V19: Optional[float] = Field(None, description="PCA feature V19")
    V20: Optional[float] = Field(None, description="PCA feature V20")
    V21: Optional[float] = Field(None, description="PCA feature V21")
    V22: Optional[float] = Field(None, description="PCA feature V22")
    V23: Optional[float] = Field(None, description="PCA feature V23")
    V24: Optional[float] = Field(None, description="PCA feature V24")
    V25: Optional[float] = Field(None, description="PCA feature V25")
    V26: Optional[float] = Field(None, description="PCA feature V26")
    V27: Optional[float] = Field(None, description="PCA feature V27")
    V28: Optional[float] = Field(None, description="PCA feature V28")
    
    Time: Optional[float] = Field(None, description="Time elapsed since first transaction")
    Amount: Optional[float] = Field(None, description="Transaction amount (duplicate for compatibility)")

    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Transaction amount must be positive')
        if v > 1000000:  # 1M limit
            raise ValueError('Transaction amount exceeds maximum limit')
        return v

    def is_valid_transaction(self) -> bool:
        """Validate if transaction data is complete and valid"""
        return (
            self.transaction_id and
            self.amount > 0 and
            self.transaction_type
        )

class TransactionResponse(BaseModel):
    """Response model for fraud detection results"""
    transaction_id: str
    is_fraud: bool = Field(..., description="Whether transaction is classified as fraud")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of fraud (0-1)")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Risk score (0-100)")
    confidence_level: str = Field(..., description="Confidence level: low, medium, high")
    
    explanation: Dict[str, Any] = Field(..., description="Explainable AI insights")
    model_version: str = Field(..., description="Version of the model used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class BatchTransactionRequest(BaseModel):
    """Request model for batch fraud detection"""
    batch_id: str = Field(..., description="Unique batch identifier")
    transactions: List[TransactionRequest] = Field(..., min_items=1, max_items=1000)
    
    @validator('transactions')
    def validate_transactions(cls, v):
        if len(v) == 0:
            raise ValueError('Batch must contain at least one transaction')
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 transactions')
        return v

class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics"""
    model_name: str
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    auc_score: float = Field(..., ge=0.0, le=1.0)
    
    total_predictions: int = Field(..., ge=0)
    true_positives: int = Field(..., ge=0)
    false_positives: int = Field(..., ge=0)
    true_negatives: int = Field(..., ge=0)
    false_negatives: int = Field(..., ge=0)
    
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class UserFeedback(BaseModel):
    """User feedback on fraud detection"""
    transaction_id: str
    predicted_fraud: bool
    actual_fraud: bool
    user_confidence: int = Field(..., ge=1, le=5, description="User confidence (1-5)")
    feedback_notes: Optional[str] = Field(None, max_length=500)
    timestamp: datetime = Field(default_factory=datetime.utcnow)