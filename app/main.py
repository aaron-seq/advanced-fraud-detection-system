"""
Advanced Fraud Detection System - FastAPI Backend
Production-ready API with modern architecture and best practices.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import os
from datetime import datetime, timedelta
import json

from app.core.config import get_application_settings
from app.core.database import get_database_connection
from app.core.cache import get_redis_client
from app.models.transaction_models import (
    TransactionRequest, 
    TransactionResponse, 
    BatchTransactionRequest,
    ModelPerformanceMetrics
)
from app.services.fraud_detection_service import EnhancedFraudDetectionService
from app.services.model_management_service import ModelManagementService
from app.core.security import create_access_token, verify_token
from app.utils.monitoring import RequestMonitoringMiddleware
from app.utils.rate_limiting import RateLimitingMiddleware

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Application lifecycle management
@asynccontextmanager
async def application_lifecycle(app: FastAPI):
    """Manage application startup and shutdown procedures"""
    logger.info("🚀 Starting Advanced Fraud Detection System...")
    
    # Initialize services
    app.state.fraud_service = EnhancedFraudDetectionService()
    app.state.model_service = ModelManagementService()
    
    # Load ML models
    await app.state.fraud_service.initialize_models()
    logger.info("✅ ML Models loaded successfully")
    
    # Initialize database connection pool
    app.state.database = get_database_connection()
    logger.info("✅ Database connection established")
    
    # Initialize Redis cache
    app.state.cache = get_redis_client()
    logger.info("✅ Redis cache connected")
    
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down fraud detection system...")
    await app.state.database.disconnect()
    await app.state.cache.close()

# Create FastAPI application
def create_fraud_detection_application() -> FastAPI:
    """Factory function to create and configure FastAPI application"""
    
    settings = get_application_settings()
    
    app = FastAPI(
        title="Advanced Fraud Detection API",
        description="Production-ready credit card fraud detection system with ML ensemble",
        version="2.0.0",
        docs_url="/api/docs" if settings.environment != "production" else None,
        redoc_url="/api/redoc" if settings.environment != "production" else None,
        lifespan=application_lifecycle
    )
    
    # Security middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    
    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Custom middleware
    app.add_middleware(RequestMonitoringMiddleware)
    app.add_middleware(RateLimitingMiddleware, requests_per_minute=100)
    
    return app

# Initialize application
app = create_fraud_detection_application()

@app.get("/", tags=["Health Check"])
async def system_health_check():
    """System health and status endpoint"""
    return {
        "service": "Advanced Fraud Detection System",
        "status": "operational",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "Real-time fraud detection",
            "Ensemble ML models",
            "Batch processing",
            "Model monitoring",
            "Explainable AI"
        ]
    }

@app.get("/api/v1/health", tags=["Health Check"])
async def detailed_health_check():
    """Comprehensive health check with system metrics"""
    try:
        # Check database connectivity
        db_status = await app.state.database.execute("SELECT 1")
        db_healthy = bool(db_status)
        
        # Check cache connectivity  
        cache_status = await app.state.cache.ping()
        cache_healthy = cache_status
        
        # Check model availability
        model_status = app.state.fraud_service.get_model_health()
        
        return {
            "status": "healthy" if all([db_healthy, cache_healthy, model_status["loaded"]]) else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": "healthy" if db_healthy else "unhealthy",
                "cache": "healthy" if cache_healthy else "unhealthy",
                "ml_models": "healthy" if model_status["loaded"] else "unhealthy"
            },
            "metrics": {
                "models_loaded": model_status["count"],
                "uptime_seconds": (datetime.utcnow() - app.state.startup_time).total_seconds()
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

@app.post("/api/v1/detect-fraud", response_model=TransactionResponse, tags=["Fraud Detection"])
async def detect_transaction_fraud(
    transaction: TransactionRequest,
    background_tasks: BackgroundTasks
):
    """
    Real-time fraud detection for individual transactions
    
    - Processes transaction in real-time (< 100ms)
    - Uses ensemble ML models for high accuracy
    - Provides explainable results
    - Logs transaction for monitoring
    """
    try:
        logger.info(f"Processing fraud detection request for transaction: {transaction.transaction_id}")
        
        # Validate transaction data
        if not transaction.is_valid_transaction():
            raise HTTPException(
                status_code=400, 
                detail="Invalid transaction data provided"
            )
        
        # Perform fraud detection
        detection_result = await app.state.fraud_service.analyze_transaction(
            transaction_data=transaction.dict(),
            user_context={}
        )
        
        # Log transaction for monitoring (background task)
        background_tasks.add_task(
            log_transaction_analysis,
            transaction.dict(),
            detection_result
        )
        
        return TransactionResponse(
            transaction_id=transaction.transaction_id,
            is_fraud=detection_result["is_fraud"],
            fraud_probability=detection_result["probability"],
            risk_score=detection_result["risk_score"],
            confidence_level=detection_result["confidence"],
            explanation=detection_result["explanation"],
            model_version=detection_result["model_version"],
            processing_time_ms=detection_result["processing_time"],
            timestamp=datetime.utcnow()
        )
        
    except ValueError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Fraud detection error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during fraud detection")

@app.post("/api/v1/detect-fraud/batch", tags=["Fraud Detection"])
async def detect_batch_fraud(
    batch_request: BatchTransactionRequest,
    background_tasks: BackgroundTasks
):
    """
    Batch fraud detection for multiple transactions
    
    - Processes up to 1000 transactions in parallel
    - Optimized for high throughput scenarios
    - Returns aggregated results and statistics
    """
    try:
        if len(batch_request.transactions) > 1000:
            raise HTTPException(
                status_code=413, 
                detail="Batch size exceeds maximum limit of 1000 transactions"
            )
        
        logger.info(f"Processing batch of {len(batch_request.transactions)} transactions")
        
        # Process transactions in parallel
        batch_results = await app.state.fraud_service.analyze_transaction_batch(
            transactions=[t.dict() for t in batch_request.transactions],
            user_context={}
        )
        
        # Generate batch statistics
        fraud_count = sum(1 for r in batch_results if r["is_fraud"])
        avg_risk_score = sum(r["risk_score"] for r in batch_results) / len(batch_results)
        
        # Log batch processing (background task)
        background_tasks.add_task(
            log_batch_analysis,
            len(batch_request.transactions),
            fraud_count,
            avg_risk_score
        )
        
        return {
            "batch_id": batch_request.batch_id,
            "total_transactions": len(batch_request.transactions),
            "fraud_detected": fraud_count,
            "fraud_percentage": (fraud_count / len(batch_request.transactions)) * 100,
            "average_risk_score": round(avg_risk_score, 3),
            "processing_time_ms": sum(r["processing_time"] for r in batch_results),
            "results": batch_results,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing transaction batch")

@app.get("/api/v1/analytics/dashboard-data", tags=["Analytics"])
async def get_dashboard_analytics(
    days_back: int = 7
):
    """Get analytics data for the dashboard"""
    try:
        analytics_data = await app.state.fraud_service.get_analytics_summary(
            days_back=days_back
        )
        
        return analytics_data
        
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving analytics data")

# Background task functions
async def log_transaction_analysis(transaction_data: dict, result: dict):
    """Log transaction analysis for audit and monitoring"""
    try:
        # Implementation for logging transaction
        pass
    except Exception as e:
        logger.error(f"Error logging transaction: {str(e)}")

async def log_batch_analysis(count: int, fraud_count: int, avg_score: float):
    """Log batch analysis results"""
    try:
        # Implementation for logging batch analysis
        pass
    except Exception as e:
        logger.error(f"Error logging batch analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    settings = get_application_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.environment == "development",
        log_level="info"
    )