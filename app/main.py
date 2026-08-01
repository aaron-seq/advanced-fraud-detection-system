"""
Advanced Fraud Detection System - FastAPI Backend.

Thin HTTP layer. All detection logic lives in src/block; this module only
translates HTTP to domain objects and back.
"""

import logging
import math
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.cache import CacheManager, close_redis_client
from app.core.config import get_application_settings
from app.core.database import dispose_engine, get_engine
from app.core.security import verify_token
from app.models.transaction_models import (
    BatchTransactionRequest,
    BatchTransactionResponse,
    TransactionRequest,
    TransactionResponse,
)
from app.services.fraud_detection_service import FraudDetectionService
from app.utils.monitoring import RequestMonitoringMiddleware
from app.utils.rate_limiting import RateLimitingMiddleware

settings = get_application_settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=(
        [logging.FileHandler(settings.log_file), logging.StreamHandler()]
        if settings.enable_file_logging
        else [logging.StreamHandler()]
    ),
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def application_lifecycle(app: FastAPI):
    """Start up and tear down shared resources."""
    logger.info("Starting Advanced Fraud Detection System")

    app.state.started_at = datetime.now(UTC)

    app.state.fraud_service = FraudDetectionService()
    await app.state.fraud_service.initialize(settings.model_path)

    app.state.engine = get_engine()
    app.state.cache = CacheManager()

    yield

    logger.info("Shutting down fraud detection system")
    await dispose_engine()
    await close_redis_client()


def create_fraud_detection_application() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Advanced Fraud Detection API",
        description="Credit card fraud detection with device fingerprinting, "
        "payment source validation and behavioural analysis.",
        version=settings.app_version,
        docs_url="/api/docs" if settings.environment != "production" else None,
        redoc_url="/api/redoc" if settings.environment != "production" else None,
        lifespan=application_lifecycle,
    )

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.add_middleware(RequestMonitoringMiddleware)
    app.add_middleware(RateLimitingMiddleware, requests_per_minute=settings.rate_limit_requests)

    return app


app = create_fraud_detection_application()


def _json_safe(value):
    """
    Coerce values that json.dumps cannot represent into strings.

    FastAPI's 422 body echoes the rejected input back. When that input is NaN
    or infinity - exactly what the finite-number validators exist to reject -
    rendering the error response itself raises, turning a clean 422 into a 500
    and hiding the validation message from the client.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return validation failures as a JSON-encodable 422."""
    return JSONResponse(
        # Renamed in Starlette 1.3; the old alias is deprecated.
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": _json_safe(exc.errors())},
    )


def request_context(request: Request) -> dict:
    """
    Build the device-fingerprinting context from the live request.

    Header and peer data are read here rather than accepted from the request
    body, so a client cannot dictate the signals used to fingerprint it.
    """
    return {
        "headers": dict(request.headers),
        "client": (request.client.host, request.client.port) if request.client else (),
    }


@app.get("/", tags=["Health"])
async def system_health_check():
    """Liveness probe."""
    return {
        "service": settings.app_name,
        "status": "operational",
        "version": settings.app_version,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/api/v1/health", tags=["Health"])
async def detailed_health_check():
    """
    Readiness probe covering database, cache and model availability.

    Dependency failures are reported as "degraded" with a 200 rather than
    raising, so an operator can see which component is down. Previously any
    single failure raised a bare 503 that hid which one it was.
    """
    components = {}

    try:
        async with app.state.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        components["database"] = "healthy"
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)
        components["database"] = "unhealthy"

    components["cache"] = "healthy" if await app.state.cache.ping() else "unhealthy"

    model_status = app.state.fraud_service.model_health()
    components["ml_models"] = "healthy" if model_status["loaded"] else "unhealthy"

    healthy = all(state == "healthy" for state in components.values())

    return {
        "status": "healthy" if healthy else "degraded",
        "timestamp": datetime.now(UTC).isoformat(),
        "components": components,
        "models": model_status,
        "uptime_seconds": (datetime.now(UTC) - app.state.started_at).total_seconds(),
    }


@app.post(
    "/api/v1/detect-fraud",
    response_model=TransactionResponse,
    tags=["Fraud Detection"],
)
async def detect_transaction_fraud(
    transaction: TransactionRequest,
    request: Request,
    user: dict = Depends(verify_token),
):
    """Analyse a single transaction for fraud."""
    request_id = str(uuid.uuid4())

    try:
        result = await app.state.fraud_service.analyze_transaction(
            transaction.to_domain(
                user_id=transaction.user_id or user["user_id"],
                device_context=request_context(request),
            ),
            request_id=request_id,
        )
    except Exception as exc:
        # Logged with the traceback and surfaced as a 500. Never downgraded to
        # a default "approve": failing open is how a broken detector silently
        # lets fraud through.
        logger.exception("Fraud detection failed for request %s", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fraud detection failed",
        ) from exc

    return TransactionResponse.from_result(transaction.transaction_id, result)


@app.post(
    "/api/v1/detect-fraud/batch",
    response_model=BatchTransactionResponse,
    tags=["Fraud Detection"],
)
async def detect_batch_fraud(
    batch_request: BatchTransactionRequest,
    request: Request,
    user: dict = Depends(verify_token),
):
    """
    Analyse a batch of transactions.

    The batch size ceiling is enforced by the request model, which returns a
    422 before any work is scheduled.
    """
    request_id = str(uuid.uuid4())
    context = request_context(request)

    try:
        results = await app.state.fraud_service.analyze_batch(
            [
                item.to_domain(
                    user_id=item.user_id or user["user_id"],
                    device_context=context,
                )
                for item in batch_request.transactions
            ],
            request_id=request_id,
        )
    except Exception as exc:
        logger.exception("Batch fraud detection failed for request %s", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch fraud detection failed",
        ) from exc

    responses = [
        TransactionResponse.from_result(item.transaction_id, result)
        # strict=True: a length mismatch means results were silently dropped,
        # which would pair a response with the wrong transaction.
        for item, result in zip(batch_request.transactions, results, strict=True)
    ]
    flagged = [r for r in responses if r.is_fraud]

    return BatchTransactionResponse(
        batch_id=batch_request.batch_id,
        total_transactions=len(responses),
        fraud_detected=len(flagged),
        fraud_percentage=round(100 * len(flagged) / len(responses), 2),
        average_risk_score=round(sum(r.risk_score for r in responses) / len(responses), 3),
        results=responses,
    )


@app.get("/api/v1/analytics/dashboard-data", tags=["Analytics"])
async def get_dashboard_analytics(
    days_back: int = Query(7, ge=1, le=365),
    user: dict = Depends(verify_token),
):
    """
    Analytics over decisions this process has recorded.

    Authenticated: it exposes transaction volumes and amounts. Counters are
    in-process and reset on restart, which the payload states in its `source`
    field rather than leaving the caller to assume the numbers are durable.
    """
    return app.state.fraud_service.analytics_summary(days_back=days_back)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=settings.port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )
