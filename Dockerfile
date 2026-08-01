# Advanced Fraud Detection System - production image.
# Multi-stage: build wheels once, ship only the runtime layer.

FROM python:3.13-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.13-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ENVIRONMENT=production

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser \
    && useradd -r -g appuser appuser

WORKDIR /app

COPY --from=builder /install /usr/local

# src/ holds the detection logic that app/ imports. The previous Dockerfile
# omitted it and instead copied models/ and scripts/, neither of which exists
# in this repository - so the build failed, and would have produced an image
# that could not import app.main even if it had succeeded.
COPY app/ ./app/
COPY src/ ./src/

RUN mkdir -p /app/logs /app/models /app/data && chown -R appuser:appuser /app

USER appuser

# Probes the readiness endpoint, which reports database, cache and model state.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

# Binding 0.0.0.0 is required for the port to be reachable from outside the
# container; the container boundary, not the bind address, is the control here.
# SECRET_KEY must be supplied at runtime - the app refuses to start without it
# when ENVIRONMENT=production.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
