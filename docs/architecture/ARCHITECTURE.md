# Advanced Fraud Detection System - Architecture

## Overview

The Advanced Fraud Detection System is a real-time fraud detection service combining device fingerprinting, payment source validation, behavioural analytics and an optional ML ensemble behind a FastAPI HTTP API.

## Architecture Diagram

````mermaid
flowchart TB
    subgraph Client["Client Layer"]
        WebApp[Web Application]
        MobileApp[Mobile App]
        API[API Clients]
    end

    subgraph Gateway["API Gateway"]
        FastAPI[FastAPI Application]
        RateLimiter[Rate Limiter]
        CORS[CORS Middleware]
    end

    subgraph Block["Business Logic Layer"]
        FDB[Fraud Detection Block]
        Models[Domain Models]
    end

    subgraph FraudDetection["Fraud Detection Services"]
        DFE[Device Fingerprint Engine]
        PSV[Payment Source Validator]
        SPA[Spending Pattern Analyzer]
    end

    subgraph ML["ML Layer"]
        Ensemble[Ensemble Predictor]
        XGBoost[XGBoost]
        LightGBM[LightGBM]
        CatBoost[CatBoost]
    end

    subgraph Telemetry["Observability"]
        Prometheus[Prometheus Metrics]
        Logging[Structured Logging]
        Analytics[Business Analytics]
    end

    subgraph Storage["Data Layer"]
        PostgreSQL[(PostgreSQL)]
        Redis[(Redis Cache)]
        Kafka[Kafka Streams]
    end

    Client --> Gateway
    Gateway --> Block
    Block --> FraudDetection
    Block --> ML
    FraudDetection --> Storage
    ML --> Storage
    Gateway --> Telemetry
    Block --> Telemetry
</diagram>

## Component Responsibilities

### API Layer (`app/`)

| Component | Responsibility |
|-----------|----------------|
| `main.py` | FastAPI application setup, routing, middleware |
| `api/` | REST endpoint handlers |
| `models/` | Pydantic request/response schemas |
| `core/config.py` | Configuration management |

### Business Logic Layer (`src/block/`)

| Component | Responsibility |
|-----------|----------------|
| `fraud_detection_block.py` | Core fraud detection orchestration |
| `models.py` | Domain models (Transaction, DeviceFingerprint, etc.) |

**Design Principle**: Pure business logic with dependency injection. No UI or data access.

### Fraud Detection Services (`src/fraud_detection/`)

| Component | Responsibility |
|-----------|----------------|
| `device_fingerprinting/` | 2000+ attribute device fingerprinting |
| `payment_validation/` | RBI April 2026 compliant validation |
| `behavioral/` | Spending pattern anomaly detection |

### Utils Layer (`src/utils/`)

| Component | Responsibility |
|-----------|----------------|
| `error_handling.py` | Centralized error handling and exceptions |
| `logging_utils.py` | Structured logging with correlation IDs |
| `data_utils.py` | Generic data manipulation utilities |

### Telemetry Layer (`src/telemetry/`, `src/analytics/`)

| Component | Responsibility |
|-----------|----------------|
| `metrics.py` | Prometheus metrics for system health |
| `business_metrics.py` | Business KPIs and fraud prevention stats |

## Data Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Block
    participant FP as DeviceFingerprint
    participant PV as PaymentValidator
    participant ML as MLModels
    participant DB as Database

    Client->>API: POST /fraud/check
    API->>Block: detect_fraud(transaction)

    par Parallel Processing
        Block->>FP: generate_fingerprint()
        FP-->>Block: DeviceFingerprint
    and
        Block->>ML: predict(features)
        ML-->>Block: MLPredictions
    end

    Block->>PV: validate(transaction, fingerprint)
    PV-->>Block: PaymentValidationResult

    Block->>Block: calculate_risk_score()
    Block->>Block: apply_fraud_rules()

    Block-->>API: FraudDetectionResult
    API-->>Client: Response (< 1ms)

    API->>DB: log_transaction (async)
````

## Design Patterns

### 1. Block Pattern (Separation of Concerns)

```
┌─────────────────────────────────────────┐
│ UI Layer (API)                          │
│ - HTTP handling                         │
│ - Request/Response serialization        │
├─────────────────────────────────────────┤
│ Block Layer (Business Logic)            │
│ - Pure business logic                   │
│ - Dependency injection                  │
│ - No external dependencies              │
├─────────────────────────────────────────┤
│ Helper Layer (Services)                 │
│ - External integrations                 │
│ - Database access                       │
│ - Third-party APIs                      │
├─────────────────────────────────────────┤
│ Utils Layer (Shared Utilities)          │
│ - Error handling                        │
│ - Logging                               │
│ - Data manipulation                     │
└─────────────────────────────────────────┘
```

### 2. Protocol-Based Dependency Injection

```python
class DeviceFingerprintService(Protocol):
    def generate(self, context: Dict) -> DeviceFingerprint: ...

class FraudDetectionBlock:
    def __init__(self, device_fingerprint_service: DeviceFingerprintService):
        self.device_fingerprint = device_fingerprint_service
```

### 3. Error Boundary Pattern

```python
@error_boundary(fallback_value={"risk_score": 1.0})
def predict_fraud(features):
    return model.predict(features)
```

## Deployment Architecture

```mermaid
graph TB
    subgraph Production["Production Environment"]
        LB[Load Balancer]

        subgraph K8s["Kubernetes Cluster"]
            Pod1[Fraud Detector Pod 1]
            Pod2[Fraud Detector Pod 2]
            PodN[Fraud Detector Pod N]
        end

        subgraph Cache["Cache Layer"]
            Redis1[Redis Primary]
            Redis2[Redis Replica]
        end

        subgraph DB["Database Layer"]
            PG1[PostgreSQL Primary]
            PG2[PostgreSQL Replica]
        end

        subgraph Monitoring["Monitoring"]
            Prometheus[Prometheus]
            Grafana[Grafana]
            Alerts[Alert Manager]
        end
    end

    LB --> K8s
    K8s --> Cache
    K8s --> DB
    K8s --> Monitoring
```

## Performance Targets

These are **targets**, not measurements. No held-out evaluation runs in this
repository, so the accuracy and false-positive columns have no measured value
to report. Publishing an unmeasured number as "current" is how an aspiration
gets cited as a result.

| Metric                    | Target     | Measured                        |
| ------------------------- | ---------- | ------------------------------- |
| Fraud Check Latency (p95) | < 1ms      | see `pytest tests/performance`  |
| Throughput                | 10,000 TPS | not measured                    |
| Detection Accuracy        | > 99.9%    | not measured - no eval pipeline |
| False Positive Rate       | < 1%       | not measured - no eval pipeline |

Establishing these requires a training and evaluation pipeline that emits a
report against a held-out set; see the roadmap in the README.

## Security Considerations

1. **No PII in Logs**: Sensitive data is masked before logging
2. **Correlation IDs**: Full request tracing without exposing data
3. **Input Validation**: Pydantic models validate all inputs
4. **Rate Limiting**: Configurable rate limits per endpoint
5. **CORS**: Strict origin whitelisting
