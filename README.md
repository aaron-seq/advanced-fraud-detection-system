# Advanced Fraud Detection System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A credit card fraud detection service built around device fingerprinting,
payment source validation, and behavioural analysis, exposed over a FastAPI
HTTP API.

> **On scores:** with no trained model present the API serves a documented
> heuristic baseline and reports `"trained": false` from `/api/v1/health`. It
> does not pretend a heuristic is a fitted model. No accuracy figures are
> published here because none have been measured on a held-out dataset in this
> repository — see [Known limitations](#known-limitations).

---

## What it does

A transaction is scored by combining four independent signals:

| Signal | Source | What it contributes |
|---|---|---|
| **Device** | `src/fraud_detection/device_fingerprinting/` | Fingerprint stability, VPN/proxy/Tor/datacenter detection, spoofing and bot indicators, whether the device is bound to the user |
| **Payment source** | `src/fraud_detection/payment_validation/` | Amount bands, deviation from the user's norm, location familiarity, velocity, cross-border rules |
| **Model** | `app/services/fraud_detection_service.py` | A trained ensemble if `MODEL_PATH` holds artefacts, otherwise the heuristic baseline |
| **Behaviour** | `src/fraud_detection/behavioral/` | Spending-pattern deviation and time-of-day anomalies |

These are fused into a 0–100 risk score and mapped to a decision
(`approve` / `additional_auth_required` / `review` / `deny`) plus an
authentication requirement (none / OTP / MFA / biometric / manual review).

### Architecture

```
HTTP request
     │
     ▼
app/main.py                      FastAPI: routing, auth, validation only
     │                           middleware: trusted host → CORS → monitoring → rate limit
     ▼
app/services/…_service.py        Adapters that satisfy the block's Protocols
     │
     ▼
src/block/fraud_detection_block.py    All detection logic. No I/O, no framework
     │                                imports, every dependency injected.
     ├── device fingerprint  →  src/fraud_detection/device_fingerprinting/
     ├── payment validation  →  src/fraud_detection/payment_validation/
     ├── behavioural         →  src/fraud_detection/behavioral/
     └── telemetry           →  src/telemetry/  (Prometheus)
```

The rule that keeps this maintainable: **`src/` never imports from `app/`.**
Business logic stays framework-free and unit-testable; `app/` only translates
HTTP to domain objects and back.

---

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/aaron-seq/advanced-fraud-detection-system.git
cd advanced-fraud-detection-system

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env             # optional; sane defaults work for local dev
uvicorn app.main:app --reload
```

Interactive docs at <http://127.0.0.1:8000/api/docs> (disabled when
`ENVIRONMENT=production`).

### Making a request

Every detection endpoint requires a bearer token.

```bash
TOKEN=$(python -c "from app.core.security import create_access_token; print(create_access_token('user_1'))")

curl -X POST http://127.0.0.1:8000/api/v1/detect-fraud \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "transaction_id": "txn_1",
        "amount": 4200.0,
        "transaction_type": "purchase",
        "transaction_country": "US",
        "transaction_city": "New York",
        "features": {"V1": -1.36, "V2": 0.07, "V3": 2.53, "Amount": 149.62}
      }'
```

```json
{
  "transaction_id": "txn_1",
  "decision": "approve",
  "is_fraud": false,
  "fraud_probability": 0.22,
  "risk_score": 31.0,
  "risk_level": "medium",
  "model_version": "2.9.0",
  "processing_time_ms": 3.3
}
```

(`processing_time_ms` varies; the rest is the actual output for that request.)

### Docker

```bash
echo "SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
docker compose up --build
```

Starts the API on `:8000` with PostgreSQL and Redis. The API refuses to start
without `SECRET_KEY` when `ENVIRONMENT=production`.

---

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/` | — | Liveness |
| `GET` | `/api/v1/health` | — | Readiness: database, cache, model state |
| `POST` | `/api/v1/detect-fraud` | Bearer | Score one transaction |
| `POST` | `/api/v1/detect-fraud/batch` | Bearer | Score up to 1000 transactions |
| `GET` | `/api/v1/analytics/dashboard-data` | Bearer | Metrics over decisions this process recorded |

`/api/v1/health` reports `degraded` with a 200 rather than a bare 503, so you
can see *which* dependency is down:

```json
{
  "status": "degraded",
  "components": {"database": "healthy", "cache": "unhealthy", "ml_models": "healthy"},
  "models": {"loaded": true, "active_model": "heuristic-baseline", "trained": false}
}
```

---

## Configuration

All settings come from environment variables (or `.env`). Full list in
[.env.example](.env.example). The ones that matter most:

| Variable | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` disables `/api/docs` and requires `SECRET_KEY` |
| `SECRET_KEY` | *(ephemeral)* | **Required in production.** Outside production a random key is generated per process, so tokens do not survive a restart |
| `DATABASE_URL` | `sqlite+aiosqlite:///./fraud_detection.db` | Must use an async driver |
| `REDIS_URL` | `redis://localhost:6379/0` | Cache is optional; unavailability degrades latency, not correctness |
| `MODEL_PATH` | `./models` | Empty ⇒ heuristic baseline. Treat as trusted infrastructure: joblib deserialises via pickle |
| `TRUST_PROXY_HEADERS` | `false` | Enable **only** behind a proxy that overwrites `X-Forwarded-For` |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1` | Requests with any other `Host` get a 400 |

### Serving a trained model

Put `*.joblib` files plus a `feature_names.joblib` in `MODEL_PATH`. The service
loads them at startup and `/api/v1/health` flips to `"trained": true`. With an
incomplete directory it logs a warning and falls back to the heuristic rather
than serving partial results.

---

## Security

- **Authentication on every detection and analytics endpoint.** Tokens are
  HS256 JWTs via PyJWT; invalid, expired, or foreign-signed tokens get a 401.
- **Dependencies audited.** `pip-audit --strict` runs in CI and currently
  reports no known vulnerabilities. Reaching that meant moving off
  `python-jose` (unpatched advisories, and it pulls in `ecdsa` and `rsa` for
  asymmetric algorithms this service never uses) and onto newer `starlette`
  and `python-multipart`.
- **No default signing key.** A shipped default is a published key.
- **Rate limiting** per client, in a bounded LRU so the limiter cannot itself
  be turned into a memory-exhaustion vector. Proxy headers are untrusted by
  default.
- **Strict input validation.** Non-finite feature values (`NaN`, `Infinity`)
  are rejected — `NaN` compares `False` against every threshold, so one could
  otherwise slip past every rule meant to catch it.
- **Fails closed.** A detection error returns a 500; it is never downgraded to
  a default approval.

Report vulnerabilities via GitHub issues.

---

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt

pytest                                    # unit + integration
pytest --cov=app --cov=src                # with coverage
pytest tests/performance                  # excluded from the default run

ruff check app/ src/ tests/               # lint
ruff format app/ src/ tests/              # format
```

CI runs lint, formatting, tests on Python 3.11/3.12/3.13, `pip-audit`, and a
package build. See [.github/workflows/ci.yml](.github/workflows/ci.yml).

### Layout

```
app/          FastAPI layer — routing, auth, validation, adapters
  core/       config, security, database, cache
  models/     Pydantic request/response schemas
  services/   adapters wiring src/block into the API
  utils/      rate limiting, request monitoring
src/          Framework-free business logic
  block/      FraudDetectionBlock + domain models
  fraud_detection/
              device_fingerprinting/, payment_validation/, behavioral/
  analytics/  business metrics
  telemetry/  Prometheus metrics
  utils/      logging, error handling, data helpers
tests/        unit/, integration/, performance/
dashboard/    Streamlit + static monitoring UI
docs/         architecture notes
```

Standalone research scripts (`Quantum_Fraud_Dectector.py`,
`federated_learning_fraud.py`, `graph_neural_network_fraud.py`,
`fraud_credit.py`) and the notebooks are kept for reference. Nothing imports
them, they need the optional extras, and they are excluded from lint and CI.

---

## Known limitations

Stated plainly rather than papered over.

- **No measured accuracy.** No held-out evaluation is run in this repository,
  so no precision/recall/AUC figures are published. The default scorer is an
  explicitly-labelled heuristic.
- **Placeholder integrations.** Geolocation, IP reputation, and VPN/Tor
  detection in `attribute_collector.py` return fixed values; they are shaped
  for a real provider but are not wired to one. Device risk is therefore driven
  mainly by stability and behavioural signals today.
- **In-memory user profiles and analytics.** `InMemoryUserProfileRepository`
  and the analytics counters do not persist and are not shared across workers,
  so behavioural history and dashboard totals reset on restart. The analytics
  payload says so in its `source` field rather than leaving you to assume.
- **`dashboard/app.py` is a UI demo serving random data.** It is labelled as
  such in its title and with a banner on the page. Real numbers come from
  `/api/v1/analytics/dashboard-data`, which `dashboard/streamlit_app.py` reads.
- **Per-process rate limiting.** With N workers the effective limit is
  `limit × N`. Move to Redis if that matters.
- **Drift detection measures mean shift only.** A distribution that keeps its
  mean but widens is not currently flagged.
- **Weekday patterns are collected but not scored**, so a transaction on a day
  the user never transacts is not itself a signal.
- **No password hashing.** There is no login endpoint. The previous `passlib`
  setup was dead code that raised on every call against modern `bcrypt`; add
  hashing with `bcrypt` directly when an endpoint needs it.

## Roadmap

1. Persist user profiles behind the existing `UserProfileRepository` Protocol.
2. Wire a real GeoIP/IP-reputation provider into `NetworkContextCollector`.
3. Add a training pipeline that emits versioned artefacts plus an evaluation
   report, so measured metrics can replace the heuristic honestly.
4. Move rate limiting to Redis for correct multi-worker limits.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Keep `src/` free of `app/` imports, add
a test with behavioural changes, and make sure `ruff check` and `pytest` pass.

## License

MIT — see [LICENSE](LICENSE).

## Author

Aaron Emmanuel Xavier Sequeira —
[github.com/aaron-seq](https://github.com/aaron-seq)
