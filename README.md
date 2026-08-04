# Advanced Fraud Detection System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A credit card fraud detection service built around device fingerprinting,
payment source validation, and behavioural analysis, exposed over a FastAPI
HTTP API.

> **On scores:** with no trained model present the API serves a documented
> heuristic baseline and reports `"trained": false` from `/api/v1/health`. It
> does not pretend a heuristic is a fitted model. Train real models with
> [`scripts/train_model.py`](scripts/train_model.py) — see
> [Training and evaluation](#training-and-evaluation) for what the numbers mean
> and what they do not.

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

## Training and evaluation

`scripts/train_model.py` trains models and writes them to `MODEL_PATH`, where
the API picks them up on its next start.

```bash
# Against the real dataset
python scripts/train_model.py --data data/creditcard.csv --output models/

# Against calibrated synthetic data, to exercise the pipeline
python scripts/train_model.py --output models/
```

### The dataset

The `V1`–`V28` schema this project scores against is the [ULB Credit Card Fraud
Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
284,807 real transactions from European cardholders over two days in September
2013, of which **492 (0.172%) are fraud**. `V1`–`V28` are PCA components
published in place of the raw fields, which is how the data can be released at
all.

That file is not redistributable and is **not in this repository**. Download it
and pass `--data`. Without it the pipeline generates data calibrated to the
dataset's *published* statistics — the same fraud rate, 48-hour span and ~$88
mean amount — so the pipeline can be exercised and tested. **Every report
records which source it used**, and `is_real_data` is in the payload:

```json
"data": { "source": "synthetic-calibrated", "is_real_data": false,
          "rows": 284807, "fraud_rows": 492, "fraud_rate": 0.001727 }
```

Synthetic results demonstrate that the pipeline works. They are not evidence of
real-world accuracy, and nothing in this repo reports them as such.

### Methodology

Two choices do most of the work, and getting either wrong produces impressive
numbers that do not survive deployment.

**Forward-in-time splits.** Transactions are ordered and fraud patterns drift,
so a random split trains the model on transactions that happened *after* the
ones it is scored on — information it will never have in production. Every
validation row here happens after every training row, and every test row after
those. `temporal_split` asserts it rather than assuming it.

**Preprocessing fitted on training rows only.** Fitting a scaler on the full
dataset folds the test set's distribution into training; oversampling before
splitting is worse, because copies of the same fraud land on both sides and the
model is scored on rows it memorised. This is [the most common flaw in
published fraud results](https://arxiv.org/html/2506.02703v1), and it is
invisible in the output — so two tests assert the scaler saw training rows only,
and both fail if the fit is widened.

The rare class is handled with class weights rather than synthetic oversampling:
no invented rows, no risk of duplicates spanning a split, and nothing extra to
install.

### Metrics

**Accuracy is not reported.** At a 0.172% fraud rate, predicting "legitimate"
for everything scores 99.83% and catches nothing — the number actively rewards
the failure being guarded against.

Average precision (PR-AUC) is the headline. ROC-AUC appears beside it only for
comparability with published work, because under heavy imbalance it flatters:
the same model can show [ROC-AUC 0.957 against PR-AUC
0.708](https://machinelearningmastery.com/roc-auc-vs-precision-recall-for-imbalanced-data/).
A random scorer's average precision equals the fraud rate, so every score is
reported next to that floor.

Also emitted per model:

| Metric | Question it answers |
|---|---|
| `recall_at_precision` | If I tolerate 1 false alarm in N, how much fraud do I catch? |
| `precision_at_recall` | To catch 75% of fraud, how many alerts must be reviewed? |
| `cost.net_savings` | Amount-weighted: is this model worth running at all? |

The cost model weights each fraud by its amount, because counting cases treats
a $2 card test and a $2,000 cash-out alike. `net_savings` can be negative — a
model that alerts on everything catches all fraud and still loses money, which
recall alone would hide.

### A worked run

Full-scale run on **calibrated synthetic data** (284,807 rows, 492 fraud),
forward-in-time split, threshold chosen on validation and applied unchanged to
test:

| Model | AP | ROC-AUC | Precision | Recall | Net savings |
|---|---|---|---|---|---|
| logistic_regression | **0.796** | 0.889 | 0.309 | 0.808 | $6,733 |
| gradient_boosting | 0.796 | 0.894 | 0.286 | 0.798 | $6,554 |
| random_forest | 0.780 | 0.887 | 0.506 | 0.788 | $6,768 |

Random-baseline AP is **0.00174**, so 0.796 is a ~458× lift. Note the shape of
the two columns: ROC-AUC 0.89 reads as unremarkable while the PR-AUC shows real
skill on the rare class — the exact gap that makes ROC-AUC the wrong headline
here.

**These numbers describe a generated problem.** They show the pipeline is
correct and the metrics are wired up. Run with `--data` to measure anything
about real fraud.

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
