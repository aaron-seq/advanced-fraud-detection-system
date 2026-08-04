"""
Fraud detection service.

Adapters that wire the pure business logic in ``src/block`` to the API layer.
The detection logic itself lives in FraudDetectionBlock and is not duplicated
here; this module only supplies the collaborators it declares as Protocols.

This replaces an earlier service that trained XGBoost, LightGBM and CatBoost on
`np.random.randn` at every startup and served the resulting probabilities as
fraud scores. Scores derived from noise are not merely useless, they are
actively misleading, so that path is gone.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from src.analytics import BusinessMetricsCollector
from src.block.fraud_detection_block import FraudDetectionBlock, FraudDetectionConfig
from src.block.models import (
    DeviceFingerprint,
    FraudDetectionResult,
    MLPrediction,
    Transaction,
    UserProfile,
)
from src.fraud_detection.device_fingerprinting import VisitorFingerprintEngine
from src.fraud_detection.payment_validation import PaymentSourceValidator
from src.telemetry import get_metrics

logger = logging.getLogger(__name__)

HEURISTIC_MODEL_NAME = "heuristic-baseline"


class FingerprintEngineAdapter:
    """Adapts VisitorFingerprintEngine to the DeviceFingerprintService protocol."""

    def __init__(self, engine: VisitorFingerprintEngine | None = None):
        self._engine = engine or VisitorFingerprintEngine()

    def generate(self, device_context: dict[str, Any]) -> DeviceFingerprint:
        return self._engine.generate_fingerprint(device_context)


class HeuristicMLModel:
    """
    Transparent baseline scorer used when no trained model is available.

    It is named ``heuristic-baseline`` in every prediction so that a caller can
    always tell a heuristic score from a trained-model score. Reporting a
    heuristic under a model's name is how unvalidated scores end up trusted as
    if they had been measured.
    """

    name = HEURISTIC_MODEL_NAME
    version = "1.0.0"

    # Scale at which the mean PCA magnitude saturates the score. The components
    # are unit-variance, so a mean absolute deviation of 6 is far into the tail.
    SATURATION_MAGNITUDE = 6.0

    def predict(self, features: dict[str, float]) -> dict[str, MLPrediction]:
        # Only the PCA components are scored. They are standardised to roughly
        # unit variance, whereas raw columns like Amount and Time are orders of
        # magnitude larger - averaging those in pins the score at 1.0 for
        # essentially every real transaction, which is worse than no signal
        # because it looks like a confident fraud call.
        pca_values = [
            abs(value)
            for name, value in features.items()
            if name.startswith("V") and name[1:].isdigit()
        ]

        if not pca_values:
            # No usable evidence either way. Deliberately not 0.0: absence of
            # features is not evidence of legitimacy.
            return {
                self.name: MLPrediction(
                    model_name=self.name,
                    model_version=self.version,
                    fraud_probability=0.5,
                    confidence=0.0,
                    features_used=[],
                )
            }

        mean_magnitude = sum(pca_values) / len(pca_values)
        probability = min(mean_magnitude / self.SATURATION_MAGNITUDE, 1.0)

        return {
            self.name: MLPrediction(
                model_name=self.name,
                model_version=self.version,
                fraud_probability=probability,
                confidence=0.3,  # deliberately low; this is not a fitted model
                features_used=sorted(
                    name for name in features if name.startswith("V") and name[1:].isdigit()
                ),
            )
        }


class JoblibEnsembleModel:
    """
    Serves predictions from trained models persisted as joblib artefacts.

    The scaler is applied here, not optional. Training standardises features
    before fitting, so serving a raw row to a model fitted on scaled input
    produces a confidently wrong probability rather than an error - the worst
    kind of failure for a fraud score, because nothing surfaces it.
    """

    def __init__(
        self,
        models: dict[str, Any],
        feature_names: list[str],
        scaler: Any = None,
        version: str = "unknown",
    ):
        self._models = models
        self._feature_names = feature_names
        self._scaler = scaler
        self._version = version

    def predict(self, features: dict[str, float]) -> dict[str, MLPrediction]:
        import numpy as np

        # Column order comes from feature_names.joblib. Reconstructing a row
        # from a dict in any other order would silently score the wrong values
        # into the wrong coefficients.
        row = np.array([[features.get(name, 0.0) for name in self._feature_names]])

        if self._scaler is not None:
            row = self._scaler.transform(row)

        predictions = {}
        for name, model in self._models.items():
            probability = float(model.predict_proba(row)[0][1])
            predictions[name] = MLPrediction(
                model_name=name,
                model_version=self._version,
                fraud_probability=probability,
                confidence=0.9,
                features_used=self._feature_names,
            )

        return predictions


class InMemoryUserProfileRepository:
    """
    User profiles held in process memory.

    Not persistent and not shared between workers: profiles are lost on restart
    and each worker builds its own view. Swap in a database-backed repository
    before relying on behavioural history in production.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, UserProfile] = {}

    def get_by_user_id(self, user_id: str) -> UserProfile | None:
        return self._profiles.get(user_id)

    def save(self, profile: UserProfile) -> None:
        self._profiles[profile.user_id] = profile


class FraudDetectionService:
    """Application-facing entry point for fraud detection."""

    def __init__(self, config: FraudDetectionConfig | None = None):
        self._config = config
        self._model_service: Any = None
        self._block: FraudDetectionBlock | None = None
        self.profiles = InMemoryUserProfileRepository()
        # Records every decision this process actually made, so the analytics
        # endpoint reports observed numbers. The endpoint it replaced returned
        # a hardcoded 15847 transactions and 99.87% accuracy with dates from
        # 2024 - figures no run had ever produced.
        self.metrics = BusinessMetricsCollector()

    async def initialize(self, model_path: str = "./models") -> None:
        """Load models and assemble the detection block."""
        self._model_service = await asyncio.to_thread(self._load_model_service, model_path)

        self._block = FraudDetectionBlock(
            device_fingerprint_service=FingerprintEngineAdapter(),
            payment_validator=PaymentSourceValidator(),
            ml_model_service=self._model_service,
            telemetry=get_metrics(),
            user_profile_repository=self.profiles,
            config=self._config,
        )

        logger.info(
            "Fraud detection service ready using model '%s'",
            self.model_health()["active_model"],
        )

    @staticmethod
    def _load_model_service(model_path: str) -> Any:
        """
        Load trained models from ``model_path``, falling back to the heuristic.

        A missing model directory is an expected state on a fresh checkout, not
        an error, but it is logged at warning level because serving heuristic
        scores in production is a meaningful degradation.
        """
        directory = Path(model_path)
        artefacts = sorted(directory.glob("*.joblib")) if directory.is_dir() else []

        if not artefacts:
            logger.warning(
                "No trained models found in %s; using the %s scorer. "
                "Predictions are heuristic, not learned.",
                directory,
                HEURISTIC_MODEL_NAME,
            )
            return HeuristicMLModel()

        # joblib deserialises via pickle, which executes arbitrary code. These
        # artefacts come from MODEL_PATH, an operator-set deployment path that
        # must be treated as trusted infrastructure - never point it at a
        # user-writable or user-uploaded directory.
        import joblib

        # feature_names and scaler are sidecars, not models. Loading either
        # into the ensemble would call predict_proba on a StandardScaler.
        models = {}
        feature_names: list[str] = []
        scaler = None
        try:
            for artefact in artefacts:
                if artefact.stem == "feature_names":
                    feature_names = joblib.load(artefact)
                elif artefact.stem == "scaler":
                    scaler = joblib.load(artefact)
                else:
                    models[artefact.stem] = joblib.load(artefact)
        except Exception:
            # Unpickling needs the libraries the artefact was built with, and
            # a stale or truncated file raises here too. Degrading to the
            # heuristic keeps the service up; crashing on startup would take
            # fraud detection down entirely over a bad file on disk.
            logger.exception(
                "Failed to load artefacts from %s; falling back to the %s scorer",
                directory,
                HEURISTIC_MODEL_NAME,
            )
            return HeuristicMLModel()

        if not models or not feature_names:
            logger.warning(
                "Model directory %s is incomplete (models=%d, feature_names=%d); "
                "falling back to the %s scorer.",
                directory,
                len(models),
                len(feature_names),
                HEURISTIC_MODEL_NAME,
            )
            return HeuristicMLModel()

        if scaler is None:
            # Refuse rather than serve unscaled input to models fitted on
            # scaled features: the resulting scores look valid and are not.
            logger.error(
                "Model directory %s has models but no scaler.joblib. Serving "
                "unscaled features would produce silently wrong probabilities, "
                "so the %s scorer is used instead.",
                directory,
                HEURISTIC_MODEL_NAME,
            )
            return HeuristicMLModel()

        version = FraudDetectionService._artefact_version(directory)
        logger.info(
            "Loaded %d trained models (version %s) from %s",
            len(models),
            version,
            directory,
        )
        return JoblibEnsembleModel(models, feature_names, scaler, version)

    @staticmethod
    def _artefact_version(directory: Path) -> str:
        """
        Read the training timestamp from metrics.json, if the pipeline wrote it.

        Reporting which artefact produced a score is what makes a bad model
        traceable back to the run that trained it.
        """
        metrics = directory / "metrics.json"
        if not metrics.is_file():
            return "unknown"

        try:
            report = json.loads(metrics.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read %s: %s", metrics, exc)
            return "unknown"

        return str(report.get("trained_at", "unknown"))

    def model_health(self) -> dict[str, Any]:
        """Report which scorer is active and whether it is a trained model."""
        if self._model_service is None:
            return {"loaded": False, "count": 0, "active_model": None}

        if isinstance(self._model_service, HeuristicMLModel):
            return {
                "loaded": True,
                "count": 1,
                "active_model": HEURISTIC_MODEL_NAME,
                "trained": False,
            }

        return {
            "loaded": True,
            "count": len(self._model_service._models),
            "active_model": "joblib-ensemble",
            "trained": True,
        }

    async def analyze_transaction(
        self,
        transaction: Transaction,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Analyse a single transaction and return the serialised result."""
        if self._block is None:
            raise RuntimeError("Service not initialized; call initialize() first")

        result = await asyncio.to_thread(self._block.detect_fraud, transaction, request_id)
        self._record(transaction, result)
        return result.to_dict()

    def _record(self, transaction: Transaction, result: FraudDetectionResult) -> None:
        """Record a decision for the analytics endpoint."""
        self.metrics.record_decision(
            transaction_id=transaction.id,
            decision=result.decision.value,
            amount=transaction.amount,
            risk_score=result.risk_score,
            model_name=self.model_health()["active_model"] or "unknown",
            latency_ms=result.processing_time_ms,
        )

    def analytics_summary(self, days_back: int = 7) -> dict[str, Any]:
        """
        Analytics over decisions this process has actually made.

        Numbers start at zero on boot and are per-process, because they come
        from an in-memory collector. That is stated in the payload rather than
        hidden, so an empty dashboard reads as "nothing recorded yet" instead
        of being mistaken for a system processing no traffic.
        """
        report = self.metrics.generate_report(period_days=days_back)
        report["realtime"] = self.metrics.get_realtime_stats()
        report["source"] = "in-process counters since startup; not persisted"
        report["model"] = self.model_health()
        return report

    async def analyze_batch(
        self,
        transactions: list[Transaction],
        request_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Analyse transactions concurrently.

        Detection is CPU-bound and synchronous, so each item runs in the
        default thread pool rather than blocking the event loop in a plain
        for-loop.
        """
        if self._block is None:
            raise RuntimeError("Service not initialized; call initialize() first")

        results = await asyncio.gather(
            *(
                asyncio.to_thread(self._block.detect_fraud, transaction, request_id)
                for transaction in transactions
            ),
        )

        for transaction, result in zip(transactions, results, strict=True):
            self._record(transaction, result)

        return [result.to_dict() for result in results]
