"""
Training pipeline for fraud detection models.

Two methodology choices do most of the work here, and getting either wrong
produces impressive numbers that do not survive deployment.

**Temporal splitting.** Transactions are ordered in time and fraud patterns
drift. A random split lets the model train on transactions that happened after
the ones it is tested on, which is information it will never have in
production. Splits here are strictly forward in time.

**Fitting only on training data.** Scaling and resampling are fitted inside the
training fold alone. Fitting a scaler on the full dataset leaks the test set's
distribution; oversampling before splitting is worse still, because copies of
the same fraud land on both sides of the split and the model is scored on rows
it memorised. This is the single most common flaw in published fraud results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.training.dataset import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    TIME_COLUMN,
    Dataset,
    load_dataset,
)
from src.training.evaluation import (
    DEFAULT_REVIEW_COST,
    EvaluationReport,
    choose_threshold,
    evaluate,
)

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = "1.0.0"


@dataclass(frozen=True)
class Split:
    """One forward-in-time slice of the data."""

    name: str
    features: np.ndarray
    labels: np.ndarray
    amounts: np.ndarray
    time_start: float
    time_end: float

    def describe(self) -> dict:
        return {
            "name": self.name,
            "rows": len(self.labels),
            "fraud_rows": int(self.labels.sum()),
            "fraud_rate": round(float(self.labels.mean()), 6),
            "time_start": round(self.time_start, 1),
            "time_end": round(self.time_end, 1),
        }


@dataclass
class TrainingConfig:
    """Knobs for a training run."""

    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    # Remainder is test.

    review_cost: float = DEFAULT_REVIEW_COST
    random_state: int = 42

    # Weighting the rare class beats duplicating it: no synthetic rows, no risk
    # of the same fraud appearing on both sides of a split, and nothing extra
    # to install.
    use_class_weights: bool = True

    models: tuple[str, ...] = ("logistic_regression", "random_forest", "gradient_boosting")


@dataclass
class TrainingResult:
    """Everything one run produced."""

    data: dict
    splits: list[dict]
    config: dict
    reports: dict[str, EvaluationReport]
    best_model: str
    threshold: float
    baseline_average_precision: float
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "trained_at": self.trained_at,
            "artifact_version": ARTIFACT_VERSION,
            "data": self.data,
            "splits": self.splits,
            "config": self.config,
            # A random model's average precision equals the fraud rate. Every
            # score below is only meaningful as a multiple of this floor.
            "baseline_average_precision": round(self.baseline_average_precision, 6),
            "best_model": self.best_model,
            "threshold": round(self.threshold, 6),
            "models": {name: report.to_dict() for name, report in self.reports.items()},
        }


def temporal_split(dataset: Dataset, config: TrainingConfig) -> tuple[Split, Split, Split]:
    """
    Cut the data into train / validation / test, forward in time.

    No shuffling anywhere. Every validation row happens after every training
    row, and every test row after those.
    """
    frame = dataset.frame
    n = len(frame)

    train_end = int(n * config.train_fraction)
    validation_end = train_end + int(n * config.validation_fraction)

    bounds = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, n),
    }

    splits = []
    for name, (start, stop) in bounds.items():
        window = frame.iloc[start:stop]
        splits.append(
            Split(
                name=name,
                features=window[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64),
                labels=window[LABEL_COLUMN].to_numpy(dtype=np.int8),
                amounts=window["Amount"].to_numpy(dtype=np.float64),
                time_start=float(window[TIME_COLUMN].iloc[0]),
                time_end=float(window[TIME_COLUMN].iloc[-1]),
            )
        )

    train, validation, test = splits

    # The guarantee this function exists to provide, asserted rather than
    # assumed: a silently random split still trains and still reports numbers.
    if not (train.time_end <= validation.time_start <= validation.time_end <= test.time_start):
        raise RuntimeError("splits overlap in time; the split is not forward-in-time")

    return train, validation, test


def _build_models(config: TrainingConfig, scale_pos_weight: float) -> dict[str, Any]:
    """
    Instantiate the model zoo.

    Every model is told the classes are imbalanced. Left at defaults, each one
    converges on predicting the majority class, which is optimal for accuracy
    and useless for fraud.
    """
    weight = "balanced" if config.use_class_weights else None
    available: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight=weight,
            random_state=config.random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced_subsample" if config.use_class_weights else None,
            n_jobs=-1,
            random_state=config.random_state,
        ),
    }

    try:
        from xgboost import XGBClassifier

        available["gradient_boosting"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight if config.use_class_weights else 1.0,
            random_state=config.random_state,
            n_jobs=-1,
        )
    except ImportError:
        # Optional extra. Its absence narrows the comparison; it does not
        # invalidate the run, so it is logged rather than raised.
        logger.warning("xgboost not installed; skipping gradient_boosting")

    return {name: model for name, model in available.items() if name in config.models}


def train(
    dataset: Dataset | None = None,
    config: TrainingConfig | None = None,
    data_path: str | Path | None = None,
) -> tuple[TrainingResult, dict[str, Any], StandardScaler]:
    """
    Train and evaluate the model zoo.

    Returns the report, the fitted models, and the fitted scaler.
    """
    config = config or TrainingConfig()
    dataset = dataset or load_dataset(data_path)

    train_split, validation_split, test_split = temporal_split(dataset, config)

    logger.info(
        "Split %d rows: train=%d validation=%d test=%d",
        len(dataset.frame),
        len(train_split.labels),
        len(validation_split.labels),
        len(test_split.labels),
    )

    # Fitted on training rows only. Calling fit_transform on the full matrix
    # would fold the test set's mean and variance into the training data.
    scaler = StandardScaler().fit(train_split.features)

    x_train = scaler.transform(train_split.features)
    x_validation = scaler.transform(validation_split.features)
    x_test = scaler.transform(test_split.features)

    n_positive = max(int(train_split.labels.sum()), 1)
    scale_pos_weight = (len(train_split.labels) - n_positive) / n_positive

    models = _build_models(config, scale_pos_weight)
    if not models:
        raise RuntimeError("no models available to train")

    reports: dict[str, EvaluationReport] = {}
    fitted: dict[str, Any] = {}

    for name, model in models.items():
        logger.info("Training %s", name)
        model.fit(x_train, train_split.labels)

        # Threshold chosen on validation, then applied unchanged to test.
        validation_scores = model.predict_proba(x_validation)[:, 1]
        threshold = choose_threshold(
            validation_split.labels,
            validation_scores,
            validation_split.amounts,
            config.review_cost,
        )

        test_scores = model.predict_proba(x_test)[:, 1]
        reports[name] = evaluate(
            model_name=name,
            y_true=test_split.labels,
            y_score=test_scores,
            amounts=test_split.amounts,
            threshold=threshold,
            review_cost=config.review_cost,
        )
        fitted[name] = model

        logger.info(
            "%s: AP=%.4f ROC-AUC=%.4f recall=%.3f precision=%.3f",
            name,
            reports[name].average_precision,
            reports[name].roc_auc,
            reports[name].at_threshold.recall,
            reports[name].at_threshold.precision,
        )

    best_model = max(reports, key=lambda name: reports[name].average_precision)

    result = TrainingResult(
        data=dataset.describe(),
        splits=[s.describe() for s in (train_split, validation_split, test_split)],
        config={
            "train_fraction": config.train_fraction,
            "validation_fraction": config.validation_fraction,
            "review_cost": config.review_cost,
            "use_class_weights": config.use_class_weights,
            "split_strategy": "temporal-forward",
        },
        reports=reports,
        best_model=best_model,
        threshold=reports[best_model].at_threshold.threshold,
        baseline_average_precision=float(test_split.labels.mean()),
    )

    return result, fitted, scaler


def save_artifacts(
    result: TrainingResult,
    models: dict[str, Any],
    scaler: StandardScaler,
    output_dir: str | Path,
) -> Path:
    """
    Persist models, scaler, feature order and the metrics report.

    Feature order is saved because the serving path reconstructs a row from a
    feature dict; a different column order would silently score nonsense.

    The report is written next to the models on purpose - an artefact whose
    measured performance cannot be found is an artefact nobody can judge.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        joblib.dump(model, directory / f"{name}.joblib")

    joblib.dump(scaler, directory / "scaler.joblib")
    joblib.dump(list(FEATURE_COLUMNS), directory / "feature_names.joblib")

    report_path = directory / "metrics.json"
    report_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    logger.info("Wrote %d models and metrics.json to %s", len(models), directory)
    return report_path
