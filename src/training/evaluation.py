"""
Evaluation metrics for fraud detection.

Accuracy is excluded deliberately. At the ULB dataset's 0.172% fraud rate, a
model that predicts "legitimate" for every transaction scores 99.83% accurate
and catches nothing - the number is not merely unhelpful, it actively rewards
the failure mode being guarded against.

Average precision (PR-AUC) is the headline instead. ROC-AUC is reported beside
it only for comparability with published work, because under heavy imbalance it
flatters: the same model can show ROC-AUC 0.957 against PR-AUC 0.708, since the
huge legitimate class dominates the false-positive rate and hides poor
precision on the rare class.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

# A missed fraud costs the transaction amount. A false alarm costs a review,
# and enough of them cost the customer. This is the exchange rate between the
# two, and it is the only reason a threshold can be chosen non-arbitrarily.
DEFAULT_REVIEW_COST = 3.0


@dataclass
class ThresholdMetrics:
    """Metrics at one operating point."""

    threshold: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    alerts: int
    alert_rate: float

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class EvaluationReport:
    """Complete evaluation of one model on one split."""

    model_name: str
    average_precision: float
    roc_auc: float
    brier_score: float
    positives: int
    negatives: int
    at_threshold: ThresholdMetrics
    recall_at_precision: dict[str, float] = field(default_factory=dict)
    precision_at_recall: dict[str, float] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            # Primary metric. Baseline for a random model is the fraud rate
            # itself, so a value must always be read against that floor.
            "average_precision": round(self.average_precision, 6),
            "roc_auc": round(self.roc_auc, 6),
            "brier_score": round(self.brier_score, 6),
            "positives": self.positives,
            "negatives": self.negatives,
            "at_threshold": self.at_threshold.to_dict(),
            "recall_at_precision": self.recall_at_precision,
            "precision_at_recall": self.precision_at_recall,
            "cost": self.cost,
        }


def metrics_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> ThresholdMetrics:
    """Confusion-matrix metrics for a single decision threshold."""
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return ThresholdMetrics(
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fp / (fp + tn) if (fp + tn) else 0.0,
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        alerts=int(tp + fp),
        alert_rate=float((tp + fp) / len(y_true)) if len(y_true) else 0.0,
    )


def recall_at_precision(
    y_true: np.ndarray, y_score: np.ndarray, targets: tuple[float, ...]
) -> dict:
    """
    Best achievable recall at each minimum precision.

    This is the question a fraud team actually asks: "if I can tolerate one
    false alarm in every N alerts, how much fraud do I catch?"
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    results = {}
    for target in targets:
        feasible = recall[precision >= target]
        results[f"p>={target:.2f}"] = round(float(feasible.max()), 6) if feasible.size else 0.0
    return results


def precision_at_recall(
    y_true: np.ndarray, y_score: np.ndarray, targets: tuple[float, ...]
) -> dict:
    """Best achievable precision at each minimum recall - the inverse trade."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    results = {}
    for target in targets:
        feasible = precision[recall >= target]
        results[f"r>={target:.2f}"] = round(float(feasible.max()), 6) if feasible.size else 0.0
    return results


def cost_analysis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    review_cost: float = DEFAULT_REVIEW_COST,
) -> dict:
    """
    Money saved versus doing nothing, in the units of ``amounts``.

    Fraud is not uniformly expensive, so counting cases treats a $2 card test
    and a $2,000 cash-out as the same event. This weights each by its amount:
    a caught fraud saves its value, a missed one loses it, and a false alarm
    costs one review.

    ``net_savings`` can be negative - a model that alerts on everything catches
    all fraud and still loses money. Reporting only recall would hide that.
    """
    flagged = y_score >= threshold
    is_fraud = y_true == 1

    prevented = float(amounts[flagged & is_fraud].sum())
    missed = float(amounts[~flagged & is_fraud].sum())
    reviews = int((flagged & ~is_fraud).sum())
    exposure = float(amounts[is_fraud].sum())

    net = prevented - reviews * review_cost

    return {
        "review_cost_per_alert": review_cost,
        "total_fraud_exposure": round(exposure, 2),
        "fraud_prevented": round(prevented, 2),
        "fraud_missed": round(missed, 2),
        "false_alarm_cost": round(reviews * review_cost, 2),
        "net_savings": round(net, 2),
        "savings_rate": round(net / exposure, 6) if exposure else 0.0,
    }


def choose_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: np.ndarray,
    review_cost: float = DEFAULT_REVIEW_COST,
) -> float:
    """
    Pick the threshold maximising net savings.

    Must be called on validation data, never on test: choosing an operating
    point on the same rows used to report performance is how a tuned threshold
    turns into an inflated result.
    """
    _, _, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5

    # Scan a bounded sample - the curve yields one threshold per distinct
    # score, which is hundreds of thousands of points on a full dataset.
    candidates = np.quantile(thresholds, np.linspace(0, 1, num=min(200, thresholds.size)))

    best_threshold, best_savings = 0.5, -np.inf
    for threshold in candidates:
        savings = cost_analysis(y_true, y_score, amounts, threshold, review_cost)["net_savings"]
        if savings > best_savings:
            best_threshold, best_savings = float(threshold), savings

    return best_threshold


def evaluate(
    model_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    review_cost: float = DEFAULT_REVIEW_COST,
) -> EvaluationReport:
    """Full evaluation of one model's scores at a chosen threshold."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    return EvaluationReport(
        model_name=model_name,
        average_precision=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        brier_score=float(np.mean((y_score - y_true) ** 2)),
        positives=int(y_true.sum()),
        negatives=int(len(y_true) - y_true.sum()),
        at_threshold=metrics_at_threshold(y_true, y_score, threshold),
        recall_at_precision=recall_at_precision(y_true, y_score, (0.5, 0.75, 0.9, 0.95)),
        precision_at_recall=precision_at_recall(y_true, y_score, (0.5, 0.75, 0.9)),
        cost=cost_analysis(y_true, y_score, amounts, threshold, review_cost),
    )
