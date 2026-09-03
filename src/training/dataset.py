"""
Dataset loading for fraud model training.

The schema this project scores against (V1..V28, Amount, Time) is the ULB
Credit Card Fraud Detection dataset: 284,807 real transactions from European
cardholders over two days in September 2013, of which 492 (0.172%) are fraud.
V1..V28 are PCA components published in place of the raw fields, which is how
the data can be released at all.

That file is not redistributable and is not in this repository. When it is
absent, a generator calibrated to the dataset's *published* summary statistics
stands in so the pipeline can be exercised and tested.

Every load returns the provenance alongside the data, and it is carried into
the metrics report. Synthetic numbers must never be readable as measured ones:
that distinction is the whole difference between a benchmark and a claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

# Published characteristics of the ULB dataset. Used to calibrate the stand-in
# generator and to sanity-check a real file when one is supplied.
ULB_ROWS = 284_807
ULB_FRAUD_ROWS = 492
ULB_FRAUD_RATE = ULB_FRAUD_ROWS / ULB_ROWS  # 0.001727
ULB_DURATION_SECONDS = 172_792  # two days
ULB_MEAN_AMOUNT = 88.35

# Difficulty knobs for the generated stand-in, calibrated by sweeping them
# against a fixed random forest and reading off the average precision:
#
#   informative  effect  stealth     AP
#             8     1.0     0.10   0.502
#             8     1.4     0.10   0.864
#             8     1.4     0.20   0.646
#            12     1.4     0.10   0.746
#            12     1.4     0.20   0.750
#
# The values below sit near 0.75, the range published models reach on the real
# dataset. Tuned deliberately rather than by eye: the first version of this
# generator produced ROC-AUC 0.9999, which measured the generator's generosity
# and nothing else.
N_INFORMATIVE_FEATURES = 10
FRAUD_EFFECT_SIZE = 1.4
STEALTH_FRAUD_FRACTION = 0.15

PCA_FEATURES = tuple(f"V{i}" for i in range(1, 29))
FEATURE_COLUMNS = (*PCA_FEATURES, "Amount")
LABEL_COLUMN = "Class"
TIME_COLUMN = "Time"


class DataSource(str, Enum):
    """Where a dataset came from. Reported with every metric."""

    ULB_CREDITCARD = "ulb-creditcard"
    SYNTHETIC_CALIBRATED = "synthetic-calibrated"

    @property
    def is_real(self) -> bool:
        return self is DataSource.ULB_CREDITCARD


@dataclass(frozen=True)
class Dataset:
    """A labelled, time-ordered transaction set."""

    frame: pd.DataFrame
    source: DataSource

    def __post_init__(self) -> None:
        missing = [c for c in (*FEATURE_COLUMNS, LABEL_COLUMN, TIME_COLUMN) if c not in self.frame]
        if missing:
            raise ValueError(f"dataset is missing required columns: {missing}")

        if not self.frame[TIME_COLUMN].is_monotonic_increasing:
            raise ValueError(
                "rows must be sorted by Time; a temporal split on unsorted rows "
                "silently becomes a random one"
            )

    @property
    def fraud_rate(self) -> float:
        return float(self.frame[LABEL_COLUMN].mean())

    @property
    def n_fraud(self) -> int:
        return int(self.frame[LABEL_COLUMN].sum())

    def describe(self) -> dict:
        """Provenance and shape, for the metrics report."""
        return {
            "source": self.source.value,
            "is_real_data": self.source.is_real,
            "rows": len(self.frame),
            "fraud_rows": self.n_fraud,
            "fraud_rate": round(self.fraud_rate, 6),
            "duration_seconds": int(self.frame[TIME_COLUMN].max() - self.frame[TIME_COLUMN].min()),
        }


def load_dataset(path: str | Path | None = None, *, rows: int = ULB_ROWS) -> Dataset:
    """
    Load the ULB dataset from ``path``, or generate a calibrated stand-in.

    Args:
        path: location of creditcard.csv. When None or absent, data is
            generated instead and labelled as such.
        rows: row count for the generated fallback only.
    """
    if path is not None:
        csv_path = Path(path)
        if csv_path.is_file():
            return _load_csv(csv_path)

    return generate_calibrated(rows=rows)


def _load_csv(path: Path) -> Dataset:
    frame = pd.read_csv(path)
    frame = frame.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)
    return Dataset(frame=frame, source=DataSource.ULB_CREDITCARD)


def generate_calibrated(*, rows: int = ULB_ROWS, seed: int = 42) -> Dataset:
    """
    Generate transactions matching the ULB dataset's published statistics.

    Calibrated to the real figures - 0.172% fraud, a 48-hour span, ~$88 mean
    amount, zero-centred PCA components - so the pipeline exercises realistic
    class imbalance and scale. The *relationships* between features are
    invented, so a model trained here learns a toy problem: useful for testing
    that the pipeline is correct, useless as evidence that it is accurate.

    The difficulty is tuned deliberately. An earlier version shifted all 28
    features by 1.2 sigma, giving roughly six sigma of separation and a
    near-perfect ROC-AUC of 0.9999 - a number that proves only that the
    generator was generous. Real fraud is partly indistinguishable from
    legitimate spending, so this reproduces two properties that make it hard:

    - only a few components carry signal, as in the real data where most PCA
      components barely discriminate
    - a share of fraud is drawn from the legitimate distribution outright,
      giving the irreducible error that keeps recall short of 1.0
    """
    rng = np.random.default_rng(seed)

    n_fraud = max(1, round(rows * ULB_FRAUD_RATE))
    n_legit = rows - n_fraud

    legit = rng.standard_normal((n_legit, len(PCA_FEATURES)))

    fraud = rng.standard_normal((n_fraud, len(PCA_FEATURES)))

    # Signal in a minority of components only.
    informative = rng.choice(len(PCA_FEATURES), size=N_INFORMATIVE_FEATURES, replace=False)
    fraud[:, informative] += rng.normal(
        loc=FRAUD_EFFECT_SIZE, scale=0.6, size=(n_fraud, N_INFORMATIVE_FEATURES)
    )
    fraud[:, informative] *= 1.4  # heavier tails on the discriminating axes

    # Stealthy fraud, indistinguishable by construction. Without it the task
    # has no error floor and every model looks perfect.
    n_stealth = int(n_fraud * STEALTH_FRAUD_FRACTION)
    if n_stealth:
        fraud[:n_stealth] = rng.standard_normal((n_stealth, len(PCA_FEATURES)))

    features = np.vstack([legit, fraud])
    labels = np.concatenate([np.zeros(n_legit, dtype=np.int8), np.ones(n_fraud, dtype=np.int8)])

    # Log-normal amounts, scaled to the published ~$88 mean. Fraud skews low in
    # the real data - card testing uses small amounts - so it is not simply the
    # large transactions that are suspicious.
    amounts = rng.lognormal(mean=3.0, sigma=1.5, size=rows)
    amounts *= ULB_MEAN_AMOUNT / amounts.mean()

    times = np.sort(rng.uniform(0, ULB_DURATION_SECONDS, size=rows))

    # Shuffle so fraud is spread across the window rather than stacked at the
    # end, which would make a temporal split trivially separable.
    order = rng.permutation(rows)

    frame = pd.DataFrame(features[order], columns=list(PCA_FEATURES))
    frame["Amount"] = amounts[order]
    frame[TIME_COLUMN] = times
    frame[LABEL_COLUMN] = labels[order]

    frame = frame.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)

    return Dataset(frame=frame, source=DataSource.SYNTHETIC_CALIBRATED)
