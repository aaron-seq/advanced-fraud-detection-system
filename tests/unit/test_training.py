"""
Tests for the training pipeline.

These guard methodology, not accuracy. A leaking pipeline still trains, still
prints metrics, and still looks better than a correct one - which is exactly
why the guarantees have to be asserted rather than assumed.
"""

import json

import numpy as np
import pytest

from src.training.dataset import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    TIME_COLUMN,
    ULB_FRAUD_RATE,
    DataSource,
    generate_calibrated,
    load_dataset,
)
from src.training.evaluation import (
    choose_threshold,
    cost_analysis,
    evaluate,
    metrics_at_threshold,
    precision_at_recall,
    recall_at_precision,
)
from src.training.pipeline import TrainingConfig, save_artifacts, temporal_split, train


@pytest.fixture(scope="module")
def dataset():
    return generate_calibrated(rows=12_000, seed=7)


class TestDatasetProvenance:
    """Synthetic numbers must never be mistakable for measured ones."""

    def test_generated_data_is_labelled_synthetic(self, dataset):
        assert dataset.source is DataSource.SYNTHETIC_CALIBRATED
        assert dataset.source.is_real is False
        assert dataset.describe()["is_real_data"] is False

    def test_missing_csv_falls_back_to_synthetic_not_an_error(self, tmp_path):
        result = load_dataset(tmp_path / "nope.csv", rows=2_000)

        assert result.source is DataSource.SYNTHETIC_CALIBRATED

    def test_real_csv_is_labelled_real(self, tmp_path, dataset):
        path = tmp_path / "creditcard.csv"
        dataset.frame.to_csv(path, index=False)

        assert load_dataset(path).source is DataSource.ULB_CREDITCARD


class TestCalibration:
    def test_fraud_rate_matches_the_published_figure(self, dataset):
        assert dataset.fraud_rate == pytest.approx(ULB_FRAUD_RATE, abs=5e-4)

    def test_rows_are_time_ordered(self, dataset):
        assert dataset.frame[TIME_COLUMN].is_monotonic_increasing

    def test_unsorted_input_is_rejected(self, dataset):
        from src.training.dataset import Dataset

        shuffled = dataset.frame.sample(frac=1.0, random_state=0)

        with pytest.raises(ValueError, match="sorted by Time"):
            Dataset(frame=shuffled, source=DataSource.SYNTHETIC_CALIBRATED)

    def test_the_task_is_neither_trivial_nor_impossible(self):
        """
        A generator that makes fraud trivially separable produces a flattering
        score that measures the generator. The first version of this one hit
        ROC-AUC 0.9999; this pins the difficulty near what published models
        reach on the real dataset.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import average_precision_score

        data = generate_calibrated(rows=40_000, seed=3).frame
        x = data[list(FEATURE_COLUMNS)].to_numpy()
        y = data[LABEL_COLUMN].to_numpy()
        cut = int(len(y) * 0.7)

        model = RandomForestClassifier(
            n_estimators=80,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=0,
        ).fit(x[:cut], y[:cut])

        ap = average_precision_score(y[cut:], model.predict_proba(x[cut:])[:, 1])

        assert 0.3 < ap < 0.95, f"generated task is miscalibrated (AP={ap:.3f})"


class TestTemporalSplit:
    """
    Splits must run forward in time. A random split lets a model train on
    transactions that happened after the ones it is scored on - information it
    will never have in production.
    """

    def test_splits_do_not_overlap_in_time(self, dataset):
        train_split, validation, test = temporal_split(dataset, TrainingConfig())

        assert train_split.time_end <= validation.time_start
        assert validation.time_end <= test.time_start

    def test_every_row_lands_in_exactly_one_split(self, dataset):
        train_split, validation, test = temporal_split(dataset, TrainingConfig())

        total = len(train_split.labels) + len(validation.labels) + len(test.labels)

        assert total == len(dataset.frame)

    def test_split_sizes_follow_the_configured_fractions(self, dataset):
        config = TrainingConfig(train_fraction=0.5, validation_fraction=0.25)
        train_split, validation, _test = temporal_split(dataset, config)

        n = len(dataset.frame)

        assert len(train_split.labels) == pytest.approx(n * 0.5, rel=0.01)
        assert len(validation.labels) == pytest.approx(n * 0.25, rel=0.01)

    def test_fraud_appears_in_every_split(self, dataset):
        """Otherwise the evaluation would have nothing to measure."""
        for split in temporal_split(dataset, TrainingConfig()):
            assert split.labels.sum() > 0, f"{split.name} contains no fraud"


class TestNoLeakage:
    """
    The scaler must see training rows only. Fitting it on the full dataset
    folds the test set's mean and variance into training - the single most
    common flaw in published fraud results, and invisible in the output.
    """

    def test_scaler_is_fitted_on_training_rows_only(self, dataset):
        config = TrainingConfig(models=("logistic_regression",))
        train_split, _, _ = temporal_split(dataset, config)

        _, _, scaler = train(dataset=dataset, config=config)

        expected = train_split.features.mean(axis=0)

        np.testing.assert_allclose(scaler.mean_, expected, rtol=1e-9)

    def test_scaler_does_not_match_full_dataset_statistics(self, dataset):
        """The inverse check: it must NOT have seen everything."""
        config = TrainingConfig(models=("logistic_regression",))

        _, _, scaler = train(dataset=dataset, config=config)

        full = dataset.frame[list(FEATURE_COLUMNS)].to_numpy().mean(axis=0)

        assert not np.allclose(scaler.mean_, full, rtol=1e-6)


class TestEvaluationMetrics:
    def test_perfect_scores_give_average_precision_one(self):
        y = np.array([0, 0, 1, 1])
        report = evaluate("perfect", y, np.array([0.1, 0.2, 0.9, 0.95]), np.ones(4), 0.5)

        assert report.average_precision == pytest.approx(1.0)
        assert report.at_threshold.recall == 1.0

    def test_threshold_metrics_match_a_hand_computed_confusion_matrix(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.8, 0.7, 0.2])

        m = metrics_at_threshold(y, scores, 0.5)

        assert (m.true_positives, m.false_positives) == (1, 1)
        assert (m.true_negatives, m.false_negatives) == (1, 1)
        assert m.precision == pytest.approx(0.5)
        assert m.recall == pytest.approx(0.5)

    def test_recall_at_precision_is_monotonically_non_increasing(self):
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.05, 2_000)
        scores = np.clip(y * 0.4 + rng.random(2_000) * 0.6, 0, 1)

        result = recall_at_precision(y, scores, (0.3, 0.6, 0.9))
        values = list(result.values())

        assert values == sorted(values, reverse=True)

    def test_precision_at_recall_returns_a_value_per_target(self):
        rng = np.random.default_rng(1)
        y = rng.binomial(1, 0.1, 500)
        scores = rng.random(500)

        assert set(precision_at_recall(y, scores, (0.5, 0.9))) == {"r>=0.50", "r>=0.90"}

    def test_cost_weights_fraud_by_amount_not_by_count(self):
        """
        One $10,000 fraud missed must not cost the same as one $10 fraud
        missed. Counting cases treats a card test and a cash-out alike.
        """
        y = np.array([1, 1])
        scores = np.array([0.9, 0.1])  # catches the first, misses the second
        amounts = np.array([10.0, 10_000.0])

        cost = cost_analysis(y, scores, amounts, threshold=0.5)

        assert cost["fraud_prevented"] == pytest.approx(10.0)
        assert cost["fraud_missed"] == pytest.approx(10_000.0)

    def test_net_savings_can_be_negative(self):
        """
        A model that alerts on everything catches all fraud and still loses
        money. Reporting recall alone would hide that.
        """
        y = np.zeros(1_000, dtype=int)
        scores = np.ones(1_000)

        cost = cost_analysis(y, scores, np.ones(1_000), threshold=0.5, review_cost=3.0)

        assert cost["net_savings"] < 0

    def test_chosen_threshold_beats_the_extremes(self):
        rng = np.random.default_rng(2)
        y = rng.binomial(1, 0.02, 3_000)
        scores = np.clip(y * 0.5 + rng.random(3_000) * 0.5, 0, 1)
        amounts = rng.lognormal(4, 1, 3_000)

        chosen = choose_threshold(y, scores, amounts)
        best = cost_analysis(y, scores, amounts, chosen)["net_savings"]
        alert_on_everything = cost_analysis(y, scores, amounts, 0.0)["net_savings"]

        assert best >= alert_on_everything


class TestTrainingRun:
    def test_produces_a_report_naming_its_data_source(self, dataset):
        config = TrainingConfig(models=("logistic_regression",))

        result, models, _ = train(dataset=dataset, config=config)
        report = result.to_dict()

        assert report["data"]["source"] == DataSource.SYNTHETIC_CALIBRATED.value
        assert report["data"]["is_real_data"] is False
        assert report["config"]["split_strategy"] == "temporal-forward"
        assert set(models) == {"logistic_regression"}

    def test_reports_the_random_baseline_beside_the_score(self, dataset):
        """
        Average precision is meaningless without it: 0.8 is excellent against a
        0.0017 baseline and unremarkable against 0.5.

        The baseline is the *test split's* own positive rate, which is what a
        random scorer would achieve on exactly those rows. It drifts from the
        population rate on small samples, and using the population figure would
        misstate the floor the score is measured against.
        """
        config = TrainingConfig(models=("logistic_regression",))
        _, _, test_split = temporal_split(dataset, config)

        result, _, _ = train(dataset=dataset, config=config)

        assert result.baseline_average_precision == pytest.approx(float(test_split.labels.mean()))
        assert (
            result.reports["logistic_regression"].average_precision
            > result.baseline_average_precision
        )

    def test_beats_the_random_baseline_by_a_wide_margin(self, dataset):
        result, _, _ = train(
            dataset=dataset, config=TrainingConfig(models=("logistic_regression",))
        )
        report = result.reports["logistic_regression"]

        assert report.average_precision > 20 * result.baseline_average_precision

    def test_saves_models_scaler_feature_order_and_metrics(self, dataset, tmp_path):
        config = TrainingConfig(models=("logistic_regression",))
        result, models, scaler = train(dataset=dataset, config=config)

        save_artifacts(result, models, scaler, tmp_path)

        assert (tmp_path / "logistic_regression.joblib").is_file()
        assert (tmp_path / "scaler.joblib").is_file()
        assert (tmp_path / "feature_names.joblib").is_file()

        report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

        assert report["data"]["source"] == DataSource.SYNTHETIC_CALIBRATED.value
        assert report["best_model"] == "logistic_regression"

    def test_saved_feature_order_matches_the_training_order(self, dataset, tmp_path):
        """
        Serving rebuilds a row from a feature dict using this order. A mismatch
        would feed values into the wrong coefficients and score silently wrong.
        """
        import joblib

        config = TrainingConfig(models=("logistic_regression",))
        result, models, scaler = train(dataset=dataset, config=config)
        save_artifacts(result, models, scaler, tmp_path)

        assert joblib.load(tmp_path / "feature_names.joblib") == list(FEATURE_COLUMNS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
