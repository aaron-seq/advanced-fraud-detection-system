"""
Integration tests for serving trained model artefacts.

Covers the seam between training and serving, where mistakes are silent: a
model fed unscaled features, or features in the wrong column order, returns a
confident probability rather than an error. Nothing downstream can tell the
difference.
"""

import numpy as np
import pytest

from app.services.fraud_detection_service import (
    HEURISTIC_MODEL_NAME,
    FraudDetectionService,
    HeuristicMLModel,
    JoblibEnsembleModel,
)
from src.training.dataset import FEATURE_COLUMNS, generate_calibrated
from src.training.pipeline import TrainingConfig, save_artifacts, train


@pytest.fixture(scope="module")
def artefacts(tmp_path_factory):
    """Train a small model once and persist it."""
    directory = tmp_path_factory.mktemp("models")
    dataset = generate_calibrated(rows=8_000, seed=11)
    config = TrainingConfig(models=("logistic_regression",))

    result, models, scaler = train(dataset=dataset, config=config)
    save_artifacts(result, models, scaler, directory)

    return directory


class TestArtefactLoading:
    def test_trained_artefacts_are_served_not_the_heuristic(self, artefacts):
        service = FraudDetectionService._load_model_service(str(artefacts))

        assert isinstance(service, JoblibEnsembleModel)

    def test_scaler_is_not_loaded_as_a_model(self, artefacts):
        """
        scaler.joblib sits beside the models. Loading it into the ensemble
        would call predict_proba on a StandardScaler.
        """
        service = FraudDetectionService._load_model_service(str(artefacts))

        assert "scaler" not in service._models
        assert "feature_names" not in service._models

    def test_missing_directory_falls_back_to_the_heuristic(self, tmp_path):
        service = FraudDetectionService._load_model_service(str(tmp_path / "absent"))

        assert isinstance(service, HeuristicMLModel)

    def test_models_without_a_scaler_are_refused(self, artefacts, tmp_path):
        """
        Serving unscaled rows to models fitted on scaled ones yields plausible,
        wrong probabilities. Refusing is the only safe response.
        """
        import shutil

        for name in ("logistic_regression.joblib", "feature_names.joblib"):
            shutil.copy(artefacts / name, tmp_path / name)

        service = FraudDetectionService._load_model_service(str(tmp_path))

        assert isinstance(service, HeuristicMLModel)

    def test_version_is_taken_from_the_training_report(self, artefacts):
        """A score should be traceable to the run that produced its model."""
        service = FraudDetectionService._load_model_service(str(artefacts))

        assert service._version != "unknown"


class TestScalingIsApplied:
    def test_scaler_transform_is_used(self, artefacts):
        service = FraudDetectionService._load_model_service(str(artefacts))
        features = dict.fromkeys(FEATURE_COLUMNS, 2.0)

        calls = []
        real_transform = service._scaler.transform
        service._scaler.transform = lambda row: calls.append(row) or real_transform(row)

        service.predict(features)

        assert calls, "predict() bypassed the scaler"

    def test_scores_match_calling_the_model_directly(self, artefacts):
        """
        The serving path must reproduce what the model produces when handed a
        correctly scaled, correctly ordered row.
        """
        service = FraudDetectionService._load_model_service(str(artefacts))
        features = dict.fromkeys(FEATURE_COLUMNS, 1.5)

        served = service.predict(features)["logistic_regression"].fraud_probability

        row = np.array([[features[name] for name in FEATURE_COLUMNS]])
        expected = service._models["logistic_regression"].predict_proba(
            service._scaler.transform(row)
        )[0][1]

        assert served == pytest.approx(expected)

    def test_feature_order_follows_the_saved_names(self, artefacts):
        """
        Rows are rebuilt from a dict. If order came from dict insertion rather
        than the saved names, values would land on the wrong coefficients.
        """
        service = FraudDetectionService._load_model_service(str(artefacts))

        forward = {name: float(i) for i, name in enumerate(FEATURE_COLUMNS)}
        reversed_insertion = dict(reversed(list(forward.items())))

        assert (
            service.predict(forward)["logistic_regression"].fraud_probability
            == service.predict(reversed_insertion)["logistic_regression"].fraud_probability
        )


class TestDiscrimination:
    def test_extreme_features_score_higher_than_typical_ones(self, artefacts):
        """
        End-to-end sanity: a trained model must separate the two. If the scaler
        were skipped, this ordering would not survive.
        """
        service = FraudDetectionService._load_model_service(str(artefacts))

        typical = service.predict(dict.fromkeys(FEATURE_COLUMNS, 0.05))
        extreme = service.predict(dict.fromkeys(FEATURE_COLUMNS, 3.5))

        assert (
            extreme["logistic_regression"].fraud_probability
            > typical["logistic_regression"].fraud_probability
        )

    def test_missing_features_default_to_zero_without_raising(self, artefacts):
        service = FraudDetectionService._load_model_service(str(artefacts))

        prediction = service.predict({"V1": 1.0})["logistic_regression"]

        assert 0.0 <= prediction.fraud_probability <= 1.0


class TestHeuristicIsDistinguishable:
    def test_heuristic_never_borrows_a_trained_model_name(self):
        prediction = HeuristicMLModel().predict({"V1": 1.0})

        assert set(prediction) == {HEURISTIC_MODEL_NAME}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
