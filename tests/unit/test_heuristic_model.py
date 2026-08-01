"""
Unit tests for the heuristic baseline scorer.

The scorer exists to be honest about being a heuristic. These tests pin the two
properties that make it safe to serve: it only reads the standardised PCA
components, and it never claims a trained model's identity.
"""

import pytest

from app.services.fraud_detection_service import HEURISTIC_MODEL_NAME, HeuristicMLModel


@pytest.fixture
def model():
    return HeuristicMLModel()


def probability(model, features):
    return model.predict(features)[HEURISTIC_MODEL_NAME].fraud_probability


class TestFeatureSelection:
    def test_raw_scale_columns_do_not_saturate_the_score(self, model):
        """
        Amount and Time are orders of magnitude larger than the unit-variance
        PCA components. Averaging them in pins the score at 1.0 for every real
        transaction - a confident fraud call built from a scaling mistake.
        """
        features = {"V1": -1.36, "V2": 0.07, "V3": 2.53, "Amount": 149.62, "Time": 86400.0}

        assert probability(model, features) < 1.0

    def test_amount_does_not_change_the_score(self, model):
        pca_only = {"V1": -1.36, "V2": 0.07, "V3": 2.53}
        with_amount = {**pca_only, "Amount": 149.62, "Time": 86400.0}

        assert probability(model, pca_only) == probability(model, with_amount)

    def test_only_pca_components_are_reported_as_used(self, model):
        prediction = model.predict({"V1": 1.0, "Amount": 500.0, "Time": 10.0})

        assert prediction[HEURISTIC_MODEL_NAME].features_used == ["V1"]

    def test_v_prefixed_non_numeric_names_are_not_treated_as_components(self, model):
        """ "Velocity" starts with V but is not a PCA component."""
        prediction = model.predict({"Velocity": 900.0, "V2": 1.0})

        assert prediction[HEURISTIC_MODEL_NAME].features_used == ["V2"]


class TestScoring:
    def test_no_features_is_uncertain_not_safe(self, model):
        """Absence of evidence must not be scored as evidence of legitimacy."""
        prediction = model.predict({})[HEURISTIC_MODEL_NAME]

        assert prediction.fraud_probability == 0.5
        assert prediction.confidence == 0.0

    def test_no_pca_components_is_uncertain(self, model):
        assert probability(model, {"Amount": 149.62}) == 0.5

    def test_typical_values_score_low(self, model):
        assert probability(model, {f"V{i}": 0.3 for i in range(1, 29)}) < 0.1

    def test_extreme_values_score_high(self, model):
        assert probability(model, {f"V{i}": 12.0 for i in range(1, 29)}) == 1.0

    def test_score_increases_with_magnitude(self, model):
        low = probability(model, {"V1": 0.5, "V2": 0.5})
        high = probability(model, {"V1": 4.0, "V2": 4.0})

        assert low < high

    def test_sign_does_not_matter(self, model):
        assert probability(model, {"V1": -3.0}) == probability(model, {"V1": 3.0})

    @pytest.mark.parametrize(
        "features",
        [{}, {"V1": 0.0}, {"V1": 1e6}, {f"V{i}": -9.0 for i in range(1, 29)}],
    )
    def test_probability_stays_within_bounds(self, model, features):
        assert 0.0 <= probability(model, features) <= 1.0


class TestHonesty:
    def test_reports_itself_as_a_heuristic(self, model):
        """
        The name is the guardrail. A heuristic served under a trained model's
        name is how an unvalidated score ends up trusted as a measured one.
        """
        prediction = model.predict({"V1": 1.0})[HEURISTIC_MODEL_NAME]

        assert prediction.model_name == HEURISTIC_MODEL_NAME
        assert prediction.confidence <= 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
