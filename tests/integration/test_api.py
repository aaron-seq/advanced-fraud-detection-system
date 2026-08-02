"""
Integration tests for the FastAPI layer.

Covers authentication, input validation boundaries and the detection endpoints
end to end against the real detection block.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.main import app

# TrustedHostMiddleware rejects TestClient's default "testserver" host, which
# is the intended behaviour, so requests are made against an allowed host.
BASE_URL = "http://localhost"

VALID_TRANSACTION = {
    "transaction_id": "txn_test_1",
    "amount": 250.0,
    "transaction_type": "purchase",
    "transaction_country": "US",
    "transaction_city": "New York",
    "features": {"V1": -1.36, "V2": 0.07, "V3": 2.53, "Amount": 149.62},
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url=BASE_URL) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def auth_headers():
    return {"Authorization": f"Bearer {create_access_token('user_test')}"}


class TestHealth:
    def test_root_is_public(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert response.json()["status"] == "operational"

    def test_detailed_health_reports_each_component(self, client):
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] in ("healthy", "degraded")
        assert set(body["components"]) == {"database", "cache", "ml_models"}

    def test_health_reports_whether_the_model_is_trained(self, client):
        """
        A heuristic scorer must never be reported as a trained model; that
        distinction is the difference between a measured score and a guess.
        """
        body = client.get("/api/v1/health").json()

        assert "trained" in body["models"]
        assert isinstance(body["models"]["trained"], bool)


class TestAuthentication:
    def test_detection_requires_a_token(self, client):
        response = client.post("/api/v1/detect-fraud", json=VALID_TRANSACTION)

        assert response.status_code == 401

    def test_batch_requires_a_token(self, client):
        response = client.post(
            "/api/v1/detect-fraud/batch",
            json={"batch_id": "b1", "transactions": [VALID_TRANSACTION]},
        )

        assert response.status_code == 401

    def test_malformed_token_is_rejected(self, client):
        response = client.post(
            "/api/v1/detect-fraud",
            json=VALID_TRANSACTION,
            headers={"Authorization": "Bearer not-a-jwt"},
        )

        assert response.status_code == 401

    def test_token_signed_with_another_key_is_rejected(self, client):
        import jwt

        forged = jwt.encode(
            {"sub": "attacker"}, "a-different-key-of-sufficient-length-32b", algorithm="HS256"
        )
        response = client.post(
            "/api/v1/detect-fraud",
            json=VALID_TRANSACTION,
            headers={"Authorization": f"Bearer {forged}"},
        )

        assert response.status_code == 401


class TestFraudDetection:
    def test_returns_a_decision(self, client, auth_headers):
        response = client.post("/api/v1/detect-fraud", json=VALID_TRANSACTION, headers=auth_headers)

        assert response.status_code == 200
        body = response.json()
        assert body["transaction_id"] == VALID_TRANSACTION["transaction_id"]
        assert body["decision"] in (
            "approve",
            "deny",
            "review",
            "additional_auth_required",
        )
        assert 0.0 <= body["risk_score"] <= 100.0
        assert 0.0 <= body["fraud_probability"] <= 1.0

    def test_batch_aggregates_results(self, client, auth_headers):
        second = {**VALID_TRANSACTION, "transaction_id": "txn_test_2", "amount": 90000.0}
        response = client.post(
            "/api/v1/detect-fraud/batch",
            json={"batch_id": "batch_1", "transactions": [VALID_TRANSACTION, second]},
            headers=auth_headers,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["total_transactions"] == 2
        assert len(body["results"]) == 2
        assert [r["transaction_id"] for r in body["results"]] == [
            "txn_test_1",
            "txn_test_2",
        ]


class TestInputValidation:
    @pytest.mark.parametrize(
        "override",
        [
            {"amount": 0},
            {"amount": -1},
            {"amount": 99_999_999},
            {"currency": "DOLLARS"},
            {"transaction_type": "not-a-type"},
            {"transaction_id": ""},
        ],
    )
    def test_invalid_fields_are_rejected(self, client, auth_headers, override):
        response = client.post(
            "/api/v1/detect-fraud",
            json={**VALID_TRANSACTION, **override},
            headers=auth_headers,
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_features_give_a_422_not_a_500(self, client, auth_headers, literal):
        """
        NaN must be rejected, and the rejection must render. FastAPI echoes the
        offending input into the 422 body, and a non-finite float there breaks
        JSON encoding - turning the validation error into a 500.
        """
        body = (
            '{"transaction_id":"t","amount":1.0,"transaction_type":"purchase",'
            f'"features":{{"V1":{literal}}}}}'
        )
        response = client.post(
            "/api/v1/detect-fraud",
            content=body,
            headers={**auth_headers, "Content-Type": "application/json"},
        )

        assert response.status_code == 422
        assert "finite" in json.dumps(response.json())

    @pytest.mark.parametrize("size", [0, 1001])
    def test_batch_size_bounds_are_enforced(self, client, auth_headers, size):
        response = client.post(
            "/api/v1/detect-fraud/batch",
            json={"batch_id": "b", "transactions": [VALID_TRANSACTION] * size},
            headers=auth_headers,
        )

        assert response.status_code == 422


class TestAnalytics:
    """
    The analytics endpoint must report what this process actually recorded.
    Its predecessor returned a hardcoded 15847 transactions and 99.87% accuracy
    with dates from 2024 - numbers no run had ever produced.
    """

    def test_requires_authentication(self, client):
        assert client.get("/api/v1/analytics/dashboard-data").status_code == 401

    def test_counts_reflect_recorded_decisions(self, client, auth_headers):
        before = client.get("/api/v1/analytics/dashboard-data", headers=auth_headers).json()
        start = before["fraud_prevention"]["transactions"]["total"]

        for index in range(3):
            client.post(
                "/api/v1/detect-fraud",
                json={**VALID_TRANSACTION, "transaction_id": f"analytics_{index}"},
                headers=auth_headers,
            )

        after = client.get("/api/v1/analytics/dashboard-data", headers=auth_headers).json()

        assert after["fraud_prevention"]["transactions"]["total"] == start + 3

    def test_batch_decisions_are_recorded(self, client, auth_headers):
        before = client.get("/api/v1/analytics/dashboard-data", headers=auth_headers).json()
        start = before["fraud_prevention"]["transactions"]["total"]

        client.post(
            "/api/v1/detect-fraud/batch",
            json={
                "batch_id": "analytics_batch",
                "transactions": [
                    {**VALID_TRANSACTION, "transaction_id": f"ab_{i}"} for i in range(4)
                ],
            },
            headers=auth_headers,
        )

        after = client.get("/api/v1/analytics/dashboard-data", headers=auth_headers).json()

        assert after["fraud_prevention"]["transactions"]["total"] == start + 4

    def test_reports_amounts_actually_processed(self, client, auth_headers):
        before = client.get("/api/v1/analytics/dashboard-data", headers=auth_headers).json()
        start = before["fraud_prevention"]["financial_impact"]["total_processed"]

        client.post(
            "/api/v1/detect-fraud",
            json={**VALID_TRANSACTION, "transaction_id": "amount_check", "amount": 777.0},
            headers=auth_headers,
        )

        after = client.get("/api/v1/analytics/dashboard-data", headers=auth_headers).json()

        assert after["fraud_prevention"]["financial_impact"]["total_processed"] == pytest.approx(
            start + 777.0
        )

    def test_declares_that_counters_are_not_persisted(self, client, auth_headers):
        """The caller must not mistake per-process counters for durable history."""
        body = client.get("/api/v1/analytics/dashboard-data", headers=auth_headers).json()

        assert "not persisted" in body["source"]

    def test_names_the_model_that_produced_the_numbers(self, client, auth_headers):
        client.post("/api/v1/detect-fraud", json=VALID_TRANSACTION, headers=auth_headers)

        body = client.get("/api/v1/analytics/dashboard-data", headers=auth_headers).json()

        assert body["model"]["active_model"] in body["model_performance"]

    @pytest.mark.parametrize("days_back", [0, -1, 366])
    def test_rejects_out_of_range_windows(self, client, auth_headers, days_back):
        response = client.get(
            f"/api/v1/analytics/dashboard-data?days_back={days_back}", headers=auth_headers
        )

        assert response.status_code == 422


class TestSecurityHeaders:
    def test_untrusted_host_is_rejected(self):
        with TestClient(app, base_url="http://evil.example.com") as evil_client:
            assert evil_client.get("/").status_code == 400

    def test_process_time_header_is_present(self, client):
        assert "X-Process-Time" in client.get("/").headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
