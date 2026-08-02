"""
Tests for the Streamlit dashboard.

The dashboard had no tests, so two breakages shipped unnoticed and had to be
caught in review: a call to `self.api_client.fetch_health()` where no
`api_client` attribute exists, and a read of `result["confidence_level"]` after
that field was removed from the API response.

Streamlit's rendering cannot be exercised without a browser session, so instead
of asserting on pixels these check the two things that actually broke - that
the attributes the code calls on itself exist, and that the response fields it
reads are really in the API schema. Both are static checks, so they catch the
error without a running server.
"""

import ast
import inspect
import re
from pathlib import Path

import pytest

from app.models.transaction_models import TransactionResponse

# The dashboard is an optional extra (see setup.py). Skip rather than error
# when it is not installed; requirements-dev.txt pulls it in so CI runs these.
pytest.importorskip("streamlit", reason="dashboard extra not installed")
pytest.importorskip("plotly", reason="dashboard extra not installed")

SOURCE_PATH = Path(__file__).resolve().parents[2] / "dashboard" / "streamlit_app.py"
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dashboard_module():
    import dashboard.streamlit_app as module

    return module


@pytest.fixture
def dashboard(dashboard_module):
    return dashboard_module.FraudDetectionDashboard()


class TestSelfAttributesResolve:
    """
    Every `self.X` the class touches must exist on it.

    `self.api_client.fetch_health()` raised AttributeError on every page load:
    fetch_health is a method on this same class, and no api_client attribute
    was ever assigned. Nothing failed until a human opened the page.
    """

    def test_every_self_attribute_exists(self, dashboard):
        tree = ast.parse(SOURCE)
        cls = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "FraudDetectionDashboard"
        )

        referenced = {
            node.attr
            for node in ast.walk(cls)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        }

        missing = sorted(name for name in referenced if not hasattr(dashboard, name))

        assert not missing, f"FraudDetectionDashboard has no attribute: {missing}"


class TestResponseContract:
    """
    The dashboard may only read fields the API actually returns.

    `result["confidence_level"]` survived that field's removal from
    TransactionResponse and would have raised KeyError on the first successful
    test transaction.
    """

    def test_every_result_field_exists_in_the_response_model(self):
        read_fields = set(re.findall(r"""result\[["']([a-z_]+)["']\]""", SOURCE))

        assert read_fields, "expected the tester to read some response fields"

        unknown = sorted(read_fields - set(TransactionResponse.model_fields))

        assert not unknown, f"dashboard reads fields absent from TransactionResponse: {unknown}"


class TestFailureIsVisible:
    """A dead backend must look dead, not busy."""

    def test_analytics_returns_none_when_the_api_is_unreachable(self, dashboard, monkeypatch):
        import requests

        def refuse(*args, **kwargs):
            raise requests.ConnectionError("connection refused")

        monkeypatch.setattr(dashboard.session, "get", refuse)

        assert dashboard.fetch_analytics_data() is None

    def test_health_returns_none_when_the_api_is_unreachable(self, dashboard, monkeypatch):
        import requests

        def refuse(*args, **kwargs):
            raise requests.ConnectionError("connection refused")

        monkeypatch.setattr(dashboard.session, "get", refuse)

        assert dashboard.fetch_health() is None

    def test_no_fabricated_fallback_data_remains(self, dashboard_module):
        """
        The removed mock returned 15847 transactions and 99.87% accuracy on any
        API failure, so an operator watching a dead system saw a healthy one.

        Checked against literals in the parsed tree rather than raw text, so a
        number quoted in a comment or docstring explaining the old behaviour
        does not count as the behaviour returning.
        """
        literals = {
            node.value
            for node in ast.walk(ast.parse(SOURCE))
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
        }

        assert not hasattr(dashboard_module.FraudDetectionDashboard, "_get_mock_data")
        assert 15847 not in literals
        assert 99.87 not in literals

    def test_api_status_is_derived_from_a_health_call(self):
        """
        The indicators were hardcoded to green and stayed green while the API
        was unreachable. A status light that cannot turn red is not one.

        The literal "API: Online" still appears, but only inside the branch
        taken when a health response came back, so this asserts on the call
        rather than on the string.
        """
        tree = ast.parse(SOURCE)
        render_sidebar = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "render_sidebar"
        )

        called = {
            node.func.attr
            for node in ast.walk(render_sidebar)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

        assert "fetch_health" in called

    def test_unreachable_api_is_reported_as_unreachable(self, dashboard, monkeypatch):
        """The red path must exist and be reachable, not just be written down."""
        monkeypatch.setattr(dashboard, "fetch_health", lambda: None)

        rendered = []
        import streamlit as st

        monkeypatch.setattr(st.sidebar, "error", lambda msg, *a, **k: rendered.append(msg))
        monkeypatch.setattr(st.sidebar, "success", lambda msg, *a, **k: rendered.append(msg))
        monkeypatch.setattr(st.sidebar, "warning", lambda msg, *a, **k: rendered.append(msg))
        monkeypatch.setattr(st.sidebar, "subheader", lambda *a, **k: None)
        monkeypatch.setattr(st.sidebar, "write", lambda *a, **k: None)
        monkeypatch.setattr(st.sidebar, "title", lambda *a, **k: None)
        monkeypatch.setattr(st.sidebar, "selectbox", lambda *a, **k: 7)

        dashboard.render_sidebar()

        assert any("Unreachable" in str(message) for message in rendered)
        assert not any("API: Online" in str(message) for message in rendered)


class TestAuthentication:
    def test_bearer_token_is_sent_when_configured(self, dashboard_module, monkeypatch):
        monkeypatch.setenv("API_TOKEN", "a-test-token")

        dashboard = dashboard_module.FraudDetectionDashboard()

        assert dashboard.session.headers["Authorization"] == "Bearer a-test-token"

    def test_no_authorization_header_without_a_token(self, dashboard_module, monkeypatch):
        monkeypatch.delenv("API_TOKEN", raising=False)

        dashboard = dashboard_module.FraudDetectionDashboard()

        assert "Authorization" not in dashboard.session.headers


class TestDemoDashboardIsLabelled:
    """
    dashboard/app.py serves random numbers. It must never be mistakable for
    real monitoring.
    """

    def test_demo_app_declares_itself(self):
        import dashboard.app as demo

        assert "DEMO" in demo.app.title

    def test_demo_page_carries_a_banner(self):
        template = (SOURCE_PATH.parent / "templates" / "index.html").read_text(encoding="utf-8")

        assert "DEMO" in template

    def test_demo_module_says_the_data_is_synthetic(self):
        import dashboard.app as demo

        assert "synthetic" in inspect.getdoc(demo).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
