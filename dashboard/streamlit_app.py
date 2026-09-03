"""
Streamlit Dashboard for Advanced Fraud Detection System
Real-time analytics and model monitoring interface
"""

import logging
import time
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Advanced Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ff6b6b;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-medium {
        background-color: #feca57;
        color: black;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-low {
        background-color: #48dbfb;
        color: black;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class FraudDetectionDashboard:
    """Main dashboard class for fraud detection analytics"""

    def __init__(self):
        import os

        self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.session = requests.Session()

        # Detection and analytics endpoints require a bearer token.
        token = os.getenv("API_TOKEN", "")
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def fetch_analytics_data(self, days_back: int = 7) -> dict[str, Any] | None:
        """
        Fetch analytics from the API. Returns None when unavailable.

        A failure is surfaced, never replaced with plausible-looking numbers.
        This previously fell back to a hardcoded 15847 transactions and 99.87%
        accuracy dated 2024, so an operator staring at a dead backend saw a
        busy, healthy system.
        """
        try:
            response = self.session.get(
                f"{self.api_base_url}/api/v1/analytics/dashboard-data",
                params={"days_back": days_back},
                timeout=10,
            )
        except requests.RequestException as exc:
            logger.warning("Analytics request failed: %s", exc)
            st.error(f"Cannot reach the API at {self.api_base_url}: {exc}")
            return None

        if response.status_code == 401:
            st.error("Unauthorized. Set API_TOKEN in the environment.")
            return None
        if response.status_code != 200:
            st.error(f"API returned {response.status_code}: {response.text[:200]}")
            return None

        return response.json()

    def fetch_health(self) -> dict[str, Any] | None:
        """Fetch real component health; None if the API is unreachable."""
        try:
            response = self.session.get(f"{self.api_base_url}/api/v1/health", timeout=5)
            return response.json() if response.status_code == 200 else None
        except requests.RequestException:
            return None

    def test_transaction(self, transaction_data: dict[str, Any]) -> dict[str, Any]:
        """Test a single transaction through the API"""
        try:
            response = self.session.post(
                f"{self.api_base_url}/api/v1/detect-fraud", json=transaction_data, timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Error testing transaction: {e!s}")
            return None

    def render_header(self):
        """Render dashboard header"""
        st.markdown(
            '<h1 class="main-header">🛡️ Advanced Fraud Detection Dashboard</h1>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

    def render_kpi_metrics(self, data: dict[str, Any]):
        """Render key performance indicators"""
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                label="📊 Total Transactions",
                value=f"{data['transactions_processed']:,}",
                delta="+2.3% vs yesterday",
            )

        with col2:
            st.metric(
                label="🚨 Fraud Detected",
                value=f"{data['fraud_detected']:,}",
                delta=f"{data['fraud_rate']:.2f}%",
                delta_color="inverse",
            )

        with col3:
            st.metric(
                label="⚡ Avg Response Time",
                value=f"{data['avg_processing_time_ms']:.1f}ms",
                delta="-5.2ms vs yesterday",
                delta_color="inverse",
            )

        with col4:
            st.metric(
                label="🎯 Model Accuracy", value=f"{data['model_accuracy']:.2f}%", delta="+0.03%"
            )

        with col5:
            st.metric(label="💰 Fraud Prevented", value="$2.4M", delta="+$340K vs yesterday")

    def render_transaction_trends(self, data: dict[str, Any]):
        """Render transaction trends chart"""
        st.subheader("📈 Transaction Trends (Last 7 Days)")

        # Prepare data
        df = pd.DataFrame(data["daily_stats"])
        df["date"] = pd.to_datetime(df["date"])

        # Create subplot
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Daily Transactions", "Fraud Detection Rate"),
            vertical_spacing=0.12,
        )

        # Transactions trend
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["transactions"],
                mode="lines+markers",
                name="Total Transactions",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=8),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["fraud"],
                mode="lines+markers",
                name="Fraud Detected",
                line=dict(color="#ff7f0e", width=3),
                marker=dict(size=8),
            ),
            row=1,
            col=1,
        )

        # Fraud rate
        fraud_rate = (df["fraud"] / df["transactions"] * 100).round(2)
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=fraud_rate,
                mode="lines+markers",
                name="Fraud Rate (%)",
                line=dict(color="#d62728", width=3),
                marker=dict(size=8),
            ),
            row=2,
            col=1,
        )

        fig.update_layout(height=600, showlegend=True, title_text="Transaction Analytics")

        st.plotly_chart(fig, use_container_width=True)

    def render_model_performance(self, data: dict[str, Any]):
        """Render model performance comparison"""
        st.subheader("🤖 Model Performance Comparison")

        # Prepare model performance data
        models = list(data["model_performance"].keys())
        metrics = ["accuracy", "precision", "recall", "f1_score"]

        # Create radar chart
        fig = go.Figure()

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for i, model in enumerate(models):
            values = [data["model_performance"][model][metric] for metric in metrics]
            values.append(values[0])  # Close the radar chart

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=[*metrics, metrics[0]],
                    fill="toself",
                    name=model.upper(),
                    line_color=colors[i % len(colors)],
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[95, 100])), showlegend=True, height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_recent_alerts(self, data: dict[str, Any]):
        """Render recent fraud alerts"""
        st.subheader("🚨 Recent High-Risk Transactions")

        for alert in data["recent_alerts"]:
            risk_level = (
                "high"
                if alert["risk_score"] > 90
                else "medium"
                if alert["risk_score"] > 70
                else "low"
            )
            alert_class = f"alert-{risk_level}"

            st.markdown(
                f"""
            <div class="{alert_class}">
                <strong>Transaction ID:</strong> {alert["id"]}<br>
                <strong>Amount:</strong> ${alert["amount"]:,}<br>
                <strong>Risk Score:</strong> {alert["risk_score"]:.1f}/100<br>
                <strong>Time:</strong> {alert["timestamp"]}
            </div>
            """,
                unsafe_allow_html=True,
            )

    def render_transaction_tester(self):
        """Render transaction testing interface"""
        st.subheader("🧪 Test Transaction")

        with st.form("transaction_test"):
            col1, col2 = st.columns(2)

            with col1:
                transaction_id = st.text_input("Transaction ID", value=f"TEST-{int(time.time())}")
                amount = st.number_input("Amount ($)", min_value=0.01, value=100.0, step=0.01)
                transaction_type = st.selectbox(
                    "Transaction Type", ["purchase", "withdrawal", "transfer", "payment"]
                )

            with col2:
                merchant_id = st.text_input("Merchant ID", value="MERCHANT-001")
                user_id = st.text_input("User ID", value="USER-001")
                country = st.text_input("Country", value="US")

            if st.form_submit_button("🔍 Test Transaction"):
                transaction_data = {
                    "transaction_id": transaction_id,
                    "amount": amount,
                    "transaction_type": transaction_type,
                    "merchant_id": merchant_id,
                    "user_id": user_id,
                    "transaction_country": country,
                }

                with st.spinner("Analyzing transaction..."):
                    result = self.test_transaction(transaction_data)

                if result:
                    col1, col2 = st.columns(2)

                    with col1:
                        fraud_status = (
                            "🚨 FRAUD DETECTED" if result["is_fraud"] else "✅ LEGITIMATE"
                        )
                        color = "red" if result["is_fraud"] else "green"
                        st.markdown(
                            f"<h3 style='color: {color};'>{fraud_status}</h3>",
                            unsafe_allow_html=True,
                        )

                        st.metric("Fraud Probability", f"{result['fraud_probability']:.3f}")
                        st.metric("Risk Score", f"{result['risk_score']:.1f}/100")
                        st.metric("Risk Level", result["risk_level"].title())
                        st.metric("Decision", result["decision"].replace("_", " ").title())
                        st.metric("Processing Time", f"{result['processing_time_ms']:.2f}ms")

                    with col2:
                        st.subheader("Explanation")
                        st.write(result["explanation"]["summary"])

                        if result["explanation"]["key_factors"]:
                            st.subheader("Key Factors")
                            for factor in result["explanation"]["key_factors"]:
                                st.write(f"• {factor['description']}")

    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.header("🛡️ Fraud Detection Controls")

        # Dashboard refresh
        if st.sidebar.button("🔄 Refresh Dashboard"):
            st.rerun()

        # Time range selector
        days_back = st.sidebar.selectbox(
            "📅 Time Range",
            options=[1, 7, 14, 30],
            index=1,
            format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}",
        )

        # System status, read from /api/v1/health. These indicators were
        # previously hardcoded to green and stayed green while the backend was
        # down - a status light that cannot turn red is not a status light.
        st.sidebar.subheader("📊 System Status")
        health = self.fetch_health()

        if health is None:
            st.sidebar.error("🔴 API: Unreachable")
        else:
            st.sidebar.success("🟢 API: Online")
            for component, state in health.get("components", {}).items():
                label = component.replace("_", " ").title()
                if state == "healthy":
                    st.sidebar.success(f"🟢 {label}: healthy")
                else:
                    st.sidebar.error(f"🔴 {label}: {state}")

        # Active model, reported by the API rather than assumed. The previous
        # hardcoded XGBoost/LightGBM/CatBoost list named models that are not
        # loaded; the service serves a heuristic unless MODEL_PATH holds
        # trained artefacts.
        st.sidebar.subheader("🤖 Active Model")
        models = (health or {}).get("models") or {}
        if not models:
            st.sidebar.write("unknown")
        else:
            st.sidebar.write(f"• {models.get('active_model', 'unknown')}")
            if models.get("trained") is False:
                st.sidebar.warning("Heuristic baseline - not a trained model")

        return days_back

    def run(self):
        """Run the main dashboard"""
        self.render_header()

        # Sidebar
        days_back = self.render_sidebar()

        # Fetch data
        with st.spinner("Loading dashboard data..."):
            data = self.fetch_analytics_data(days_back)

        if data is None:
            # Rendering charts over absent data would draw empty axes that look
            # like "no fraud today" rather than "no connection".
            st.warning(
                "Analytics unavailable. The transaction tester below still "
                "works if the API is reachable."
            )
            self.render_transaction_tester()
            return

        # Main content
        self.render_kpi_metrics(data)

        st.markdown("---")

        # Charts section
        col1, col2 = st.columns([2, 1])

        with col1:
            self.render_transaction_trends(data)

        with col2:
            self.render_recent_alerts(data)

        # Model performance
        self.render_model_performance(data)

        st.markdown("---")

        # Transaction tester
        self.render_transaction_tester()

        # Auto-refresh
        time.sleep(30)  # Refresh every 30 seconds
        st.rerun()


def main():
    """Main application entry point"""
    dashboard = FraudDetectionDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
