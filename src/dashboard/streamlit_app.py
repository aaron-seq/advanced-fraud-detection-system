# src/dashboard/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

API_URL = "http://localhost:8000"

@st.cache_data(ttl=60)
def load_data():
    """
    Loads data from the API and caches it for performance.
    """
    try:
        response = requests.get(f"{API_URL}/api/v1/analytics/dashboard-data")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not load data from API: {e}")
        return None

def render_kpi_metrics(data):
    """
    Renders the Key Performance Indicators (KPIs) for the dashboard.
    """
    cols = st.columns(4)
    cols[0].metric("Total Transactions", f"{data['transactions_processed']:,}")
    cols[1].metric("Fraud Detected", f"{data['fraud_detected']:,}", f"{data['fraud_rate']:.2f}%")
    cols[2].metric("Avg. Processing Time", f"{data['avg_processing_time_ms']:.1f}ms")
    cols[3].metric("Model Accuracy", f"{data['model_accuracy']:.2f}%")

def render_transaction_chart(data):
    """
    Renders a chart showing transaction trends over time.
    """
    df = pd.DataFrame(data['daily_stats'])
    df['date'] = pd.to_datetime(df['date'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['transactions'], mode='lines+markers', name='Total Transactions'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['fraud'], mode='lines+markers', name='Fraud Detected'))

    st.plotly_chart(fig, use_container_width=True)

def main():
    """
    Main function to render the Streamlit dashboard.
    """
    st.title("Advanced Fraud Detection Dashboard")

    data = load_data()

    if data:
        render_kpi_metrics(data)
        render_transaction_chart(data)
    else:
        st.warning("No data available to display.")

if __name__ == "__main__":
    main()
