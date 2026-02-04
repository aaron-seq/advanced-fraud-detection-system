"""
Locust Load Testing Configuration

Run with: locust -f tests/performance/locustfile.py --host=http://localhost:8000
"""

from locust import HttpUser, task, between
import random
import json


class FraudDetectionUser(HttpUser):
    """Simulates user behavior for load testing."""
    
    wait_time = between(0.1, 0.5)  # 100-500ms between requests
    
    @task(10)
    def check_transaction(self):
        """Simulate fraud check request."""
        transaction = {
            "transaction_id": f"txn_{random.randint(100000, 999999)}",
            "user_id": f"user_{random.randint(1000, 9999)}",
            "amount": round(random.uniform(10, 10000), 2),
            "currency": "USD",
            "merchant_id": f"merchant_{random.randint(100, 999)}",
            "merchant_category": random.choice([
                "retail", "grocery", "restaurant", "travel", "entertainment"
            ]),
            "payment_method": random.choice(["card", "bank_transfer", "wallet"]),
            "device_info": {
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120",
                "ip_address": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                "screen_resolution": "1920x1080",
            },
            "location": {
                "country": random.choice(["US", "UK", "CA", "AU", "IN"]),
                "city": random.choice(["New York", "London", "Toronto", "Sydney", "Mumbai"]),
            },
        }
        
        with self.client.post(
            "/api/v1/fraud/check",
            json=transaction,
            catch_response=True,
            name="Fraud Check"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(3)
    def batch_check(self):
        """Simulate batch fraud check."""
        transactions = [
            {
                "transaction_id": f"batch_txn_{i}",
                "user_id": f"user_{random.randint(1000, 9999)}",
                "amount": round(random.uniform(10, 5000), 2),
            }
            for i in range(10)
        ]
        
        with self.client.post(
            "/api/v1/fraud/batch",
            json={"transactions": transactions},
            catch_response=True,
            name="Batch Check"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(2)
    def get_analytics(self):
        """Fetch analytics data."""
        self.client.get("/api/v1/analytics/summary", name="Analytics Summary")
    
    @task(1)
    def health_check(self):
        """Health check endpoint."""
        self.client.get("/health", name="Health Check")


class HighLoadUser(HttpUser):
    """Aggressive user for stress testing."""
    
    wait_time = between(0.01, 0.05)  # Very fast requests
    
    @task
    def rapid_fraud_check(self):
        """Rapid-fire fraud checks."""
        transaction = {
            "transaction_id": f"stress_{random.randint(0, 999999999)}",
            "user_id": "stress_user",
            "amount": random.uniform(1, 100),
        }
        
        self.client.post(
            "/api/v1/fraud/check",
            json=transaction,
            name="Stress Fraud Check"
        )
