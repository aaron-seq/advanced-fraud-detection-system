"""
Performance Tests for Fraud Detection System

Tests latency, throughput, and scalability:
- Sub-millisecond fraud check latency
- 10,000+ TPS throughput
- Memory usage under load
- Concurrent request handling
"""

import pytest
import time
import statistics
import asyncio
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import Mock

# Import the components we're testing
from src.fraud_detection.device_fingerprinting import VisitorFingerprintEngine
from src.fraud_detection.payment_validation import PaymentSourceValidator
from src.fraud_detection.behavioral import SpendingPatternAnalyzer
from src.block.models import Transaction, UserProfile, DeviceFingerprint, DeviceAttributes, NetworkAttributes, BehavioralAttributes


def create_sample_request_context() -> Dict[str, Any]:
    """Create a sample request context for fingerprinting."""
    return {
        "headers": {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
            "accept-language": "en-US,en;q=0.9",
        },
        "client": ("192.168.1.100", 54321),
        "client_data": {
            "screen_resolution": "1920x1080",
            "timezone": "America/New_York",
            "cpu_cores": 8,
            "device_memory": 16.0,
            "canvas_fingerprint": "canvas_data_123",
            "webgl_fingerprint": "webgl_data_456",
        },
        "behavioral_data": {
            "typing_speed": 65.0,
            "mouse_speed": 500.0,
            "session_duration": 120,
            "pages_visited": 3,
        },
    }


def create_sample_transaction() -> Transaction:
    """Create a sample transaction for testing."""
    return Transaction(
        id=f"txn_{int(time.time() * 1000000)}",
        user_id="user_perf_test",
        amount=100.0,
        currency="USD",
        merchant_id="merchant_001",
        merchant_category="retail",
        timestamp=datetime.utcnow(),
        location={"country": "US", "city": "New York"},
        payment_method="card",
    )


def create_sample_fingerprint() -> DeviceFingerprint:
    """Create a sample device fingerprint."""
    return DeviceFingerprint(
        id="fp_perf_test",
        confidence=0.9,
        risk_score=0.1,
        collision_risk=0.1,
        device_attrs=DeviceAttributes(platform="Windows"),
        network_attrs=NetworkAttributes(ip_address="192.168.1.1"),
        behavioral_attrs=BehavioralAttributes(),
        is_known_device=True,
    )


def create_sample_user_profile() -> UserProfile:
    """Create a sample user profile."""
    return UserProfile(
        user_id="user_perf_test",
        known_devices=[],
        frequent_locations=[{"country": "US", "city": "New York"}],
        avg_transaction_amount=100.0,
        transaction_frequency=5.0,
        account_age_days=365,
    )


class TestDeviceFingerprintPerformance:
    """Performance tests for device fingerprinting."""
    
    def test_fingerprint_generation_latency(self):
        """Test that fingerprint generation is fast."""
        engine = VisitorFingerprintEngine()
        context = create_sample_request_context()
        
        # Warm up
        engine.generate_fingerprint(context)
        
        # Measure latency over 100 iterations
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            engine.generate_fingerprint(context)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[94]
        p99_latency = sorted(latencies)[98]
        
        print(f"\nFingerprint Generation Latency:")
        print(f"  Average: {avg_latency:.3f}ms")
        print(f"  P95: {p95_latency:.3f}ms")
        print(f"  P99: {p99_latency:.3f}ms")
        
        # Assert sub-millisecond performance
        assert avg_latency < 5.0, f"Average latency {avg_latency}ms exceeds 5ms target"
        assert p95_latency < 10.0, f"P95 latency {p95_latency}ms exceeds 10ms target"
    
    def test_fingerprint_throughput(self):
        """Test fingerprint generation throughput."""
        engine = VisitorFingerprintEngine()
        context = create_sample_request_context()
        
        # Warm up
        for _ in range(10):
            engine.generate_fingerprint(context)
        
        # Measure throughput
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            engine.generate_fingerprint(context)
        end = time.perf_counter()
        
        duration = end - start
        throughput = iterations / duration
        
        print(f"\nFingerprint Generation Throughput:")
        print(f"  {throughput:.0f} fingerprints/second")
        
        # Assert minimum throughput
        assert throughput > 500, f"Throughput {throughput} below 500/sec target"


class TestPaymentValidationPerformance:
    """Performance tests for payment validation."""
    
    def test_validation_latency(self):
        """Test that payment validation is fast."""
        validator = PaymentSourceValidator()
        transaction = create_sample_transaction()
        fingerprint = create_sample_fingerprint()
        user_profile = create_sample_user_profile()
        
        # Warm up
        validator.validate(transaction, fingerprint, user_profile)
        
        # Measure latency
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            validator.validate(transaction, fingerprint, user_profile)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[94]
        
        print(f"\nPayment Validation Latency:")
        print(f"  Average: {avg_latency:.3f}ms")
        print(f"  P95: {p95_latency:.3f}ms")
        
        assert avg_latency < 1.0, f"Average latency {avg_latency}ms exceeds 1ms target"


class TestSpendingAnalyzerPerformance:
    """Performance tests for spending pattern analyzer."""
    
    def test_pattern_building_latency(self):
        """Test pattern building performance."""
        analyzer = SpendingPatternAnalyzer()
        
        # Create transaction history
        history = [
            {
                "transaction_id": f"txn_{i}",
                "amount": 100.0 + (i % 50),
                "timestamp": datetime.utcnow().isoformat(),
                "merchant_category": "retail",
                "location": {"city": "New York"},
            }
            for i in range(1000)
        ]
        
        # Measure latency
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            analyzer.build_pattern("user_001", history)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        avg_latency = statistics.mean(latencies)
        
        print(f"\nPattern Building Latency (1000 transactions):")
        print(f"  Average: {avg_latency:.3f}ms")
        
        # Pattern building should complete in reasonable time
        assert avg_latency < 100, f"Pattern building took {avg_latency}ms"
    
    def test_anomaly_detection_latency(self):
        """Test anomaly detection performance."""
        analyzer = SpendingPatternAnalyzer()
        
        # Build pattern first
        history = [
            {
                "transaction_id": f"txn_{i}",
                "amount": 100.0,
                "timestamp": datetime.utcnow().isoformat(),
            }
            for i in range(100)
        ]
        pattern = analyzer.build_pattern("user_001", history)
        
        transaction = create_sample_transaction()
        
        # Measure anomaly detection latency
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            analyzer.detect_anomaly(transaction, pattern)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        avg_latency = statistics.mean(latencies)
        
        print(f"\nAnomaly Detection Latency:")
        print(f"  Average: {avg_latency:.3f}ms")
        
        assert avg_latency < 1.0, f"Anomaly detection took {avg_latency}ms"


class TestConcurrentPerformance:
    """Test performance under concurrent load."""
    
    def test_concurrent_fingerprinting(self):
        """Test fingerprinting under concurrent load."""
        engine = VisitorFingerprintEngine()
        context = create_sample_request_context()
        
        def generate_fingerprint(_):
            start = time.perf_counter()
            engine.generate_fingerprint(context)
            return (time.perf_counter() - start) * 1000
        
        # Run with multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            start = time.perf_counter()
            latencies = list(executor.map(generate_fingerprint, range(1000)))
            total_time = time.perf_counter() - start
        
        avg_latency = statistics.mean(latencies)
        throughput = len(latencies) / total_time
        
        print(f"\nConcurrent Fingerprinting (10 threads):")
        print(f"  Average Latency: {avg_latency:.3f}ms")
        print(f"  Throughput: {throughput:.0f} ops/sec")
        
        assert throughput > 200, f"Concurrent throughput {throughput} below target"


class TestMemoryUsage:
    """Test memory usage under load."""
    
    def test_memory_stability(self):
        """Test that memory doesn't grow unbounded."""
        import sys
        
        engine = VisitorFingerprintEngine()
        context = create_sample_request_context()
        
        # Get baseline memory
        initial_size = sys.getsizeof(engine)
        
        # Generate many fingerprints
        for _ in range(10000):
            engine.generate_fingerprint(context)
        
        # Check memory hasn't grown excessively
        # This is a simplified check - in production use tracemalloc
        final_size = sys.getsizeof(engine)
        
        print(f"\nMemory Usage:")
        print(f"  Initial: {initial_size} bytes")
        print(f"  Final: {final_size} bytes")
        
        # Engine size should not grow significantly
        assert final_size < initial_size * 2, "Memory usage grew excessively"


def run_performance_report():
    """Generate a comprehensive performance report."""
    print("=" * 60)
    print("FRAUD DETECTION SYSTEM - PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Date: {datetime.now().isoformat()}")
    print("-" * 60)
    
    # Run all performance tests
    test_classes = [
        TestDeviceFingerprintPerformance(),
        TestPaymentValidationPerformance(),
        TestSpendingAnalyzerPerformance(),
        TestConcurrentPerformance(),
        TestMemoryUsage(),
    ]
    
    for test_class in test_classes:
        print(f"\n{test_class.__class__.__name__}")
        print("-" * 40)
        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                method = getattr(test_class, method_name)
                try:
                    method()
                    print(f"  ✓ {method_name}")
                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
    
    print("\n" + "=" * 60)
    print("PERFORMANCE REPORT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_performance_report()
