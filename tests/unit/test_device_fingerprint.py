"""
Unit Tests for Device Fingerprinting Module

Tests cover:
- Attribute collection
- Fingerprint generation
- Collision detection
- Risk evaluation
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from src.fraud_detection.device_fingerprinting.attribute_collector import (
    DeviceAttributeCollector,
    NetworkContextCollector,
    BehavioralAttributeCollector,
    AttributeCollectionError,
)
from src.fraud_detection.device_fingerprinting.fingerprint_generator import (
    StableFingerprintGenerator,
)
from src.fraud_detection.device_fingerprinting.collision_detector import (
    CollisionDetector,
)
from src.fraud_detection.device_fingerprinting.risk_evaluator import (
    DeviceRiskEvaluator,
)
from src.fraud_detection.device_fingerprinting.visitor_fingerprint import (
    VisitorFingerprintEngine,
)
from src.block.models import DeviceAttributes, NetworkAttributes, BehavioralAttributes


class TestDeviceAttributeCollector:
    """Tests for DeviceAttributeCollector."""
    
    def test_collect_basic_attributes(self):
        """Test collection of basic device attributes."""
        collector = DeviceAttributeCollector()
        
        request_context = {
            "headers": {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
                "accept-language": "en-US,en;q=0.9",
            },
            "client_data": {
                "screen_resolution": "1920x1080",
                "timezone": "America/New_York",
                "cpu_cores": 8,
                "device_memory": 16.0,
            }
        }
        
        attrs = collector.collect(request_context)
        
        assert isinstance(attrs, DeviceAttributes)
        assert "Chrome" in attrs.user_agent or "Mozilla" in attrs.user_agent
        assert attrs.screen_resolution == "1920x1080"
        assert attrs.timezone == "America/New_York"
        assert attrs.cpu_cores == 8
        assert attrs.memory_gb == 16.0
    
    def test_collect_with_fingerprint_data(self):
        """Test collection including canvas and WebGL fingerprints."""
        collector = DeviceAttributeCollector()
        
        request_context = {
            "headers": {"user-agent": "Mozilla/5.0"},
            "client_data": {
                "canvas_fingerprint": "abc123canvasdata",
                "webgl_fingerprint": "xyz789webgldata",
                "audio_fingerprint": "audio_fp_data",
                "gpu_renderer": "NVIDIA GeForce RTX 3080",
                "fonts": ["Arial", "Helvetica", "Times"],
            }
        }
        
        attrs = collector.collect(request_context)
        
        assert attrs.canvas_fingerprint  # Should be hashed
        assert attrs.webgl_fingerprint
        assert attrs.audio_fingerprint
        assert attrs.gpu_renderer == "NVIDIA GeForce RTX 3080"
        assert attrs.fonts_hash  # Should be hashed
    
    def test_collect_handles_missing_data(self):
        """Test collection handles missing optional data gracefully."""
        collector = DeviceAttributeCollector()
        
        request_context = {
            "headers": {},
            "client_data": {}
        }
        
        attrs = collector.collect(request_context)
        
        assert isinstance(attrs, DeviceAttributes)
        assert attrs.user_agent == ""
        assert attrs.screen_resolution == ""
    
    def test_parse_user_agent_platform_detection(self):
        """Test platform detection from user agent."""
        collector = DeviceAttributeCollector()
        
        test_cases = [
            ("Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Windows"),
            ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15)", "macOS"),
            ("Mozilla/5.0 (Linux; Android 12)", "Android"),
            ("Mozilla/5.0 (iPhone; CPU iPhone OS 15_0)", "iOS"),
        ]
        
        for user_agent, expected_platform in test_cases:
            result = collector._parse_user_agent(user_agent)
            assert result["platform"] == expected_platform, f"Failed for: {user_agent}"


class TestNetworkContextCollector:
    """Tests for NetworkContextCollector."""
    
    def test_collect_basic_network_info(self):
        """Test collection of basic network attributes."""
        collector = NetworkContextCollector()
        
        request_context = {
            "headers": {},
            "client": ("192.168.1.100", 54321),
        }
        
        attrs = collector.collect(request_context)
        
        assert isinstance(attrs, NetworkAttributes)
        assert attrs.ip_address == "192.168.1.100"
        assert attrs.ip_version == "4"
    
    def test_collect_with_forwarded_header(self):
        """Test IP extraction from X-Forwarded-For header."""
        collector = NetworkContextCollector()
        
        request_context = {
            "headers": {
                "x-forwarded-for": "203.0.113.50, 192.168.1.1",
            },
            "client": ("10.0.0.1", 54321),
        }
        
        attrs = collector.collect(request_context)
        
        # Should use the first IP from X-Forwarded-For
        assert attrs.ip_address == "203.0.113.50"
    
    def test_detect_ipv6(self):
        """Test IPv6 detection."""
        collector = NetworkContextCollector()
        
        request_context = {
            "headers": {},
            "ip": "2001:db8::1",
        }
        
        attrs = collector.collect(request_context)
        
        assert attrs.ip_address == "2001:db8::1"
        assert attrs.ip_version == "6"


class TestBehavioralAttributeCollector:
    """Tests for BehavioralAttributeCollector."""
    
    def test_collect_behavioral_data(self):
        """Test collection of behavioral attributes."""
        collector = BehavioralAttributeCollector()
        
        request_context = {
            "behavioral_data": {
                "typing_speed": 65.0,
                "typing_cadence": [100, 120, 95, 110],
                "mouse_speed": 500.0,
                "mouse_acceleration": 2.5,
                "session_duration": 300,
                "pages_visited": 5,
            }
        }
        
        attrs = collector.collect(request_context)
        
        assert isinstance(attrs, BehavioralAttributes)
        assert attrs.typing_speed_wpm == 65.0
        assert len(attrs.typing_cadence) == 4
        assert attrs.mouse_speed == 500.0
        assert attrs.session_duration_seconds == 300
        assert attrs.pages_visited == 5
    
    def test_collect_handles_empty_data(self):
        """Test handling of empty behavioral data."""
        collector = BehavioralAttributeCollector()
        
        request_context = {}
        
        attrs = collector.collect(request_context)
        
        assert isinstance(attrs, BehavioralAttributes)
        assert attrs.typing_speed_wpm is None
        assert attrs.pages_visited == 0


class TestStableFingerprintGenerator:
    """Tests for StableFingerprintGenerator."""
    
    def test_generate_fingerprint(self):
        """Test fingerprint generation from attributes."""
        generator = StableFingerprintGenerator()
        
        device = DeviceAttributes(
            user_agent="Mozilla/5.0 Chrome/120",
            platform="Windows",
            screen_resolution="1920x1080",
            timezone="America/New_York",
            canvas_fingerprint="abc123",
            webgl_fingerprint="xyz789",
        )
        network = NetworkAttributes()
        behavioral = BehavioralAttributes()
        
        fp = generator.create(device, network, behavioral)
        
        assert fp.id  # Should have an ID
        assert len(fp.id) == 64  # SHA256 hex
        assert fp.stability_score > 0
        assert fp.hierarchy_level in ("coarse", "medium", "fine")
    
    def test_fingerprint_stability(self):
        """Test that same inputs produce same fingerprint."""
        generator = StableFingerprintGenerator()
        
        device = DeviceAttributes(
            user_agent="Mozilla/5.0 Chrome/120",
            platform="Windows",
            screen_resolution="1920x1080",
        )
        
        fp1 = generator.create(device, NetworkAttributes(), BehavioralAttributes())
        fp2 = generator.create(device, NetworkAttributes(), BehavioralAttributes())
        
        assert fp1.id == fp2.id
    
    def test_fingerprint_hierarchy_levels(self):
        """Test hierarchy levels based on attribute availability."""
        generator = StableFingerprintGenerator()
        
        # Minimal attributes -> coarse
        device_minimal = DeviceAttributes(platform="Windows")
        fp_minimal = generator.create(device_minimal, NetworkAttributes(), BehavioralAttributes())
        
        # Many attributes -> fine
        device_full = DeviceAttributes(
            platform="Windows",
            user_agent="Mozilla/5.0 Chrome/120",
            screen_resolution="1920x1080",
            timezone="America/New_York",
            cpu_cores=8,
            memory_gb=16,
            canvas_fingerprint="abc123",
            webgl_fingerprint="xyz789",
            audio_fingerprint="audio123",
        )
        fp_full = generator.create(device_full, NetworkAttributes(), BehavioralAttributes())
        
        assert fp_minimal.hierarchy_level in ("coarse", "medium")
        assert fp_full.hierarchy_level == "fine"


class TestCollisionDetector:
    """Tests for CollisionDetector."""
    
    def test_evaluate_low_collision_risk(self):
        """Test evaluation of low collision risk fingerprint."""
        from src.fraud_detection.device_fingerprinting.fingerprint_generator import RawFingerprint
        
        detector = CollisionDetector()
        
        fp = RawFingerprint(
            id="abcd1234" * 8,
            attributes={
                "device": {
                    "canvas_fingerprint": "abc",
                    "webgl_fingerprint": "xyz",
                    "fonts_hash": "fonts123",
                    "platform": "Windows",
                    "screen_resolution": "1920x1080",
                },
                "network": {},
            },
            stability_score=0.8,
            hierarchy_level="fine",
        )
        
        risk = detector.evaluate(fp)
        
        assert risk.probability < 0.3
        assert risk.risk_level == "low"
    
    def test_evaluate_high_collision_risk(self):
        """Test evaluation of high collision risk fingerprint."""
        from src.fraud_detection.device_fingerprinting.fingerprint_generator import RawFingerprint
        
        detector = CollisionDetector()
        
        fp = RawFingerprint(
            id="abcd1234" * 8,
            attributes={
                "device": {"platform": "iOS"},  # iOS is uniform
                "network": {},
            },
            stability_score=0.3,
            hierarchy_level="coarse",  # Coarse = high collision
        )
        
        risk = detector.evaluate(fp)
        
        assert risk.probability >= 0.5
        assert risk.risk_level in ("medium", "high")


class TestDeviceRiskEvaluator:
    """Tests for DeviceRiskEvaluator."""
    
    def test_score_low_risk_device(self):
        """Test scoring of low-risk device."""
        from src.fraud_detection.device_fingerprinting.fingerprint_generator import RawFingerprint
        
        evaluator = DeviceRiskEvaluator()
        
        fp = RawFingerprint(
            id="abcd1234" * 8,
            attributes={
                "device": {
                    "platform": "Windows",
                    "canvas_fingerprint": "abc",
                    "webgl_fingerprint": "xyz",
                },
                "network": {
                    "is_vpn": False,
                    "is_proxy": False,
                    "is_tor": False,
                    "threat_score": 0.0,
                },
                "behavioral": {
                    "typing_speed_wpm": 60,
                    "mouse_speed": 500,
                    "session_duration_seconds": 120,
                    "pages_visited": 3,
                },
            },
            stability_score=0.8,
            hierarchy_level="fine",
        )
        
        score = evaluator.score(fp)
        
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should be low risk
    
    def test_score_high_risk_vpn(self):
        """Test scoring of VPN user."""
        from src.fraud_detection.device_fingerprinting.fingerprint_generator import RawFingerprint
        
        evaluator = DeviceRiskEvaluator()
        
        fp = RawFingerprint(
            id="abcd1234" * 8,
            attributes={
                "device": {},
                "network": {
                    "is_vpn": True,
                    "is_tor": True,
                    "threat_score": 0.8,
                },
                "behavioral": {},
            },
            stability_score=0.3,
            hierarchy_level="coarse",
        )
        
        score = evaluator.score(fp)
        
        assert score > 0.5  # Should be high risk


class TestVisitorFingerprintEngine:
    """Integration tests for VisitorFingerprintEngine."""
    
    def test_generate_complete_fingerprint(self):
        """Test complete fingerprint generation flow."""
        engine = VisitorFingerprintEngine()
        
        request_context = {
            "headers": {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120",
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
        
        fingerprint = engine.generate_fingerprint(request_context)
        
        assert fingerprint.id
        assert 0.0 <= fingerprint.confidence <= 1.0
        assert 0.0 <= fingerprint.risk_score <= 1.0
        assert 0.0 <= fingerprint.collision_risk <= 1.0
        assert fingerprint.device_attrs
        assert fingerprint.network_attrs
    
    def test_generate_fingerprint_minimal_context(self):
        """Test fingerprint generation with minimal context."""
        engine = VisitorFingerprintEngine()
        
        request_context = {
            "headers": {"user-agent": "Mozilla/5.0"},
            "client": ("10.0.0.1", 12345),
        }
        
        fingerprint = engine.generate_fingerprint(request_context)
        
        # Should still generate a fingerprint, just with lower confidence
        assert fingerprint.id
        assert fingerprint.confidence < 0.8  # Lower confidence due to sparse data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
