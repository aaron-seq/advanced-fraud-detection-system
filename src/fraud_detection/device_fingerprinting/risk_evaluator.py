"""
Device Risk Evaluator

Evaluates risk score for a device fingerprint based on:
- Device reputation
- Behavioral signals
- Network context
- Historical patterns
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

from .fingerprint_generator import RawFingerprint
from ...block.models import NetworkAttributes
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DeviceRiskFactors:
    """Individual risk factors contributing to score."""
    network_risk: float
    device_reputation: float
    behavioral_risk: float
    anomaly_score: float
    
    @property
    def total(self) -> float:
        """Calculate total risk score (0.0 to 1.0)."""
        weights = {
            "network": 0.30,
            "device": 0.25,
            "behavioral": 0.25,
            "anomaly": 0.20,
        }
        
        total = (
            self.network_risk * weights["network"] +
            self.device_reputation * weights["device"] +
            self.behavioral_risk * weights["behavioral"] +
            self.anomaly_score * weights["anomaly"]
        )
        
        return min(max(total, 0.0), 1.0)


class DeviceRiskEvaluator:
    """
    Evaluates fraud risk based on device fingerprint.
    
    Risk factors:
    1. Network risk (VPN, proxy, Tor, datacenter IPs)
    2. Device reputation (known fraudulent devices)
    3. Behavioral anomalies (unusual patterns)
    4. Device-account linkage patterns
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Risk thresholds
        self.high_risk_threshold = self.config.get("high_risk_threshold", 0.7)
        self.medium_risk_threshold = self.config.get("medium_risk_threshold", 0.4)
        
        # In production, would connect to device reputation database
        self.reputation_service = None
    
    def score(self, fingerprint: RawFingerprint) -> float:
        """
        Calculate overall risk score for device fingerprint.
        
        Args:
            fingerprint: Device fingerprint to evaluate
            
        Returns:
            Risk score from 0.0 (safe) to 1.0 (high risk)
        """
        # Calculate individual risk factors
        network_risk = self._evaluate_network_risk(fingerprint)
        device_reputation = self._evaluate_device_reputation(fingerprint)
        behavioral_risk = self._evaluate_behavioral_risk(fingerprint)
        anomaly_score = self._evaluate_anomalies(fingerprint)
        
        factors = DeviceRiskFactors(
            network_risk=network_risk,
            device_reputation=device_reputation,
            behavioral_risk=behavioral_risk,
            anomaly_score=anomaly_score,
        )
        
        total_risk = factors.total
        
        logger.info(
            "Device risk evaluated",
            extra={
                "fingerprint_id": fingerprint.id[:16],
                "total_risk": total_risk,
                "network_risk": network_risk,
                "device_reputation": device_reputation,
                "behavioral_risk": behavioral_risk,
                "anomaly_score": anomaly_score,
            }
        )
        
        return total_risk
    
    def _evaluate_network_risk(self, fingerprint: RawFingerprint) -> float:
        """
        Evaluate risk from network context.
        
        High risk indicators:
        - VPN/Proxy/Tor usage
        - Datacenter IPs
        - High threat score from reputation service
        - Geolocation mismatches
        """
        risk = 0.0
        network_attrs = fingerprint.attributes.get("network", {})
        
        # VPN detection
        if network_attrs.get("is_vpn"):
            risk += 0.4
        
        # Proxy detection
        if network_attrs.get("is_proxy"):
            risk += 0.3
        
        # Tor detection (highest risk)
        if network_attrs.get("is_tor"):
            risk += 0.6
        
        # Datacenter IP
        if network_attrs.get("is_datacenter"):
            risk += 0.5
        
        # Threat score from IP reputation
        threat_score = network_attrs.get("threat_score", 0.0)
        risk += threat_score * 0.4
        
        return min(risk, 1.0)
    
    def _evaluate_device_reputation(self, fingerprint: RawFingerprint) -> float:
        """
        Evaluate device reputation based on historical data.
        
        In production, would query device reputation database to check:
        - Previous fraud associations
        - Account linkage patterns
        - Velocity of new accounts
        """
        risk = 0.0
        
        # Check fingerprint stability (low stability = suspicious)
        if fingerprint.stability_score < 0.3:
            risk += 0.3
        elif fingerprint.stability_score < 0.5:
            risk += 0.15
        
        # Check hierarchy level (coarse = less reliable)
        if fingerprint.hierarchy_level == "coarse":
            risk += 0.2
        
        # In production: Query device reputation database
        # reputation = self.reputation_service.get(fingerprint.id)
        # if reputation and reputation.is_fraudulent:
        #     risk += 0.8
        
        return min(risk, 1.0)
    
    def _evaluate_behavioral_risk(self, fingerprint: RawFingerprint) -> float:
        """
        Evaluate risk from behavioral signals.
        
        Suspicious indicators:
        - Automated behavior patterns
        - Inconsistent typing/mouse patterns
        - Abnormal session behavior
        """
        risk = 0.0
        behavioral_attrs = fingerprint.attributes.get("behavioral", {})
        
        # Check for missing behavioral data (could be bot)
        if not behavioral_attrs:
            risk += 0.3
        else:
            # Abnormal typing speed (too fast = automated)
            typing_speed = behavioral_attrs.get("typing_speed_wpm")
            if typing_speed and typing_speed > 150:
                risk += 0.4  # Likely automated
            elif typing_speed and typing_speed > 100:
                risk += 0.1  # Suspicious
            
            # No mouse movement (could be automated)
            if not behavioral_attrs.get("mouse_speed"):
                risk += 0.2
            
            # Very short session with immediate action
            session_duration = behavioral_attrs.get("session_duration_seconds", 0)
            pages_visited = behavioral_attrs.get("pages_visited", 0)
            
            if session_duration < 5 and pages_visited <= 1:
                risk += 0.3  # Suspiciously fast
        
        return min(risk, 1.0)
    
    def _evaluate_anomalies(self, fingerprint: RawFingerprint) -> float:
        """
        Evaluate general anomalies in the fingerprint.
        
        Checks for:
        - Attribute inconsistencies
        - Known evasion techniques
        - Unusual configurations
        """
        risk = 0.0
        device_attrs = fingerprint.attributes.get("device", {})
        
        # Check for inconsistencies
        platform = device_attrs.get("platform", "")
        user_agent = device_attrs.get("user_agent", "")
        
        # Platform/User-Agent mismatch
        if platform and user_agent:
            if platform.lower() == "ios" and "android" in user_agent.lower():
                risk += 0.5  # Clear mismatch
            elif platform.lower() == "android" and "iphone" in user_agent.lower():
                risk += 0.5
        
        # Missing canvas fingerprint (could be blocking/evasion)
        if not device_attrs.get("canvas_fingerprint"):
            risk += 0.15
        
        # Missing WebGL fingerprint
        if not device_attrs.get("webgl_fingerprint"):
            risk += 0.15
        
        # Check for known spoofing indicators
        if self._detect_spoofing_indicators(device_attrs):
            risk += 0.4
        
        return min(risk, 1.0)
    
    def _detect_spoofing_indicators(self, device_attrs: Dict[str, Any]) -> bool:
        """Detect signs of fingerprint spoofing."""
        # Check for common spoofing tool signatures
        user_agent = device_attrs.get("user_agent", "").lower()
        
        spoofing_indicators = [
            "selenium",
            "webdriver",
            "headless",
            "phantomjs",
            "puppeteer",
        ]
        
        for indicator in spoofing_indicators:
            if indicator in user_agent:
                return True
        
        # Check for impossible configurations
        cpu_cores = device_attrs.get("cpu_cores")
        memory_gb = device_attrs.get("memory_gb")
        
        if cpu_cores and memory_gb:
            # Very high cores with very low memory is suspicious
            if cpu_cores > 16 and memory_gb < 1:
                return True
        
        return False
