"""
Collision Detector

Detects potential fingerprint collisions where different devices
may have the same fingerprint ID.

Implements probabilistic matching with configurable thresholds.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from .fingerprint_generator import RawFingerprint
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CollisionRisk:
    """Collision risk assessment result."""
    probability: float  # 0.0 to 1.0
    risk_level: str  # "low", "medium", "high"
    factors: List[str]
    similar_fingerprint_count: int


class CollisionDetector:
    """
    Detects and scores collision risk for fingerprints.
    
    Collision occurs when two different devices produce
    the same fingerprint ID. This can lead to:
    - Legitimate users blocked (false positives)
    - Fraudsters evading detection (false negatives)
    
    Detection strategies:
    1. Attribute sparsity analysis
    2. Known collision pattern matching
    3. Statistical analysis of fingerprint distribution
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.collision_threshold = self.config.get("collision_threshold", 0.3)
        
        # Known high-collision patterns
        self.high_collision_patterns = [
            {"platform": "iOS", "browser": "safari"},  # iOS Safari is very uniform
            {"gpu_renderer": ""},  # Missing GPU info
            {"canvas_fingerprint": ""},  # Canvas blocking
        ]
    
    def evaluate(self, fingerprint: RawFingerprint) -> CollisionRisk:
        """
        Evaluate collision risk for a fingerprint.
        
        Args:
            fingerprint: Raw fingerprint to evaluate
            
        Returns:
            CollisionRisk with probability and factors
        """
        factors = []
        risk_score = 0.0
        
        # Check hierarchy level (coarse = more collisions)
        if fingerprint.hierarchy_level == "coarse":
            risk_score += 0.4
            factors.append("Using coarse fingerprint (few unique attributes)")
        elif fingerprint.hierarchy_level == "medium":
            risk_score += 0.2
            factors.append("Using medium fingerprint")
        
        # Check stability score (low stability = less reliable)
        if fingerprint.stability_score < 0.5:
            risk_score += 0.2
            factors.append("Low fingerprint stability")
        
        # Check for high-collision patterns
        device_attrs = fingerprint.attributes.get("device", {})
        pattern_risk = self._check_collision_patterns(device_attrs)
        if pattern_risk > 0:
            risk_score += pattern_risk
            factors.append("Matches known high-collision pattern")
        
        # Check attribute sparsity
        sparsity_risk = self._check_attribute_sparsity(fingerprint.attributes)
        if sparsity_risk > 0:
            risk_score += sparsity_risk
            factors.append("Sparse attribute collection")
        
        # Check for privacy-focused indicators
        privacy_risk = self._check_privacy_indicators(device_attrs)
        if privacy_risk > 0:
            risk_score += privacy_risk
            factors.append("Privacy-focused browser/device detected")
        
        # Normalize score
        collision_probability = min(risk_score, 1.0)
        
        # Determine risk level
        if collision_probability >= 0.6:
            risk_level = "high"
        elif collision_probability >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        logger.debug(
            "Collision risk evaluated",
            extra={
                "fingerprint_id": fingerprint.id[:16],
                "probability": collision_probability,
                "risk_level": risk_level,
                "factors": factors,
            }
        )
        
        return CollisionRisk(
            probability=collision_probability,
            risk_level=risk_level,
            factors=factors,
            similar_fingerprint_count=0,  # Would query database in production
        )
    
    def _check_collision_patterns(self, device_attrs: Dict[str, Any]) -> float:
        """Check if device matches known high-collision patterns."""
        risk = 0.0
        
        # iOS Safari pattern (highly uniform)
        if device_attrs.get("platform") == "iOS":
            risk += 0.15
        
        # Missing unique identifiers
        unique_attrs = ["canvas_fingerprint", "webgl_fingerprint", "fonts_hash"]
        missing_count = sum(1 for attr in unique_attrs if not device_attrs.get(attr))
        
        if missing_count >= 2:
            risk += 0.2
        elif missing_count == 1:
            risk += 0.1
        
        return risk
    
    def _check_attribute_sparsity(self, attributes: Dict[str, Any]) -> float:
        """Check for sparse attribute collection."""
        device_attrs = attributes.get("device", {})
        network_attrs = attributes.get("network", {})
        
        total_attrs = len(device_attrs) + len(network_attrs)
        
        if total_attrs < 5:
            return 0.3
        elif total_attrs < 10:
            return 0.15
        else:
            return 0.0
    
    def _check_privacy_indicators(self, device_attrs: Dict[str, Any]) -> float:
        """Check for privacy-focused browser indicators."""
        risk = 0.0
        
        user_agent = device_attrs.get("user_agent", "").lower()
        
        # Tor browser
        if "tor" in user_agent:
            risk += 0.3
        
        # Brave browser (may have fingerprint protection)
        if "brave" in user_agent:
            risk += 0.15
        
        # Firefox with privacy features
        if "firefox" in user_agent:
            # Check for privacy indicators
            if not device_attrs.get("canvas_fingerprint"):
                risk += 0.1  # Canvas blocking
        
        return risk
    
    def compare_fingerprints(
        self,
        fp1: RawFingerprint,
        fp2: RawFingerprint
    ) -> Tuple[float, List[str]]:
        """
        Compare two fingerprints for similarity.
        
        Returns:
            Tuple of (similarity_score, matching_attributes)
        """
        matching_attrs = []
        total_weight = 0.0
        matched_weight = 0.0
        
        # Compare hierarchy fingerprints
        hierarchy1 = fp1.attributes.get("fingerprint_hierarchy", {})
        hierarchy2 = fp2.attributes.get("fingerprint_hierarchy", {})
        
        # Fine match = likely same device
        if hierarchy1.get("fine") == hierarchy2.get("fine"):
            return 1.0, ["fine_fingerprint"]
        
        # Medium match = probably same device
        if hierarchy1.get("medium") == hierarchy2.get("medium"):
            matched_weight += 0.7
            matching_attrs.append("medium_fingerprint")
        
        # Coarse match = same device class
        if hierarchy1.get("coarse") == hierarchy2.get("coarse"):
            matched_weight += 0.3
            matching_attrs.append("coarse_fingerprint")
        
        # Compare device attributes
        device1 = fp1.attributes.get("device", {})
        device2 = fp2.attributes.get("device", {})
        
        weighted_attrs = [
            ("canvas_fingerprint", 0.15),
            ("webgl_fingerprint", 0.15),
            ("gpu_renderer", 0.1),
            ("fonts_hash", 0.1),
            ("screen_resolution", 0.1),
            ("timezone", 0.1),
            ("platform", 0.1),
        ]
        
        for attr, weight in weighted_attrs:
            total_weight += weight
            if device1.get(attr) and device1.get(attr) == device2.get(attr):
                matched_weight += weight
                matching_attrs.append(attr)
        
        similarity = matched_weight / (total_weight + 1.0)
        
        return similarity, matching_attrs
