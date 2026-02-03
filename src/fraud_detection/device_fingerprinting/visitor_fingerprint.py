"""
Visitor Fingerprint Engine

Main entry point for device fingerprinting.
Combines all collectors and analyzers into a unified engine.

Implements Sardine-style Visitor Fingerprint methodology:
- 2000+ device attributes
- Network context integration
- 95%+ collision reduction
"""

from typing import Dict, Any, Optional
from datetime import datetime
import logging

from ...block.models import DeviceFingerprint, DeviceAttributes, NetworkAttributes, BehavioralAttributes
from ...utils.error_handling import DeviceFingerprintError, handle_error
from ...utils.logging_utils import get_logger, log_with_context

from .attribute_collector import (
    DeviceAttributeCollector,
    NetworkContextCollector,
    BehavioralAttributeCollector,
    AttributeCollectionError,
)
from .fingerprint_generator import StableFingerprintGenerator
from .collision_detector import CollisionDetector
from .risk_evaluator import DeviceRiskEvaluator

logger = get_logger(__name__)


class VisitorFingerprintEngine:
    """
    Advanced device fingerprinting with network context.
    
    Based on 2026 industry standards for fraud detection.
    
    Features:
    - Collects 2000+ device attributes
    - Hierarchical fingerprinting for stability
    - Network context integration (IP, geolocation, VPN detection)
    - Behavioral signals (typing, mouse, navigation)
    - 95%+ collision reduction target
    - 97%+ division reduction target
    
    Usage:
        engine = VisitorFingerprintEngine(config)
        fingerprint = engine.generate_fingerprint(request_context)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the fingerprint engine.
        
        Args:
            config: Configuration options:
                - attribute_depth: "basic", "standard", "comprehensive"
                - confidence_threshold: Minimum confidence (0.0 to 1.0)
                - network_context: Enable network context (default: True)
        """
        self.config = config or {}
        self.attribute_depth = self.config.get("attribute_depth", "comprehensive")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.85)
        self.enable_network_context = self.config.get("network_context", True)
        
        # Initialize collectors
        self.device_collector = DeviceAttributeCollector(self.config)
        self.network_collector = NetworkContextCollector(self.config)
        self.behavioral_collector = BehavioralAttributeCollector(self.config)
        
        # Initialize analyzers
        self.fingerprint_generator = StableFingerprintGenerator(self.config)
        self.collision_detector = CollisionDetector(self.config)
        self.risk_evaluator = DeviceRiskEvaluator(self.config)
        
        logger.info(
            "VisitorFingerprintEngine initialized",
            extra={
                "attribute_depth": self.attribute_depth,
                "confidence_threshold": self.confidence_threshold,
                "network_context_enabled": self.enable_network_context,
            }
        )
    
    def generate_fingerprint(
        self,
        request_context: Dict[str, Any]
    ) -> DeviceFingerprint:
        """
        Generate comprehensive device fingerprint with confidence scoring.
        
        Args:
            request_context: HTTP request metadata including:
                - headers: HTTP headers dict
                - ip: Client IP address
                - client: Client connection info (host, port)
                - client_data: Additional client-collected data
                - behavioral_data: Behavioral signals
                
        Returns:
            DeviceFingerprint object with ID, confidence score, risk level
            
        Raises:
            DeviceFingerprintError: If critical attributes cannot be collected
        """
        request_id = request_context.get("request_id")
        
        try:
            # Step 1: Collect device attributes
            device_attrs = self.device_collector.collect(request_context)
            
            # Step 2: Collect network attributes
            if self.enable_network_context:
                network_attrs = self.network_collector.collect(request_context)
            else:
                network_attrs = NetworkAttributes()
            
            # Step 3: Collect behavioral attributes
            behavioral_attrs = self.behavioral_collector.collect(request_context)
            
            # Step 4: Generate stable fingerprint
            raw_fingerprint = self.fingerprint_generator.create(
                device=device_attrs,
                network=network_attrs,
                behavioral=behavioral_attrs
            )
            
            # Step 5: Calculate confidence score
            confidence = self._calculate_confidence(raw_fingerprint)
            
            # Step 6: Detect potential collisions
            collision_risk = self.collision_detector.evaluate(raw_fingerprint)
            
            # Step 7: Calculate risk score
            risk_score = self.risk_evaluator.score(raw_fingerprint)
            
            # Step 8: Build final fingerprint object
            fingerprint = DeviceFingerprint(
                id=raw_fingerprint.id,
                confidence=confidence,
                risk_score=risk_score,
                collision_risk=collision_risk.probability,
                device_attrs=device_attrs,
                network_attrs=network_attrs,
                behavioral_attrs=behavioral_attrs,
                is_known_device=False,  # Would check against user's known devices
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                raw_attributes=raw_fingerprint.attributes,
            )
            
            # Log success with full context
            logger.info(
                "Device fingerprint generated successfully",
                extra={
                    "fingerprint_id": fingerprint.id[:16] + "...",
                    "confidence": confidence,
                    "risk_score": risk_score,
                    "collision_risk": collision_risk.probability,
                    "attribute_count": fingerprint.attribute_count,
                    "hierarchy_level": raw_fingerprint.hierarchy_level,
                    "request_id": request_id,
                }
            )
            
            return fingerprint
            
        except AttributeCollectionError as e:
            # Edge case: partial attribute collection
            logger.warning(
                "Partial fingerprint generated due to missing attributes",
                extra={
                    "error": str(e),
                    "collected_attributes": list(e.partial_attributes.keys()),
                    "request_id": request_id,
                }
            )
            
            # Generate fallback fingerprint with lower confidence
            return self._generate_fallback_fingerprint(
                e.partial_attributes,
                request_id
            )
            
        except Exception as e:
            # Critical error - log with full context
            handle_error(
                e,
                context={
                    "request_id": request_id,
                    "operation": "generate_fingerprint",
                },
                reraise=False
            )
            
            # Re-raise as domain-specific exception
            raise DeviceFingerprintError(
                "Failed to generate device fingerprint",
                context={"request_id": request_id}
            ) from e
    
    def _calculate_confidence(self, raw_fingerprint) -> float:
        """
        Calculate confidence score for the fingerprint.
        
        Based on:
        - Attribute availability
        - Fingerprint stability
        - Collision risk
        
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Base confidence from stability score
        confidence = raw_fingerprint.stability_score * 0.6
        
        # Boost for fine-grained fingerprint
        hierarchy_boost = {
            "fine": 0.4,
            "medium": 0.25,
            "coarse": 0.1,
        }
        confidence += hierarchy_boost.get(raw_fingerprint.hierarchy_level, 0.1)
        
        # Penalty for sparse attributes
        device_attrs = raw_fingerprint.attributes.get("device", {})
        if len(device_attrs) < 5:
            confidence *= 0.7
        elif len(device_attrs) < 10:
            confidence *= 0.9
        
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_fallback_fingerprint(
        self,
        partial_attributes: Dict[str, Any],
        request_id: Optional[str]
    ) -> DeviceFingerprint:
        """
        Generate fallback fingerprint when full collection fails.
        
        Uses available attributes with lower confidence score.
        """
        # Create basic device attributes from partials
        device_attrs = DeviceAttributes(
            user_agent=partial_attributes.get("user_agent", ""),
            platform=partial_attributes.get("platform", ""),
        )
        
        # Generate minimal fingerprint
        raw_fingerprint = self.fingerprint_generator.create(
            device=device_attrs,
            network=NetworkAttributes(),
            behavioral=BehavioralAttributes()
        )
        
        # Low confidence for fallback
        confidence = 0.3
        
        # Higher risk for uncertain fingerprint
        risk_score = 0.6
        
        fingerprint = DeviceFingerprint(
            id=raw_fingerprint.id,
            confidence=confidence,
            risk_score=risk_score,
            collision_risk=0.5,  # High collision risk for sparse fingerprint
            device_attrs=device_attrs,
            network_attrs=NetworkAttributes(),
            behavioral_attrs=BehavioralAttributes(),
            is_known_device=False,
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow(),
            raw_attributes={"fallback": True, "partial": partial_attributes},
        )
        
        logger.warning(
            "Fallback fingerprint generated",
            extra={
                "fingerprint_id": fingerprint.id[:16] + "...",
                "confidence": confidence,
                "request_id": request_id,
            }
        )
        
        return fingerprint
    
    def match_fingerprint(
        self,
        fingerprint: DeviceFingerprint,
        known_fingerprints: list
    ) -> Optional[DeviceFingerprint]:
        """
        Match fingerprint against known devices.
        
        Args:
            fingerprint: Fingerprint to match
            known_fingerprints: List of known device fingerprints
            
        Returns:
            Matching fingerprint if found, None otherwise
        """
        for known_fp in known_fingerprints:
            if known_fp.id == fingerprint.id:
                return known_fp
            
            # Check for probabilistic match at hierarchy levels
            fp_hierarchy = fingerprint.raw_attributes.get("fingerprint_hierarchy", {})
            known_hierarchy = known_fp.raw_attributes.get("fingerprint_hierarchy", {})
            
            # Fine match is definitive
            if (fp_hierarchy.get("fine") and 
                fp_hierarchy.get("fine") == known_hierarchy.get("fine")):
                return known_fp
            
            # Medium match with high confidence
            if (fp_hierarchy.get("medium") and
                fp_hierarchy.get("medium") == known_hierarchy.get("medium") and
                fingerprint.confidence >= 0.8):
                return known_fp
        
        return None
