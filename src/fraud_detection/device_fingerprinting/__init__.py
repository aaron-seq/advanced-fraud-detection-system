"""
Device Fingerprinting Package

Implements advanced device fingerprinting with:
- 2000+ device attributes
- Network context integration
- Behavioral signals
- Collision/division reduction
"""

from .visitor_fingerprint import VisitorFingerprintEngine
from .attribute_collector import (
    DeviceAttributeCollector,
    NetworkContextCollector,
    BehavioralAttributeCollector,
)
from .fingerprint_generator import StableFingerprintGenerator
from .collision_detector import CollisionDetector
from .risk_evaluator import DeviceRiskEvaluator

__all__ = [
    "VisitorFingerprintEngine",
    "DeviceAttributeCollector",
    "NetworkContextCollector",
    "BehavioralAttributeCollector",
    "StableFingerprintGenerator",
    "CollisionDetector",
    "DeviceRiskEvaluator",
]
