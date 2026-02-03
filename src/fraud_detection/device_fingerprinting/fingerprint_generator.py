"""
Stable Fingerprint Generator

Generates stable, unique device fingerprints that:
- Resist cookie clearing
- Have low collision rates
- Track across sessions
- Use hierarchical fingerprinting (coarse -> fine)
"""

import hashlib
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

from ...block.models import DeviceAttributes, NetworkAttributes, BehavioralAttributes
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RawFingerprint:
    """Raw fingerprint data before final ID generation."""
    id: str
    attributes: Dict[str, Any]
    stability_score: float
    hierarchy_level: str  # "coarse", "medium", "fine"


class StableFingerprintGenerator:
    """
    Generates stable device fingerprints using hierarchical approach.
    
    Hierarchy levels:
    1. Coarse: Platform + browser family (most stable, least unique)
    2. Medium: + screen + timezone + language
    3. Fine: + canvas + WebGL + audio fingerprints (most unique)
    
    This approach reduces division (same device getting different IDs)
    while maintaining uniqueness.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.salt = self.config.get("fingerprint_salt", "fraud-detection-2026")
    
    def create(
        self,
        device: DeviceAttributes,
        network: NetworkAttributes,
        behavioral: BehavioralAttributes
    ) -> RawFingerprint:
        """
        Generate stable fingerprint from collected attributes.
        
        Args:
            device: Device-level attributes
            network: Network-level attributes
            behavioral: Behavioral attributes
            
        Returns:
            RawFingerprint with ID and metadata
        """
        # Generate fingerprints at each hierarchy level
        coarse_fp = self._generate_coarse_fingerprint(device)
        medium_fp = self._generate_medium_fingerprint(device, coarse_fp)
        fine_fp = self._generate_fine_fingerprint(device, network, behavioral, medium_fp)
        
        # Calculate stability score based on attribute reliability
        stability_score = self._calculate_stability_score(device, network)
        
        # Determine which level to use based on attribute availability
        if fine_fp["available_attributes"] >= 8:
            final_id = fine_fp["id"]
            hierarchy_level = "fine"
        elif medium_fp["available_attributes"] >= 4:
            final_id = medium_fp["id"]
            hierarchy_level = "medium"
        else:
            final_id = coarse_fp["id"]
            hierarchy_level = "coarse"
        
        # Combine all attributes for storage
        all_attributes = {
            "device": self._serialize_device_attrs(device),
            "network": self._serialize_network_attrs(network),
            "behavioral": self._serialize_behavioral_attrs(behavioral),
            "fingerprint_hierarchy": {
                "coarse": coarse_fp["id"],
                "medium": medium_fp["id"],
                "fine": fine_fp["id"],
            }
        }
        
        logger.info(
            "Fingerprint generated",
            extra={
                "fingerprint_id": final_id[:16] + "...",
                "hierarchy_level": hierarchy_level,
                "stability_score": stability_score,
                "attribute_count": fine_fp["available_attributes"],
            }
        )
        
        return RawFingerprint(
            id=final_id,
            attributes=all_attributes,
            stability_score=stability_score,
            hierarchy_level=hierarchy_level,
        )
    
    def _generate_coarse_fingerprint(
        self,
        device: DeviceAttributes
    ) -> Dict[str, Any]:
        """
        Generate coarse fingerprint from stable attributes.
        
        Uses: platform, browser family, screen size category
        """
        components = []
        available = 0
        
        # Platform (very stable)
        if device.platform:
            components.append(f"p:{device.platform}")
            available += 1
        
        # Browser family from user agent
        if device.user_agent:
            browser_family = self._extract_browser_family(device.user_agent)
            components.append(f"b:{browser_family}")
            available += 1
        
        # Screen size category (stable across sessions)
        if device.screen_resolution:
            category = self._categorize_screen(device.screen_resolution)
            components.append(f"s:{category}")
            available += 1
        
        fingerprint_str = "|".join(components) + self.salt
        fingerprint_id = hashlib.sha256(fingerprint_str.encode()).hexdigest()
        
        return {
            "id": fingerprint_id,
            "components": components,
            "available_attributes": available,
        }
    
    def _generate_medium_fingerprint(
        self,
        device: DeviceAttributes,
        coarse: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate medium fingerprint adding timezone, language.
        
        Builds on coarse fingerprint.
        """
        components = coarse["components"].copy()
        available = coarse["available_attributes"]
        
        # Timezone
        if device.timezone:
            components.append(f"tz:{device.timezone}")
            available += 1
        
        # Language preference
        if device.language:
            # Use primary language only
            primary_lang = device.language.split(",")[0].split(";")[0].strip()
            components.append(f"l:{primary_lang}")
            available += 1
        
        # CPU cores (stable hardware attribute)
        if device.cpu_cores:
            components.append(f"cpu:{device.cpu_cores}")
            available += 1
        
        # Memory (stable but may vary with browser)
        if device.memory_gb:
            mem_category = self._categorize_memory(device.memory_gb)
            components.append(f"mem:{mem_category}")
            available += 1
        
        fingerprint_str = "|".join(components) + self.salt
        fingerprint_id = hashlib.sha256(fingerprint_str.encode()).hexdigest()
        
        return {
            "id": fingerprint_id,
            "components": components,
            "available_attributes": available,
        }
    
    def _generate_fine_fingerprint(
        self,
        device: DeviceAttributes,
        network: NetworkAttributes,
        behavioral: BehavioralAttributes,
        medium: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate fine fingerprint with canvas, WebGL, etc.
        
        Most unique but may vary across browser updates.
        """
        components = medium["components"].copy()
        available = medium["available_attributes"]
        
        # Canvas fingerprint
        if device.canvas_fingerprint:
            components.append(f"canvas:{device.canvas_fingerprint[:16]}")
            available += 1
        
        # WebGL fingerprint
        if device.webgl_fingerprint:
            components.append(f"webgl:{device.webgl_fingerprint[:16]}")
            available += 1
        
        # Audio fingerprint
        if device.audio_fingerprint:
            components.append(f"audio:{device.audio_fingerprint[:16]}")
            available += 1
        
        # GPU renderer
        if device.gpu_renderer:
            gpu_hash = hashlib.md5(device.gpu_renderer.encode()).hexdigest()[:8]
            components.append(f"gpu:{gpu_hash}")
            available += 1
        
        # Fonts hash (very unique)
        if device.fonts_hash:
            components.append(f"fonts:{device.fonts_hash[:8]}")
            available += 1
        
        # Network stability indicator (not the IP itself)
        if network.isp:
            isp_hash = hashlib.md5(network.isp.encode()).hexdigest()[:8]
            components.append(f"isp:{isp_hash}")
            available += 1
        
        fingerprint_str = "|".join(components) + self.salt
        fingerprint_id = hashlib.sha256(fingerprint_str.encode()).hexdigest()
        
        return {
            "id": fingerprint_id,
            "components": components,
            "available_attributes": available,
        }
    
    def _calculate_stability_score(
        self,
        device: DeviceAttributes,
        network: NetworkAttributes
    ) -> float:
        """
        Calculate how stable/reliable this fingerprint is.
        
        Returns 0.0 (unreliable) to 1.0 (very stable).
        """
        score = 0.0
        max_score = 0.0
        
        # Stable attributes (weight more)
        stable_attrs = [
            (device.platform, 2.0),
            (device.screen_resolution, 1.5),
            (device.timezone, 1.5),
            (device.cpu_cores, 1.0),
            (device.gpu_renderer, 1.5),
        ]
        
        for attr, weight in stable_attrs:
            max_score += weight
            if attr:
                score += weight
        
        # Less stable but useful
        variable_attrs = [
            (device.canvas_fingerprint, 1.0),
            (device.webgl_fingerprint, 1.0),
            (device.fonts_hash, 0.5),
        ]
        
        for attr, weight in variable_attrs:
            max_score += weight
            if attr:
                score += weight
        
        return score / max_score if max_score > 0 else 0.0
    
    def _extract_browser_family(self, user_agent: str) -> str:
        """Extract browser family from user agent."""
        ua_lower = user_agent.lower()
        
        if "chrome" in ua_lower and "edg" not in ua_lower:
            return "chrome"
        elif "firefox" in ua_lower:
            return "firefox"
        elif "safari" in ua_lower and "chrome" not in ua_lower:
            return "safari"
        elif "edg" in ua_lower:
            return "edge"
        else:
            return "other"
    
    def _categorize_screen(self, resolution: str) -> str:
        """Categorize screen resolution to reduce variance."""
        try:
            parts = resolution.lower().replace("x", " ").split()
            width = int(parts[0])
            
            if width >= 2560:
                return "4k+"
            elif width >= 1920:
                return "fhd"
            elif width >= 1366:
                return "hd"
            elif width >= 1024:
                return "tablet"
            else:
                return "mobile"
        except (ValueError, IndexError):
            return "unknown"
    
    def _categorize_memory(self, memory_gb: float) -> str:
        """Categorize memory to reduce variance."""
        if memory_gb >= 16:
            return "high"
        elif memory_gb >= 8:
            return "medium"
        elif memory_gb >= 4:
            return "low"
        else:
            return "minimal"
    
    def _serialize_device_attrs(self, device: DeviceAttributes) -> Dict[str, Any]:
        """Serialize device attributes for storage."""
        return {
            k: v for k, v in device.__dict__.items()
            if v not in (None, "", [], {})
        }
    
    def _serialize_network_attrs(self, network: NetworkAttributes) -> Dict[str, Any]:
        """Serialize network attributes for storage."""
        return {
            k: v for k, v in network.__dict__.items()
            if v not in (None, "", [], {}) and k != "ip_address"  # Don't store raw IP
        }
    
    def _serialize_behavioral_attrs(self, behavioral: BehavioralAttributes) -> Dict[str, Any]:
        """Serialize behavioral attributes for storage."""
        return {
            k: v for k, v in behavioral.__dict__.items()
            if v not in (None, "", [], {})
        }
