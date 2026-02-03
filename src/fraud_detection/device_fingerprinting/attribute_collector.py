"""
Attribute Collectors for Device Fingerprinting

Collects various attributes from request context for fingerprinting:
- Device attributes (hardware, browser, etc.)
- Network attributes (IP, geolocation, etc.)
- Behavioral attributes (mouse, keyboard, etc.)
"""

import hashlib
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ...block.models import DeviceAttributes, NetworkAttributes, BehavioralAttributes
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class AttributeCollectionError(Exception):
    """Raised when attribute collection partially fails."""
    
    def __init__(self, message: str, partial_attributes: Dict[str, Any] = None):
        super().__init__(message)
        self.partial_attributes = partial_attributes or {}


class DeviceAttributeCollector:
    """
    Collects device-level attributes for fingerprinting.
    
    Attributes collected:
    - User agent parsed components
    - Screen resolution and color depth
    - Timezone and language
    - Platform and CPU info
    - Canvas fingerprint
    - WebGL fingerprint
    - Audio context fingerprint
    - Installed fonts hash
    - Browser plugins hash
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.comprehensive_mode = self.config.get("comprehensive", True)
    
    def collect(self, request_context: Dict[str, Any]) -> DeviceAttributes:
        """
        Collect device attributes from request context.
        
        Args:
            request_context: HTTP request metadata
            
        Returns:
            DeviceAttributes with collected values
            
        Raises:
            AttributeCollectionError: If critical attributes cannot be collected
        """
        try:
            headers = request_context.get("headers", {})
            client_data = request_context.get("client_data", {})
            
            # Parse user agent
            user_agent = self._get_header(headers, "user-agent", "")
            platform_info = self._parse_user_agent(user_agent)
            
            # Build device attributes
            attrs = DeviceAttributes(
                user_agent=user_agent[:500],  # Limit length
                screen_resolution=client_data.get("screen_resolution", ""),
                timezone=client_data.get("timezone", ""),
                language=self._get_header(headers, "accept-language", "")[:100],
                platform=platform_info.get("platform", ""),
                cpu_cores=client_data.get("cpu_cores"),
                memory_gb=client_data.get("device_memory"),
                gpu_renderer=client_data.get("gpu_renderer", "")[:200],
                canvas_fingerprint=self._hash_fingerprint(
                    client_data.get("canvas_fingerprint", "")
                ),
                webgl_fingerprint=self._hash_fingerprint(
                    client_data.get("webgl_fingerprint", "")
                ),
                audio_fingerprint=self._hash_fingerprint(
                    client_data.get("audio_fingerprint", "")
                ),
                fonts_hash=self._hash_fingerprint(
                    str(client_data.get("fonts", []))
                ),
                plugins_hash=self._hash_fingerprint(
                    str(client_data.get("plugins", []))
                ),
            )
            
            logger.debug(
                "Device attributes collected",
                extra={
                    "platform": attrs.platform,
                    "has_canvas": bool(attrs.canvas_fingerprint),
                    "has_webgl": bool(attrs.webgl_fingerprint),
                }
            )
            
            return attrs
            
        except Exception as e:
            logger.warning(f"Partial device attribute collection: {e}")
            raise AttributeCollectionError(
                f"Failed to collect device attributes: {e}",
                partial_attributes={"user_agent": request_context.get("headers", {}).get("user-agent", "")}
            )
    
    def _get_header(
        self,
        headers: Dict[str, Any],
        name: str,
        default: str = ""
    ) -> str:
        """Get header value (case-insensitive)."""
        for key, value in headers.items():
            if key.lower() == name.lower():
                if isinstance(value, bytes):
                    return value.decode("utf-8", errors="ignore")
                return str(value)
        return default
    
    def _parse_user_agent(self, user_agent: str) -> Dict[str, str]:
        """Parse user agent string to extract platform info."""
        result = {
            "platform": "Unknown",
            "browser": "Unknown",
            "version": "",
        }
        
        if not user_agent:
            return result
        
        ua_lower = user_agent.lower()
        
        # Detect platform
        if "windows" in ua_lower:
            result["platform"] = "Windows"
        elif "mac os" in ua_lower or "macintosh" in ua_lower:
            result["platform"] = "macOS"
        elif "linux" in ua_lower:
            result["platform"] = "Linux"
        elif "android" in ua_lower:
            result["platform"] = "Android"
        elif "iphone" in ua_lower or "ipad" in ua_lower:
            result["platform"] = "iOS"
        
        # Detect browser
        if "chrome" in ua_lower and "edg" not in ua_lower:
            result["browser"] = "Chrome"
        elif "firefox" in ua_lower:
            result["browser"] = "Firefox"
        elif "safari" in ua_lower and "chrome" not in ua_lower:
            result["browser"] = "Safari"
        elif "edg" in ua_lower:
            result["browser"] = "Edge"
        
        return result
    
    def _hash_fingerprint(self, value: str) -> str:
        """Hash fingerprint value for consistent length."""
        if not value:
            return ""
        return hashlib.sha256(value.encode()).hexdigest()[:32]


class NetworkContextCollector:
    """
    Collects network-level attributes for fingerprinting.
    
    Attributes collected:
    - IP address and version
    - Geolocation (country, region, city)
    - ISP and organization
    - VPN/Proxy/Tor detection
    - Datacenter IP detection
    - IP reputation score
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # In production, would integrate with IP reputation services
        self.reputation_service = None
    
    def collect(self, request_context: Dict[str, Any]) -> NetworkAttributes:
        """
        Collect network attributes from request context.
        
        Args:
            request_context: HTTP request metadata
            
        Returns:
            NetworkAttributes with collected values
        """
        try:
            headers = request_context.get("headers", {})
            
            # Get IP address (considering proxies)
            ip_address = self._get_client_ip(request_context, headers)
            
            # Determine IP version
            ip_version = "6" if ":" in ip_address else "4"
            
            # Get geolocation (would integrate with GeoIP service)
            geolocation = self._get_geolocation(ip_address)
            
            # Detect VPN/Proxy (simplified - would use threat intelligence)
            is_vpn, is_proxy, is_tor, is_datacenter = self._detect_anonymizers(
                ip_address, headers
            )
            
            # Get threat score (would use IP reputation service)
            threat_score = self._get_threat_score(ip_address)
            
            attrs = NetworkAttributes(
                ip_address=ip_address,
                ip_version=ip_version,
                geolocation=geolocation,
                isp=geolocation.get("isp", ""),
                organization=geolocation.get("org", ""),
                is_vpn=is_vpn,
                is_proxy=is_proxy,
                is_tor=is_tor,
                is_datacenter=is_datacenter,
                threat_score=threat_score,
            )
            
            logger.debug(
                "Network attributes collected",
                extra={
                    "ip_version": ip_version,
                    "country": geolocation.get("country", "Unknown"),
                    "is_vpn": is_vpn,
                    "threat_score": threat_score,
                }
            )
            
            return attrs
            
        except Exception as e:
            logger.warning(f"Error collecting network attributes: {e}")
            return NetworkAttributes()
    
    def _get_client_ip(
        self,
        request_context: Dict[str, Any],
        headers: Dict[str, Any]
    ) -> str:
        """Extract client IP considering proxy headers."""
        # Check forwarded headers
        forwarded_for = None
        for header_name in ["x-forwarded-for", "x-real-ip", "cf-connecting-ip"]:
            for key, value in headers.items():
                if key.lower() == header_name:
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="ignore")
                    forwarded_for = value.split(",")[0].strip()
                    break
            if forwarded_for:
                break
        
        if forwarded_for:
            return forwarded_for
        
        # Fallback to direct connection IP
        client = request_context.get("client", ())
        if client and len(client) >= 1:
            return str(client[0])
        
        return request_context.get("ip", "0.0.0.0")
    
    def _get_geolocation(self, ip_address: str) -> Dict[str, Any]:
        """
        Get geolocation for IP address.
        
        In production, would integrate with MaxMind GeoIP or similar service.
        """
        # Placeholder - would use actual geolocation service
        return {
            "country": "Unknown",
            "country_code": "XX",
            "region": "Unknown",
            "city": "Unknown",
            "latitude": 0.0,
            "longitude": 0.0,
            "isp": "Unknown",
            "org": "Unknown",
        }
    
    def _detect_anonymizers(
        self,
        ip_address: str,
        headers: Dict[str, Any]
    ) -> tuple[bool, bool, bool, bool]:
        """
        Detect if connection is through VPN, proxy, or Tor.
        
        Returns: (is_vpn, is_proxy, is_tor, is_datacenter)
        """
        is_proxy = False
        is_vpn = False
        is_tor = False
        is_datacenter = False
        
        # Check for proxy headers
        proxy_headers = [
            "via", "x-proxy-id", "x-forwarded-for",
            "forwarded", "x-real-ip"
        ]
        for key in headers:
            if key.lower() in proxy_headers:
                is_proxy = True
                break
        
        # Would integrate with threat intelligence for VPN/Tor detection
        # This is a simplified placeholder
        
        return is_vpn, is_proxy, is_tor, is_datacenter
    
    def _get_threat_score(self, ip_address: str) -> float:
        """
        Get threat score for IP address.
        
        In production, would query IP reputation service.
        Returns 0.0 (safe) to 1.0 (malicious).
        """
        # Placeholder - would use actual reputation service
        return 0.0


class BehavioralAttributeCollector:
    """
    Collects behavioral attributes for fingerprinting.
    
    Attributes collected:
    - Typing speed and cadence
    - Mouse movement patterns
    - Scroll behavior
    - Session interaction patterns
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    def collect(self, request_context: Dict[str, Any]) -> BehavioralAttributes:
        """
        Collect behavioral attributes from client-provided data.
        
        Args:
            request_context: HTTP request with behavioral signals
            
        Returns:
            BehavioralAttributes with collected values
        """
        try:
            behavioral_data = request_context.get("behavioral_data", {})
            
            attrs = BehavioralAttributes(
                typing_speed_wpm=behavioral_data.get("typing_speed"),
                typing_cadence=behavioral_data.get("typing_cadence", []),
                mouse_speed=behavioral_data.get("mouse_speed"),
                mouse_acceleration=behavioral_data.get("mouse_acceleration"),
                scroll_behavior=behavioral_data.get("scroll_behavior"),
                session_duration_seconds=behavioral_data.get("session_duration"),
                pages_visited=behavioral_data.get("pages_visited", 0),
                interaction_pattern_hash=self._hash_interaction_pattern(
                    behavioral_data
                ),
            )
            
            logger.debug(
                "Behavioral attributes collected",
                extra={
                    "has_typing_data": attrs.typing_speed_wpm is not None,
                    "has_mouse_data": attrs.mouse_speed is not None,
                    "pages_visited": attrs.pages_visited,
                }
            )
            
            return attrs
            
        except Exception as e:
            logger.warning(f"Error collecting behavioral attributes: {e}")
            return BehavioralAttributes()
    
    def _hash_interaction_pattern(self, behavioral_data: Dict[str, Any]) -> str:
        """Create hash of interaction pattern for comparison."""
        pattern_elements = []
        
        if behavioral_data.get("typing_cadence"):
            # Quantize typing cadence
            cadence = behavioral_data["typing_cadence"]
            if isinstance(cadence, list) and len(cadence) > 5:
                avg = sum(cadence) / len(cadence)
                pattern_elements.append(f"tc:{avg:.2f}")
        
        if behavioral_data.get("mouse_speed"):
            pattern_elements.append(f"ms:{behavioral_data['mouse_speed']:.1f}")
        
        if not pattern_elements:
            return ""
        
        pattern_str = "|".join(pattern_elements)
        return hashlib.md5(pattern_str.encode()).hexdigest()[:16]
