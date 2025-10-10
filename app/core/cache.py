"""
Redis cache management for the fraud detection system
"""

import redis.asyncio as redis
import json
from typing import Any, Optional, Union
from app.core.config import get_application_settings
import logging

logger = logging.getLogger(__name__)

# Global Redis connection
redis_client = None

def get_redis_client():
    """
    Get Redis client connection
    """
    global redis_client
    
    if redis_client is None:
        settings = get_application_settings()
        
        try:
            redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            logger.info("Redis client initialized")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using mock client")
            redis_client = MockRedisClient()
    
    return redis_client

class CacheManager:
    """
    Cache management with JSON serialization support
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.settings = get_application_settings()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache with JSON deserialization
        """
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with JSON serialization
        """
        try:
            ttl = ttl or self.settings.cache_ttl
            serialized_value = json.dumps(value, default=str)
            return await self.redis.setex(key, ttl, serialized_value)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache
        """
        try:
            return await self.redis.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache
        """
        try:
            return await self.redis.exists(key)
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment counter in cache
        """
        try:
            return await self.redis.incr(key, amount)
        except Exception as e:
            logger.error(f"Cache increment error: {e}")
            return 0
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration for key
        """
        try:
            return await self.redis.expire(key, ttl)
        except Exception as e:
            logger.error(f"Cache expire error: {e}")
            return False

class MockRedisClient:
    """
    Mock Redis client for development/testing when Redis is not available
    """
    
    def __init__(self):
        self._data = {}
        self._expiry = {}
    
    async def get(self, key: str):
        return self._data.get(key)
    
    async def setex(self, key: str, ttl: int, value: str):
        self._data[key] = value
        return True
    
    async def delete(self, key: str):
        self._data.pop(key, None)
        return True
    
    async def exists(self, key: str):
        return key in self._data
    
    async def incr(self, key: str, amount: int = 1):
        current = int(self._data.get(key, 0))
        self._data[key] = str(current + amount)
        return current + amount
    
    async def expire(self, key: str, ttl: int):
        return True
    
    async def ping(self):
        return True
    
    async def close(self):
        pass

# Global cache manager instance
cache_manager = None

def get_cache_manager():
    """
    Get cache manager instance
    """
    global cache_manager
    
    if cache_manager is None:
        redis_client = get_redis_client()
        cache_manager = CacheManager(redis_client)
    
    return cache_manager