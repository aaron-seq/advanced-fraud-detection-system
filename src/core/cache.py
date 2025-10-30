# src/core/cache.py

import redis.asyncio as redis
import json
from typing import Any, Optional
from app.core.config import get_application_settings
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages caching operations with JSON serialization support.
    """

    def __init__(self, redis_client):
        self.redis = redis_client
        self.settings = get_application_settings()

    async def get(self, key: str) -> Optional[Any]:
        """Retrieves and deserializes a JSON value from the cache."""
        try:
            value = await self.redis.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Cache get error for key '{key}': {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Serializes a JSON value and stores it in the cache with a TTL."""
        try:
            ttl = ttl or self.settings.cache_ttl
            serialized_value = json.dumps(value, default=str)
            return await self.redis.setex(key, ttl, serialized_value)
        except Exception as e:
            logger.error(f"Cache set error for key '{key}': {e}")
            return False

class MockRedisClient:
    """
    A mock Redis client for development and testing when Redis is unavailable.
    """

    def __init__(self):
        self._data = {}

    async def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> bool:
        self._data[key] = value
        return True

    async def delete(self, key: str) -> bool:
        self._data.pop(key, None)
        return True

# Global instances for application-wide use
redis_client: Optional[redis.Redis] = None
cache_manager: Optional[CacheManager] = None

def get_redis_client() -> redis.Redis:
    """Initializes and returns the Redis client, with a mock fallback."""
    global redis_client
    if redis_client is None:
        try:
            settings = get_application_settings()
            redis_client = redis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
            logger.info("Redis client initialized successfully.")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using mock client.")
            redis_client = MockRedisClient()
    return redis_client

def get_cache_manager() -> CacheManager:
    """Initializes and returns the cache manager."""
    global cache_manager
    if cache_manager is None:
        cache_manager = CacheManager(get_redis_client())
    return cache_manager
