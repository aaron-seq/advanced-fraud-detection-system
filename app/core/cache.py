"""
Redis cache management for the fraud detection system.
"""

import json
import logging
from typing import Any

import redis.asyncio as redis

from app.core.config import get_application_settings

logger = logging.getLogger(__name__)

_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """
    Return the process-wide Redis client.

    ``from_url`` only builds a connection pool, it does not dial the server, so
    wrapping it in try/except caught nothing and the in-memory fallback it
    guarded was unreachable. Connection failures surface on first command and
    are handled by CacheManager, which degrades to a cache miss.
    """
    global _redis_client

    if _redis_client is None:
        settings = get_application_settings()
        _redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_keepalive=True,
            health_check_interval=30,
        )
        logger.info("Redis client initialized for %s", settings.redis_url)

    return _redis_client


async def close_redis_client() -> None:
    """Close the Redis connection pool. Call during application shutdown."""
    global _redis_client

    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis client closed")


class CacheManager:
    """
    JSON-serialising cache wrapper.

    Redis is treated as an optional accelerator: if it is unavailable, reads
    report a miss and writes report failure rather than propagating, so a cache
    outage degrades latency instead of taking fraud detection down. Failures
    are logged, never silently swallowed.
    """

    def __init__(self, client: redis.Redis | None = None):
        self.redis = client or get_redis_client()
        self.settings = get_application_settings()

    async def get(self, key: str) -> Any | None:
        """Return the cached value for ``key``, or None on miss or error."""
        try:
            value = await self.redis.get(key)
        except redis.RedisError as exc:
            logger.warning("Cache unavailable on get(%s): %s", key, exc)
            return None

        if value is None:
            return None

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning("Discarding malformed cache entry for %s", key)
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Store ``value`` under ``key``. Returns False if the write failed."""
        try:
            await self.redis.setex(
                key, ttl or self.settings.cache_ttl, json.dumps(value, default=str)
            )
            return True
        except redis.RedisError as exc:
            logger.warning("Cache unavailable on set(%s): %s", key, exc)
            return False
        except (TypeError, ValueError) as exc:
            logger.error("Value for %s is not JSON-serialisable: %s", key, exc)
            return False

    async def delete(self, key: str) -> bool:
        """Remove ``key``. Returns False if the delete failed."""
        try:
            await self.redis.delete(key)
            return True
        except redis.RedisError as exc:
            logger.warning("Cache unavailable on delete(%s): %s", key, exc)
            return False

    async def ping(self) -> bool:
        """Report whether Redis is reachable."""
        try:
            return bool(await self.redis.ping())
        except redis.RedisError:
            return False
