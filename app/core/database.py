"""
Database connection management for the fraud detection system.

Uses SQLAlchemy 2.0's native async engine. The previous implementation layered
the third-party ``databases`` package on top of a second, synchronous engine:
two connection pools for one database, with the async one never actually
connected, so every query against it failed at runtime.
"""

import logging
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_application_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for ORM models."""


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _pool_options(database_url: str) -> dict:
    """
    Pool sizing options for the engine.

    SQLite is excluded: it is a local file, so a multi-connection pool buys
    nothing and file-locking makes it actively worse.
    """
    if database_url.startswith("sqlite"):
        return {}

    settings = get_application_settings()
    return {
        "pool_size": settings.database_pool_size,
        "max_overflow": settings.database_max_overflow,
        "pool_pre_ping": True,
    }


def get_engine() -> AsyncEngine:
    """Return the process-wide async engine, creating it on first use."""
    global _engine, _session_factory

    if _engine is None:
        settings = get_application_settings()
        _engine = create_async_engine(
            settings.database_url,
            **_pool_options(settings.database_url),
        )
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
        logger.info("Database engine created")

    return _engine


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield a database session, for use as a FastAPI dependency."""
    get_engine()  # sets _session_factory as a side effect
    if _session_factory is None:  # pragma: no cover - defensive
        raise RuntimeError("Session factory was not initialised")

    async with _session_factory() as session:
        yield session


async def dispose_engine() -> None:
    """Close all pooled connections. Call during application shutdown."""
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database engine disposed")
