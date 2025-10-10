"""
Database connection management for the fraud detection system
"""

import asyncio
from databases import Database
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import get_application_settings
import logging

logger = logging.getLogger(__name__)

# SQLAlchemy setup
Base = declarative_base()
metadata = MetaData()

# Database connection
database = None
engine = None
SessionLocal = None

def get_database_connection():
    """
    Get database connection based on configuration
    """
    global database, engine, SessionLocal
    
    if database is None:
        settings = get_application_settings()
        
        # Create async database connection
        database = Database(settings.database_url)
        
        # Create SQLAlchemy engine for sync operations
        engine = create_engine(
            settings.database_url.replace('postgresql+asyncpg://', 'postgresql://'),
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow
        )
        
        # Create session factory
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
        
        logger.info("Database connection initialized")
    
    return database

def get_db_session():
    """
    Get database session for sync operations
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def connect_database():
    """
    Connect to database
    """
    global database
    if database:
        await database.connect()
        logger.info("Database connected")

async def disconnect_database():
    """
    Disconnect from database
    """
    global database
    if database:
        await database.disconnect()
        logger.info("Database disconnected")