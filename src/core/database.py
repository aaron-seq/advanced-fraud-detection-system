# src/core/database.py

from databases import Database
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import get_application_settings
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages the database connection, session, and engine for the application.
    """

    def __init__(self, db_url: str):
        self.database = Database(db_url)
        self.engine = create_engine(db_url.replace('postgresql+asyncpg://', 'postgresql://'))
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.Base = declarative_base()

    async def connect(self):
        """Establishes the database connection."""
        try:
            await self.database.connect()
            logger.info("Database connection established.")
        except Exception as e:
            logger.error(f"Could not connect to the database: {e}")

    async def disconnect(self):
        """Closes the database connection."""
        try:
            await self.database.disconnect()
            logger.info("Database connection closed.")
        except Exception as e:
            logger.error(f"Error while disconnecting from the database: {e}")

    def get_session(self):
        """Provides a new database session."""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def create_tables(self):
        """Creates all database tables defined in the Base metadata."""
        try:
            self.Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully.")
        except Exception as e:
            logger.error(f"Could not create database tables: {e}")

# Global instance for application-wide use
settings = get_application_settings()
db_manager = DatabaseManager(settings.database_url)
