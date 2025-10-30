# src/core/config.py

import os
from typing import List
from pydantic import BaseSettings, Field
from functools import lru_cache

class BaseAppSettings(BaseSettings):
    """
    Base application settings, loaded from environment variables.
    """
    app_name: str = "Advanced Fraud Detection System"
    app_version: str = "2.0.0"
    environment: str = Field(..., env="ENVIRONMENT")
    debug: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class ApiSettings(BaseAppSettings):
    """
    API-specific settings for security, CORS, and database connections.
    """
    secret_key: str = Field(..., env="SECRET_KEY")
    database_url: str = Field(..., env="DATABASE_URL")
    redis_url: str = Field(..., env="REDIS_URL")

    cors_origins: List[str] = ["http://localhost:3000"]

    # Model settings
    model_path: str = "./models"

# --- Environment-Specific Settings ---

class DevelopmentSettings(ApiSettings):
    """Settings for the development environment."""
    debug: bool = True

class ProductionSettings(ApiSettings):
    """Settings for the production environment."""
    pass

class TestingSettings(ApiSettings):
    """Settings for the testing environment."""
    database_url: str = "sqlite:///./test_fraud_detection.db"
    redis_url: str = "redis://localhost:6379/1"

@lru_cache()
def get_application_settings() -> BaseAppSettings:
    """
    Loads and returns the appropriate settings based on the ENVIRONMENT variable.
    """
    env = os.getenv("ENVIRONMENT", "development").lower()
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    return DevelopmentSettings()

# Global settings instance
settings = get_application_settings()
