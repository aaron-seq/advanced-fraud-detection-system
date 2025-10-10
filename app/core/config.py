"""
Configuration management for the fraud detection system
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field
from functools import lru_cache

class ApplicationSettings(BaseSettings):
    """
    Application configuration settings with environment variable support
    """
    
    # Application Settings
    app_name: str = Field(default="Advanced Fraud Detection System", env="APP_NAME")
    app_version: str = Field(default="2.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    port: int = Field(default=8000, env="PORT")
    
    # Security Settings
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS Settings
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "https://your-frontend-domain.vercel.app"
        ],
        env="CORS_ORIGINS"
    )
    
    # Trusted Hosts
    allowed_hosts: List[str] = Field(
        default=[
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            ".railway.app",
            ".render.com",
            ".vercel.app"
        ],
        env="ALLOWED_HOSTS"
    )
    
    # Database Settings
    database_url: str = Field(
        default="sqlite:///./fraud_detection.db",
        env="DATABASE_URL"
    )
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
    # Redis Cache Settings
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # ML Model Settings
    model_path: str = Field(default="./models", env="MODEL_PATH")
    model_reload_interval: int = Field(default=86400, env="MODEL_RELOAD_INTERVAL")  # 24 hours
    enable_model_monitoring: bool = Field(default=True, env="ENABLE_MODEL_MONITORING")
    
    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="fraud_detection.log", env="LOG_FILE")
    enable_file_logging: bool = Field(default=True, env="ENABLE_FILE_LOGGING")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Monitoring & Analytics
    enable_analytics: bool = Field(default=True, env="ENABLE_ANALYTICS")
    analytics_retention_days: int = Field(default=90, env="ANALYTICS_RETENTION_DAYS")
    
    # External Services
    enable_notifications: bool = Field(default=False, env="ENABLE_NOTIFICATIONS")
    notification_webhook_url: Optional[str] = Field(default=None, env="NOTIFICATION_WEBHOOK_URL")
    
    # Performance Settings
    max_batch_size: int = Field(default=1000, env="MAX_BATCH_SIZE")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    worker_processes: int = Field(default=1, env="WORKER_PROCESSES")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class DevelopmentSettings(ApplicationSettings):
    """Development environment specific settings"""
    environment: str = "development"
    debug: bool = True
    log_level: str = "DEBUG"
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://localhost:8501"  # Streamlit
    ]

class ProductionSettings(ApplicationSettings):
    """Production environment specific settings"""
    environment: str = "production"
    debug: bool = False
    log_level: str = "INFO"
    
    # Production security
    secret_key: str = Field(..., env="SECRET_KEY")  # Required in production
    
    # Production database
    database_url: str = Field(..., env="DATABASE_URL")  # Required in production
    
    # Production cache
    redis_url: str = Field(..., env="REDIS_URL")  # Required in production

class TestingSettings(ApplicationSettings):
    """Testing environment specific settings"""
    environment: str = "testing"
    debug: bool = True
    database_url: str = "sqlite:///./test_fraud_detection.db"
    redis_url: str = "redis://localhost:6379/1"
    log_level: str = "DEBUG"

@lru_cache()
def get_application_settings() -> ApplicationSettings:
    """
    Get application settings based on environment
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()

# Export settings instance
settings = get_application_settings()