"""
Configuration management for the fraud detection system.

Settings come from environment variables (or a local .env file), which is the
only configuration mechanism. Environment-specific subclasses used to exist but
were removed: every value they overrode is already an environment variable, so
they only added a second place where the same setting could be defined.
"""

import logging
import secrets
from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

Environment = Literal["development", "testing", "production"]

# Shortest signing key accepted, in characters.
#
# HS256 is HMAC-SHA256, so a key shorter than the 32-byte hash output adds no
# entropy the algorithm can use, and a short one is brute-forceable offline:
# recovering it lets anyone mint a token for any account. PyJWT raises
# InsecureKeyLengthWarning below this same threshold (RFC 7518 section 3.2);
# this turns that warning into a refusal to start.
MINIMUM_SECRET_KEY_LENGTH = 32


class ApplicationSettings(BaseSettings):
    """
    Application configuration, populated from the environment.

    Field names map to upper-case environment variables automatically
    (``app_name`` reads ``APP_NAME``), so no explicit env= aliases are needed.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # model_path / model_reload_interval refer to ML models, not Pydantic
        # models, so the "model_" prefix guard does not apply here.
        protected_namespaces=(),
    )

    # Application
    app_name: str = "Advanced Fraud Detection System"
    app_version: str = "2.9.0"
    environment: Environment = "development"
    debug: bool = False
    port: int = 8000

    # Security.
    #
    # There is deliberately no usable default. A shipped default signing key is
    # a published signing key: anyone can mint a valid token for any account.
    # Outside production an ephemeral random key is generated per process, so
    # local development works with no setup while tokens stay unforgeable and
    # do not survive a restart.
    # Empty string means "unset"; the validator below always fills it, so the
    # type stays a plain str for callers that pass it to jwt.encode/decode.
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # CORS / trusted hosts
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    allowed_hosts: list[str] = ["localhost", "127.0.0.1"]

    # Database
    database_url: str = "sqlite+aiosqlite:///./fraud_detection.db"
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Redis cache
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 3600

    # ML models
    model_path: str = "./models"
    model_reload_interval: int = 86400

    # Logging
    log_level: str = "INFO"
    log_file: str = "fraud_detection.log"
    enable_file_logging: bool = True

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # Trust proxy headers (X-Forwarded-For) for client IP resolution.
    #
    # Off by default. When the app is directly reachable, these headers are
    # attacker-controlled, and trusting them lets a client forge a new identity
    # per request to slip past rate limiting. Enable only behind a proxy that
    # overwrites the header.
    trust_proxy_headers: bool = False

    # Performance
    max_batch_size: int = 1000
    request_timeout: int = 30

    @field_validator("cors_origins", "allowed_hosts", mode="before")
    @classmethod
    def _split_comma_separated(cls, value):
        """Accept both JSON arrays and plain comma-separated env values."""
        if isinstance(value, str) and not value.strip().startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _require_a_strong_secret_key(self) -> "ApplicationSettings":
        """
        Enforce that the signing key both exists and is long enough to be safe.

        Presence used to be the whole check, which let ``SECRET_KEY=x`` start a
        production process: the token signature was then guessable offline, so
        the auth layer looked enforced while being trivially forgeable. Length
        is validated in every environment, not just production, because a key
        weak enough to forge tokens is a defect wherever it is set - and a
        development value is exactly how a weak key reaches production.
        """
        generate = 'Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"'

        if not self.secret_key:
            if self.environment == "production":
                raise ValueError(f"SECRET_KEY must be set when ENVIRONMENT=production. {generate}")

            self.secret_key = secrets.token_urlsafe(MINIMUM_SECRET_KEY_LENGTH)
            logger.warning(
                "SECRET_KEY is not set; generated an ephemeral key for the %s "
                "environment. Tokens will be invalidated on restart.",
                self.environment,
            )
            return self

        if len(self.secret_key) < MINIMUM_SECRET_KEY_LENGTH:
            raise ValueError(
                f"SECRET_KEY must be at least {MINIMUM_SECRET_KEY_LENGTH} "
                f"characters; got {len(self.secret_key)}. A shorter key can be "
                f"recovered offline, which lets anyone mint a valid token for "
                f"any account. {generate}"
            )

        return self

    @model_validator(mode="after")
    def _reject_wildcard_cors_with_credentials(self) -> "ApplicationSettings":
        """
        A wildcard origin cannot be combined with credentialed requests. The
        app sends allow_credentials=True, so "*" here would be silently
        rejected by browsers while looking permissive in config.
        """
        if "*" in self.cors_origins and self.environment == "production":
            raise ValueError(
                "CORS_ORIGINS may not contain '*' in production; list the "
                "exact origins that are allowed to send credentials."
            )
        return self


@lru_cache
def get_application_settings() -> ApplicationSettings:
    """Return the process-wide settings, loaded once."""
    return ApplicationSettings()
