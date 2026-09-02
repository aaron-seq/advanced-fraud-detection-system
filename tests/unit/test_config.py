"""
Tests for the settings validators.

These guard production invariants that nothing else can catch: the app reads
its configuration once at startup, so a weak value accepted here is a weak
value for the lifetime of the process.
"""

import pytest
from pydantic import ValidationError

from app.core.config import MINIMUM_SECRET_KEY_LENGTH, ApplicationSettings


def settings(**overrides) -> ApplicationSettings:
    """
    Build settings from explicit values only.

    ``_env_file=None`` stops a developer's local .env from leaking into the
    run, which would make these tests pass or fail depending on the machine.
    """
    return ApplicationSettings(_env_file=None, **overrides)


class TestSecretKeyStrength:
    """
    The signing key is the only thing standing between a request and a forged
    identity. HS256 with a short key is brute-forceable offline, so presence
    alone is not a sufficient check.
    """

    def test_production_rejects_a_missing_key(self):
        with pytest.raises(ValidationError, match="SECRET_KEY must be set"):
            settings(environment="production", secret_key="")

    def test_production_rejects_a_short_key(self):
        """
        The defect this test was written for: the validator checked only that
        the key was non-empty, so SECRET_KEY=x booted production happily.
        """
        with pytest.raises(ValidationError, match="at least"):
            settings(environment="production", secret_key="x")

    def test_production_rejects_a_key_one_byte_under_the_floor(self):
        short = "a" * (MINIMUM_SECRET_KEY_LENGTH - 1)
        with pytest.raises(ValidationError, match="at least"):
            settings(environment="production", secret_key=short)

    def test_production_accepts_a_key_at_the_floor(self):
        exact = "a" * MINIMUM_SECRET_KEY_LENGTH
        assert settings(environment="production", secret_key=exact).secret_key == exact

    def test_the_floor_matches_the_hs256_block_size(self):
        """
        32 bytes is not arbitrary: it is SHA-256's output size, and the length
        below which PyJWT itself raises InsecureKeyLengthWarning.
        """
        assert MINIMUM_SECRET_KEY_LENGTH == 32

    @pytest.mark.parametrize("environment", ["development", "testing"])
    def test_a_weak_key_is_refused_outside_production_too(self, environment):
        """
        A key weak enough to forge tokens is a bug wherever it is set. The
        production-only rule was that a key must *exist*; strength is not
        environment-specific, and allowing "x" in development is how "x"
        reaches production.
        """
        with pytest.raises(ValidationError, match="at least"):
            settings(environment=environment, secret_key="x")

    @pytest.mark.parametrize("environment", ["development", "testing"])
    def test_an_absent_key_is_generated_outside_production(self, environment):
        """Local development must still work with no configuration at all."""
        generated = settings(environment=environment, secret_key="").secret_key
        assert len(generated) >= MINIMUM_SECRET_KEY_LENGTH

    def test_generated_keys_differ_between_processes(self):
        assert settings(secret_key="").secret_key != settings(secret_key="").secret_key


class TestCorsOrigins:
    def test_production_rejects_a_wildcard_origin(self):
        with pytest.raises(ValidationError, match="may not contain"):
            settings(
                environment="production",
                secret_key="a" * MINIMUM_SECRET_KEY_LENGTH,
                cors_origins=["*"],
            )

    def test_a_comma_separated_string_is_split_into_a_list(self):
        """The env var arrives as one string; every value must be usable."""
        parsed = settings(cors_origins="https://a.example, https://b.example")
        assert parsed.cors_origins == ["https://a.example", "https://b.example"]
