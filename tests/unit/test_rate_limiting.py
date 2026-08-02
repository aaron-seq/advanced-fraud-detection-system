"""
Unit tests for rate limiting.

These pin down the three defects the limiter previously had:
- a 429 raised as HTTPException inside middleware surfaced as a 500
- X-Forwarded-For was trusted unconditionally, so the limit was bypassable
- the identifier map grew without bound
"""

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.utils.rate_limiting import (
    WINDOW_SECONDS,
    InMemoryRateLimiter,
    RateLimitingMiddleware,
)


class TestInMemoryRateLimiter:
    def test_allows_up_to_the_limit_then_blocks(self):
        limiter = InMemoryRateLimiter()

        assert all(limiter.is_allowed("client", 3) for _ in range(3))
        assert limiter.is_allowed("client", 3) is False

    def test_clients_are_tracked_independently(self):
        limiter = InMemoryRateLimiter()

        assert limiter.is_allowed("a", 1) is True
        assert limiter.is_allowed("a", 1) is False
        assert limiter.is_allowed("b", 1) is True

    def test_remaining_counts_down_and_floors_at_zero(self):
        limiter = InMemoryRateLimiter()

        assert limiter.get_remaining("client", 2) == 2
        limiter.is_allowed("client", 2)
        assert limiter.get_remaining("client", 2) == 1
        limiter.is_allowed("client", 2)
        assert limiter.get_remaining("client", 2) == 0

    def test_requests_expire_out_of_the_window(self, monkeypatch):
        limiter = InMemoryRateLimiter()
        clock = [1000.0]
        monkeypatch.setattr(time, "monotonic", lambda: clock[0])

        assert limiter.is_allowed("client", 1) is True
        assert limiter.is_allowed("client", 1) is False

        clock[0] += WINDOW_SECONDS + 1
        assert limiter.is_allowed("client", 1) is True

    def test_identifier_map_is_bounded(self):
        """
        Unbounded growth would make the limiter its own DoS vector: one entry
        per spoofed source, never reclaimed.
        """
        limiter = InMemoryRateLimiter(max_identifiers=10)

        for i in range(500):
            limiter.is_allowed(f"client-{i}", 5)

        assert len(limiter._requests) <= 10

    def test_reset_clears_a_client(self):
        limiter = InMemoryRateLimiter()

        limiter.is_allowed("client", 1)
        assert limiter.is_allowed("client", 1) is False

        limiter.reset("client")
        assert limiter.is_allowed("client", 1) is True


def build_client(trust_proxy_headers: bool, limit: int = 2) -> TestClient:
    app = FastAPI()
    app.add_middleware(
        RateLimitingMiddleware,
        requests_per_minute=limit,
        trust_proxy_headers=trust_proxy_headers,
    )

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    return TestClient(app)


class TestRateLimitingMiddleware:
    def test_over_limit_returns_429_not_500(self):
        client = build_client(trust_proxy_headers=False)

        assert client.get("/ping").status_code == 200
        assert client.get("/ping").status_code == 200

        response = client.get("/ping")
        assert response.status_code == 429
        assert response.headers["Retry-After"] == str(WINDOW_SECONDS)

    def test_rate_limit_headers_are_present(self):
        client = build_client(trust_proxy_headers=False, limit=5)
        response = client.get("/ping")

        assert response.headers["X-RateLimit-Limit"] == "5"
        assert response.headers["X-RateLimit-Remaining"] == "4"

    def test_forwarded_header_cannot_be_used_to_evade_the_limit(self):
        """A spoofed X-Forwarded-For must not mint a fresh bucket."""
        client = build_client(trust_proxy_headers=False)

        statuses = [
            client.get("/ping", headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code
            for i in range(5)
        ]

        assert 429 in statuses

    def test_forwarded_header_is_honoured_when_explicitly_trusted(self):
        """Behind a trusted proxy the real client IP does come from the header."""
        client = build_client(trust_proxy_headers=True)

        statuses = [
            client.get("/ping", headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code
            for i in range(5)
        ]

        assert statuses == [200] * 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
