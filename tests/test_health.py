"""Tests for the health endpoint."""

from vectordb_rag.api import health


def test_health_returns_ok() -> None:
    assert health() == {"status": "ok"}
