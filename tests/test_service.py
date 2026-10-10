"""Service-layer tests for storage, searching, and request limiting."""

from pathlib import Path

from vectordb_rag.service import FixedWindowRateLimiter, VectorService
from vectordb_rag.storage import PersistentVectorStore


def test_service_add_search_and_delete(tmp_path: Path) -> None:
    service = VectorService(PersistentVectorStore(tmp_path))
    service.add("east", [1.0, 0.0])
    service.add("north", [0.0, 1.0])

    assert service.search([1.0, 0.0], limit=1)[0].id == "east"

    service.delete("east")
    assert service.search([1.0, 0.0], limit=10)[0].id == "north"


def test_rate_limiter_rejects_requests_past_limit() -> None:
    limiter = FixedWindowRateLimiter(limit=2, window_seconds=60)

    assert limiter.allow("client")
    assert limiter.allow("client")
    assert not limiter.allow("client")
