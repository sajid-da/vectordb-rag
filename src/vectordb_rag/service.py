"""Thread-safe application operations over the persistent vector store."""

from dataclasses import dataclass
from threading import Lock
from time import monotonic

import numpy as np

from vectordb_rag.search import exact_search
from vectordb_rag.storage import PersistentVectorStore


class FixedWindowRateLimiter:
    """Small in-memory rate limiter for a single-process public API."""

    def __init__(self, limit: int, window_seconds: float) -> None:
        self._limit = limit
        self._window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        """Record a request when it fits within the key's current window."""
        now = monotonic()
        with self._lock:
            recent = [
                timestamp
                for timestamp in self._requests.get(key, [])
                if timestamp > now - self._window_seconds
            ]
            if len(recent) >= self._limit:
                self._requests[key] = recent
                return False
            recent.append(now)
            self._requests[key] = recent
            return True


@dataclass(frozen=True)
class ServiceSearchResult:
    """A persisted vector result returned by the service layer."""

    id: str
    score: float


class VectorService:
    """Serialize storage mutations and perform exact searches over stored vectors."""

    def __init__(self, store: PersistentVectorStore) -> None:
        self._store = store
        self._lock = Lock()

    def add(self, identifier: str, values: list[float]) -> None:
        """Add or replace a vector under an application-level lock."""
        with self._lock:
            self._store.add(identifier, values)

    def delete(self, identifier: str) -> None:
        """Delete a vector under an application-level lock."""
        with self._lock:
            self._store.delete(identifier)

    def search(self, values: list[float], limit: int) -> list[ServiceSearchResult]:
        """Search stored vectors and return stable client IDs with scores."""
        with self._lock:
            stored = self._store.get_all()
        if not stored:
            return []
        matrix = np.array([item.values for item in stored], dtype=np.float64)
        results = exact_search(matrix, np.array(values, dtype=np.float64), limit)
        return [
            ServiceSearchResult(id=stored[result.index].id, score=result.score)
            for result in results
        ]
