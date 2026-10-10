"""Minimal Prometheus text-format metrics without a runtime dependency."""

from threading import Lock


class Metrics:
    """Thread-safe counters exported at the Prometheus scrape endpoint."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests = 0
        self._errors = 0

    def record_request(self, failed: bool = False) -> None:
        """Record one request and optionally one failure."""
        with self._lock:
            self._requests += 1
            self._errors += int(failed)

    def render(self) -> str:
        """Return Prometheus-compatible counter exposition text."""
        with self._lock:
            return (
                "# TYPE vectordb_requests_total counter\n"
                f"vectordb_requests_total {self._requests}\n"
                "# TYPE vectordb_errors_total counter\n"
                f"vectordb_errors_total {self._errors}\n"
            )
