"""Tests for Prometheus metrics exposition."""

from vectordb_rag.metrics import Metrics


def test_metrics_render_counters() -> None:
    metrics = Metrics()
    metrics.record_request()
    metrics.record_request(failed=True)

    rendered = metrics.render()

    assert "vectordb_requests_total 2" in rendered
    assert "vectordb_errors_total 1" in rendered
