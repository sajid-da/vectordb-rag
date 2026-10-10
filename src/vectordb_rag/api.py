"""HTTP entry point for the service."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from vectordb_rag.metrics import Metrics
from vectordb_rag.service import FixedWindowRateLimiter, VectorService
from vectordb_rag.storage import PersistentVectorStore

app = FastAPI(title="VectorDB + RAG")
service = VectorService(PersistentVectorStore(Path("data/runtime")))
rate_limiter = FixedWindowRateLimiter(limit=60, window_seconds=60)
metrics = Metrics()


class VectorInput(BaseModel):
    """Validated vector mutation payload."""

    id: str = Field(min_length=1, max_length=200)
    values: list[float] = Field(min_length=1, max_length=4096)


class SearchInput(BaseModel):
    """Validated search payload."""

    values: list[float] = Field(min_length=1, max_length=4096)
    limit: int = Field(default=10, ge=1, le=100)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a minimal liveness response without exposing internal details."""
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    """Expose operational counters in Prometheus's text format."""
    return Response(metrics.render(), media_type="text/plain; version=0.0.4")


def check_rate_limit(request: Request) -> None:
    """Reject clients above the public endpoint request limit."""
    client = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS)


@app.post("/vectors", status_code=status.HTTP_201_CREATED)
def add_vector(payload: VectorInput, request: Request) -> dict[str, str]:
    """Persist one vector."""
    metrics.record_request()
    check_rate_limit(request)
    try:
        service.add(payload.id, payload.values)
    except ValueError as error:
        metrics.record_request(failed=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY) from error
    return {"id": payload.id}


@app.delete("/vectors/{identifier}", status_code=status.HTTP_204_NO_CONTENT)
def delete_vector(identifier: str, request: Request) -> None:
    """Delete one vector by ID."""
    metrics.record_request()
    check_rate_limit(request)
    service.delete(identifier)


@app.post("/search")
def search_vectors(
    payload: SearchInput, request: Request
) -> dict[str, list[dict[str, object]]]:
    """Search vectors by cosine similarity."""
    metrics.record_request()
    check_rate_limit(request)
    try:
        results = service.search(payload.values, payload.limit)
    except ValueError as error:
        metrics.record_request(failed=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY) from error
    return {"results": [result.__dict__ for result in results]}
