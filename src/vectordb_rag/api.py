"""HTTP entry point for the service."""

from fastapi import FastAPI

app = FastAPI(title="VectorDB + RAG")


@app.get("/health")
def health() -> dict[str, str]:
    """Return a minimal liveness response without exposing internal details."""
    return {"status": "ok"}
