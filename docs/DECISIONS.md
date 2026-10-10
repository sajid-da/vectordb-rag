# Decisions

- FastAPI and Uvicorn for V0 HTTP serving; specified stack; rejected a custom HTTP server.
- pytest, ruff, and mypy for V0 verification; specified stack; rejected untyped checks only.
- Direct handler unit test for V0; avoids an incompatible local Python 3.14 test client; rejected introducing an unneeded HTTP test dependency.
- Setuptools packaging with a `src/` layout; standard Python import isolation; rejected running application code from the repository root.
- No model or external API in V0; the milestone needs only a liveness endpoint; rejected premature embedding integration.
- NumPy powers exact vector math; required by the specified stack; rejected hand-written numeric loops.
- Zero vectors raise a clear validation error; cosine similarity is mathematically undefined; rejected silently assigning a score.
- V2 uses provider protocols and deterministic fakes; preserves a real-model integration seam while keeping tests free and repeatable; rejected live LLM calls in tests.
- Chunks are fixed-size word windows initially; transparent and easy to test; rejected complex overlapping chunking before an evaluation set exists.
