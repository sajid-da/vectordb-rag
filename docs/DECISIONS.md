# Decisions

- FastAPI and Uvicorn for V0 HTTP serving; specified stack; rejected a custom HTTP server.
- pytest, ruff, and mypy for V0 verification; specified stack; rejected untyped checks only.
- Direct handler unit test for V0; avoids an incompatible local Python 3.14 test client; rejected introducing an unneeded HTTP test dependency.
- Setuptools packaging with a `src/` layout; standard Python import isolation; rejected running application code from the repository root.
- No model or external API in V0; the milestone needs only a liveness endpoint; rejected premature embedding integration.
