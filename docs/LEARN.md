# Learn

## V0

- `src/vectordb_rag/api.py` starts the web API and defines one health-check route.
- The health check lets deployment systems confirm that the process is responding.
- `tests/test_health.py` calls the app without opening a real network port.
- The Makefile gives short, repeatable commands for running checks and the server.
- Next, V1 will build the simple exact search used to verify all faster indexes.
