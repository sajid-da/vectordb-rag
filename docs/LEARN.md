# Learn

## V0

- `src/vectordb_rag/api.py` starts the web API and defines one health-check route.
- The health check lets deployment systems confirm that the process is responding.
- `tests/test_health.py` calls the app without opening a real network port.
- The Makefile gives short, repeatable commands for running checks and the server.
- Next, V1 will build the simple exact search used to verify all faster indexes.

## V1

- `search.py` holds the slow but correct baseline: it compares a query to every vector.
- Cosine similarity measures directional similarity, so vectors with different sizes can still match.
- `SearchResult` pairs a vector's original position with its score.
- Later HNSW search will be checked against this simple implementation.
- The tests use tiny vectors whose correct order can be worked out by hand.
