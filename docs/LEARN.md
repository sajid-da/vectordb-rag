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

## V2

- A chunk is a small named piece of a source document that can be retrieved later.
- The embedding provider and answer provider are interfaces, so real models can be swapped in without changing search code.
- `RagPipeline` embeds chunks, finds the closest ones, and gives only those chunks to the answer provider.
- `RagAnswer` includes the chunks used, which is the first step toward citations.
- Tests replace model calls with simple fakes so they stay fast and repeatable.

## V3

- An evaluation case records a question and the source chunks that should be found.
- Recall@k asks whether a correct chunk appeared in the first k results.
- MRR rewards putting the first correct chunk closer to the top.
- The 100 cases are deliberately labeled self-made synthetic data, not real-user evidence.
- Unanswerable questions have no expected chunk and will later test abstention separately.
