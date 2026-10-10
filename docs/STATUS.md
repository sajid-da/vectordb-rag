# Status

V0 is implemented locally: the FastAPI app exposes `GET /health` and has one
deterministic handler test. Run `python -m pip install -e '.[dev]'`, then
`make test`; run the app with `make run`.

Known issues: CI is configured but cannot be confirmed until this repository is
pushed to a GitHub remote. The local machine has Python 3.14, not the intended
Python 3.12 CI runtime; `mypy` is also not installed locally.

V1 is implemented: `vectordb_rag.search` compares every non-zero stored vector
using cosine similarity, returning its index and score in descending order.

V2 is implemented: `vectordb_rag.rag` chunks documents, embeds and retrieves
them through V1, then returns a generated answer plus its source chunks. Tests
use deterministic embedding and answer fakes rather than an external LLM.

V3 is implemented: `data/eval_cases.jsonl` contains 100 self-made cases, 80
answerable and 20 intentionally unanswerable. The evaluation module loads it
and calculates recall@k and MRR for answerable queries.

V4 is implemented: `HnswIndex` assigns random hierarchy levels, connects a
bounded neighbor graph, and searches it with configurable `m` and `ef_search`.
Its retrieval result IDs and recall are tested against the exact baseline.

Exact next step: V5, add snapshots and a write-ahead log with crash recovery.
