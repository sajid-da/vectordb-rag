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

Exact next step: V3, create a 100+ question evaluation set with expected source
chunks and add recall@k and MRR measurement.
