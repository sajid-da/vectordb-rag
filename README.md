# VectorDB + RAG

A from-scratch vector search engine with a citation-grounded RAG layer. It keeps
an exact baseline beside a compact HNSW index.

## Run

```sh
python -m pip install -e '.[dev]'
make test
make run
```

Then open `http://127.0.0.1:8000/health`.

## Architecture

```mermaid
flowchart LR
  Client --> API[FastAPI service]
  API --> Store[WAL + snapshots]
  API --> Search[Exact or HNSW search]
  Search --> RAG[Cited RAG pipeline]
  API --> Metrics[/metrics]
```

## Results

| Self-made benchmark | Result |
| --- | --- |
| HNSW recall@10 | 0.964 |
| HNSW query batch, 25 queries | 0.0449936 s |
| FAISS query batch, 25 queries | 0.0012869 s |

Run `.venv/Scripts/python scripts/benchmark.py` to reproduce.

## Security

Inputs are size-limited and validated; the API has a single-process rate limit.
`.env` is ignored, only `.env.example` is committed, and Dependabot watches
dependencies. This prototype has no authentication, so do not expose it to
untrusted users or store sensitive document content.

## Limitations

HNSW is single-process and not tuned for production workloads. The benchmark is
self-made synthetic data, not a production claim. Hosted deployment needs its
own provider and security review.
