# VectorDB + RAG

An incremental, from-scratch vector search engine with a citation-grounded RAG
layer. V0 provides a tested FastAPI health endpoint.

## Run

```sh
python -m pip install -e '.[dev]'
make test
make run
```

Then open `http://127.0.0.1:8000/health`.

## Current architecture

```mermaid
flowchart LR
  Client --> API[FastAPI]
  API --> Health[GET /health]
```

## Security

V0 accepts no user data and stores no credentials. `.env` is ignored; only
`.env.example` is committed. Rate limiting and input limits will be added when
public data endpoints arrive in V6.

## Limitations

There is no vector search, persistence, embedding model, or RAG workflow yet.
