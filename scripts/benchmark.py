"""Compare the local HNSW implementation with FAISS on synthetic vectors."""

import json
import time

import faiss
import numpy as np

from vectordb_rag.hnsw import HnswIndex
from vectordb_rag.search import exact_search


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors for inner-product cosine search."""
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def main() -> None:
    """Run a deterministic, self-made local benchmark."""
    random = np.random.default_rng(42)
    vectors = normalize(random.normal(size=(500, 32)).astype("float32"))
    queries = normalize(random.normal(size=(25, 32)).astype("float32"))
    index = HnswIndex(m=8, ef_construction=64, seed=42)
    start = time.perf_counter()
    for vector in vectors:
        index.add(vector.astype(np.float64))
    hnsw_build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    hnsw_results = [index.search(query.astype(np.float64), 10, 64) for query in queries]
    hnsw_query_seconds = time.perf_counter() - start
    exact_results = [
        exact_search(vectors.astype(np.float64), query, 10) for query in queries
    ]
    recall = sum(
        len({x.index for x in approximate} & {x.index for x in exact}) / 10
        for approximate, exact in zip(hnsw_results, exact_results, strict=True)
    ) / len(queries)
    faiss_index = faiss.IndexFlatIP(32)
    start = time.perf_counter()
    faiss_index.add(vectors)
    faiss_build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    faiss_index.search(queries, 10)
    faiss_query_seconds = time.perf_counter() - start
    print(json.dumps({
        "dataset": "self-made: 500 random 32-dimensional vectors, 25 queries",
        "hnsw_build_seconds": hnsw_build_seconds,
        "hnsw_query_seconds": hnsw_query_seconds,
        "hnsw_recall_at_10": recall,
        "faiss_build_seconds": faiss_build_seconds,
        "faiss_query_seconds": faiss_query_seconds,
    }, indent=2))


if __name__ == "__main__":
    main()
