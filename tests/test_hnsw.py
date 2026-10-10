"""Properties and recall checks for the HNSW index."""

import numpy as np
import pytest

from vectordb_rag.hnsw import HnswIndex
from vectordb_rag.search import exact_search


def test_search_returns_only_inserted_vectors() -> None:
    index = HnswIndex(m=4, ef_construction=12, seed=4)
    for number in range(12):
        index.add(np.array([float(number + 1), float(12 - number)]))

    results = index.search(np.array([1.0, 1.0]), limit=4, ef_search=12)

    assert {result.index for result in results}.issubset(set(range(12)))
    assert len(results) == 4


def test_hnsw_recall_against_exact_search() -> None:
    random = np.random.default_rng(9)
    vectors = random.normal(size=(40, 8))
    query = random.normal(size=8)
    index = HnswIndex(m=8, ef_construction=40, seed=9)
    for vector in vectors:
        index.add(vector)

    approximate = index.search(query, limit=5, ef_search=40)
    exact = exact_search(vectors, query, limit=5)
    approximate_ids = {item.index for item in approximate}
    exact_ids = {item.index for item in exact}
    recall = len(approximate_ids & exact_ids) / 5

    assert recall >= 0.8


def test_search_rejects_invalid_parameters() -> None:
    index = HnswIndex()
    index.add(np.array([1.0, 0.0]))

    with pytest.raises(ValueError, match="no larger"):
        index.search(np.array([1.0, 0.0]), limit=2, ef_search=1)
