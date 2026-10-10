"""Exact vector search used as the correctness baseline for later indexes."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class SearchResult:
    """One matching vector, ordered from most to least similar."""

    index: int
    score: float


def cosine_similarity(query: FloatVector, vector: FloatVector) -> float:
    """Return cosine similarity for two non-zero vectors of the same length."""
    if query.shape != vector.shape:
        raise ValueError("query and vector must have the same shape")

    denominator = float(np.linalg.norm(query) * np.linalg.norm(vector))
    if denominator == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")

    return float(np.dot(query, vector) / denominator)


def exact_search(
    vectors: FloatMatrix, query: FloatVector, limit: int
) -> list[SearchResult]:
    """Search every stored vector and return the highest cosine scores."""
    if vectors.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    if query.ndim != 1 or vectors.shape[1] != query.shape[0]:
        raise ValueError("query dimension must match the stored vector dimension")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if len(vectors) == 0:
        return []

    scores = [cosine_similarity(query, vector) for vector in vectors]
    ranked_indexes = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    return [
        SearchResult(index=index, score=scores[index])
        for index in ranked_indexes[:limit]
    ]
