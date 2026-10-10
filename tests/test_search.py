"""Known-answer tests for brute-force cosine search."""

import numpy as np
import pytest

from vectordb_rag.search import cosine_similarity, exact_search


def test_cosine_similarity_of_matching_vectors_is_one() -> None:
    score = cosine_similarity(
        np.array([1.0, 0.0]),
        np.array([3.0, 0.0]),
    )

    assert score == pytest.approx(1.0)


def test_exact_search_returns_results_in_similarity_order() -> None:
    vectors = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    results = exact_search(vectors, np.array([1.0, 0.0]), limit=2)

    assert [result.index for result in results] == [1, 2]
    assert [result.score for result in results] == pytest.approx([1.0, 2**-0.5])


def test_exact_search_rejects_zero_vectors() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        exact_search(np.array([[0.0, 0.0]]), np.array([1.0, 0.0]), limit=1)
