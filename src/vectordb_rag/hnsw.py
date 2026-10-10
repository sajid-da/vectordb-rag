"""A compact, single-process hierarchical navigable small-world index."""

from __future__ import annotations

import math
from collections import defaultdict
from heapq import heappop, heappush
from random import Random

import numpy as np

from vectordb_rag.search import FloatVector, SearchResult, cosine_similarity


class HnswIndex:
    """Approximate cosine search with randomly assigned hierarchical layers."""

    def __init__(self, m: int = 8, ef_construction: int = 32, seed: int = 0) -> None:
        if m < 2:
            raise ValueError("m must be at least 2")
        if ef_construction < m:
            raise ValueError("ef_construction must be at least m")
        self._m = m
        self._ef_construction = ef_construction
        self._random = Random(seed)
        self._vectors: list[FloatVector] = []
        self._levels: list[int] = []
        self._neighbors: dict[int, dict[int, set[int]]] = defaultdict(dict)
        self._entry_point: int | None = None
        self._max_level = -1

    def __len__(self) -> int:
        return len(self._vectors)

    def add(self, vector: FloatVector) -> int:
        """Insert a non-zero vector and return its stable integer identifier."""
        if vector.ndim != 1 or float(np.linalg.norm(vector)) == 0.0:
            raise ValueError("vector must be a non-zero one-dimensional array")
        if self._vectors and vector.shape != self._vectors[0].shape:
            raise ValueError("vector dimension must match the existing index")

        identifier = len(self._vectors)
        level = self._random_level()
        self._vectors.append(vector.copy())
        self._levels.append(level)

        if self._entry_point is None:
            self._entry_point = identifier
            self._max_level = level
            self._ensure_node(identifier, level)
            return identifier

        assert self._entry_point is not None
        entry_points = [self._entry_point]
        for current_level in range(self._max_level, level, -1):
            entry_points = [self._greedy_search(vector, entry_points[0], current_level)]

        for current_level in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(
                vector, entry_points, self._ef_construction, current_level
            )
            selected = self._select_neighbors(candidates, self._m)
            self._connect(identifier, selected, current_level)
            entry_points = selected or entry_points

        if level > self._max_level:
            self._entry_point = identifier
            self._max_level = level
        self._ensure_node(identifier, level)
        return identifier

    def search(
        self, query: FloatVector, limit: int, ef_search: int = 32
    ) -> list[SearchResult]:
        """Return up to `limit` approximate nearest neighbours by cosine similarity."""
        if limit < 1 or ef_search < limit:
            raise ValueError("limit must be positive and no larger than ef_search")
        if self._entry_point is None:
            return []
        if query.ndim != 1 or query.shape != self._vectors[0].shape:
            raise ValueError("query dimension must match the indexed vectors")
        if float(np.linalg.norm(query)) == 0.0:
            raise ValueError("query must be non-zero")

        entry_point = self._entry_point
        for level in range(self._max_level, 0, -1):
            entry_point = self._greedy_search(query, entry_point, level)
        candidates = self._search_layer(query, [entry_point], ef_search, 0)
        return [
            SearchResult(index=index, score=score)
            for score, index in candidates[:limit]
        ]

    def _random_level(self) -> int:
        level = 0
        while self._random.random() < 1 / math.e:
            level += 1
        return level

    def _ensure_node(self, identifier: int, level: int) -> None:
        for current_level in range(level + 1):
            self._neighbors[current_level].setdefault(identifier, set())

    def _score(self, query: FloatVector, identifier: int) -> float:
        return cosine_similarity(query, self._vectors[identifier])

    def _greedy_search(self, query: FloatVector, entry_point: int, level: int) -> int:
        best = entry_point
        best_score = self._score(query, best)
        changed = True
        while changed:
            changed = False
            for neighbor in self._neighbors[level].get(best, set()):
                score = self._score(query, neighbor)
                if score > best_score:
                    best, best_score, changed = neighbor, score, True
        return best

    def _search_layer(
        self, query: FloatVector, entry_points: list[int], ef: int, level: int
    ) -> list[tuple[float, int]]:
        visited = set(entry_points)
        candidates = [(-self._score(query, point), point) for point in entry_points]
        best = [(self._score(query, point), point) for point in entry_points]
        while candidates:
            negative_score, current = heappop(candidates)
            score = -negative_score
            worst_score = min(best)[0]
            if score < worst_score and len(best) >= ef:
                break
            for neighbor in self._neighbors[level].get(current, set()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                neighbor_score = self._score(query, neighbor)
                if len(best) < ef or neighbor_score > worst_score:
                    heappush(candidates, (-neighbor_score, neighbor))
                    heappush(best, (neighbor_score, neighbor))
                    if len(best) > ef:
                        best.remove(min(best))
                    worst_score = min(best)[0]
        return sorted(best, reverse=True)

    def _select_neighbors(
        self, candidates: list[tuple[float, int]], limit: int
    ) -> list[int]:
        return [identifier for _, identifier in candidates[:limit]]

    def _connect(self, identifier: int, neighbors: list[int], level: int) -> None:
        self._neighbors[level].setdefault(identifier, set())
        for neighbor in neighbors:
            self._neighbors[level][identifier].add(neighbor)
            self._neighbors[level].setdefault(neighbor, set()).add(identifier)
            if len(self._neighbors[level][neighbor]) > self._m:
                self._trim_neighbors(neighbor, level)

    def _trim_neighbors(self, identifier: int, level: int) -> None:
        neighbors = self._neighbors[level][identifier]
        ranked = sorted(
            neighbors,
            key=lambda neighbor: cosine_similarity(
                self._vectors[identifier], self._vectors[neighbor]
            ),
            reverse=True,
        )
        self._neighbors[level][identifier] = set(ranked[: self._m])
