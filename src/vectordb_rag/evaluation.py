"""Retrieval evaluation loading and metrics."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    """One query and the chunk IDs that should support it."""

    id: str
    question: str
    expected_chunk_ids: list[str]

    @property
    def is_answerable(self) -> bool:
        """Whether this query has expected supporting evidence."""
        return bool(self.expected_chunk_ids)


@dataclass(frozen=True)
class RetrievalMetrics:
    """Aggregate retrieval quality over answerable evaluation cases."""

    recall_at_k: float
    mean_reciprocal_rank: float
    evaluated_cases: int


def load_eval_cases(path: Path) -> list[EvalCase]:
    """Load newline-delimited JSON evaluation cases."""
    cases: list[EvalCase] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            case = EvalCase(
                id=value["id"],
                question=value["question"],
                expected_chunk_ids=value["expected_chunk_ids"],
            )
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            message = f"invalid evaluation case on line {line_number}"
            raise ValueError(message) from error
        cases.append(case)
    return cases


def calculate_retrieval_metrics(
    cases: list[EvalCase], retrieved_chunk_ids: dict[str, list[str]], k: int
) -> RetrievalMetrics:
    """Calculate recall@k and MRR; exclude intentionally unanswerable queries."""
    if k < 1:
        raise ValueError("k must be at least 1")

    answerable_cases = [case for case in cases if case.is_answerable]
    if not answerable_cases:
        raise ValueError("at least one answerable case is required")

    hits = 0
    reciprocal_ranks: list[float] = []
    for case in answerable_cases:
        results = retrieved_chunk_ids.get(case.id, [])[:k]
        expected = set(case.expected_chunk_ids)
        matching_rank = next(
            (rank for rank, chunk_id in enumerate(results, 1) if chunk_id in expected),
            None,
        )
        if matching_rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / matching_rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(answerable_cases)
    return RetrievalMetrics(
        recall_at_k=hits / total,
        mean_reciprocal_rank=sum(reciprocal_ranks) / total,
        evaluated_cases=total,
    )
