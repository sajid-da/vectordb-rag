"""Tests for retrieval evaluation metrics."""

from pathlib import Path

import pytest

from vectordb_rag.evaluation import (
    EvalCase,
    calculate_retrieval_metrics,
    load_eval_cases,
)


def test_metrics_ignore_unanswerable_cases() -> None:
    cases = [
        EvalCase("one", "first", ["chunk-a"]),
        EvalCase("two", "second", ["chunk-b"]),
        EvalCase("three", "unanswerable", []),
    ]
    retrieved = {"one": ["chunk-x", "chunk-a"], "two": ["chunk-b"]}

    metrics = calculate_retrieval_metrics(cases, retrieved, k=2)

    assert metrics.recall_at_k == 1.0
    assert metrics.mean_reciprocal_rank == pytest.approx(0.75)
    assert metrics.evaluated_cases == 2


def test_load_eval_cases_reads_jsonl() -> None:
    path = Path("data/eval_cases.jsonl")

    cases = load_eval_cases(path)

    assert len(cases) == 100
    assert sum(case.is_answerable for case in cases) == 80
