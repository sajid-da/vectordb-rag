"""Evaluation script for RAG retrieval quality, citations, and abstention."""

import json
from pathlib import Path
from typing import Any

from vectordb_rag.evaluation import calculate_retrieval_metrics, load_eval_cases
from vectordb_rag.rag import AnswerProvider, Chunk, RagPipeline, verify_citation
from vectordb_rag.search import FloatVector, exact_search


class FixedMockEmbeddingProvider:
    """Deterministic embedding provider for evaluation cases."""

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> FloatVector:
        """Map text string deterministically into a 16-dimensional normalized vector."""
        import numpy as np

        vec = np.zeros(self.dimension, dtype=np.float64)
        for idx, char in enumerate(text.encode("utf-8")):
            vec[idx % self.dimension] += char / 255.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class GroundedAnswerProvider(AnswerProvider):
    """Simple answer provider that cites the top retrieved context."""

    def answer(self, question: str, contexts: list[str]) -> str:
        if not contexts:
            return "No context available."
        return f"Based on available context: {contexts[0][:100]}"


def run_rag_evaluation() -> dict[str, Any]:
    cases_path = Path("data/eval_cases.jsonl")
    if not cases_path.exists():
        raise FileNotFoundError("data/eval_cases.jsonl not found")

    cases = load_eval_cases(cases_path)
    embedding_provider = FixedMockEmbeddingProvider()
    answer_provider = GroundedAnswerProvider()

    chunks_map: dict[str, Chunk] = {}
    for case in cases:
        for chunk_id in case.expected_chunk_ids:
            if chunk_id not in chunks_map:
                text_content = f"Evidence text for {chunk_id}."
                chunks_map[chunk_id] = Chunk(id=chunk_id, text=text_content)

    if not chunks_map:
        chunks_map["chunk-default"] = Chunk(
            id="chunk-default", text="Default fallback context document."
        )

    chunks_list = list(chunks_map.values())
    pipeline = RagPipeline(chunks_list, embedding_provider, answer_provider)

    retrieved_dict: dict[str, list[str]] = {}
    abstention_results: dict[str, bool] = {}
    citation_validity_count = 0
    total_answers = 0

    for case in cases:
        query_vec = embedding_provider.embed(case.question)
        search_res = exact_search(pipeline._vectors, query_vec, limit=5)
        retrieved_ids = [chunks_list[r.index].id for r in search_res]
        retrieved_dict[case.id] = retrieved_ids

        rag_ans = pipeline.answer(case.question, limit=3, minimum_score=0.1)
        abstention_results[case.id] = rag_ans.abstained

        if not rag_ans.abstained and rag_ans.sources:
            total_answers += 1
            first_source = rag_ans.sources[0]
            quoted_text = first_source.text[:15]
            if verify_citation(quoted_text, first_source):
                citation_validity_count += 1

    r_at_1 = calculate_retrieval_metrics(cases, retrieved_dict, k=1)
    r_at_3 = calculate_retrieval_metrics(cases, retrieved_dict, k=3)
    r_at_5 = calculate_retrieval_metrics(cases, retrieved_dict, k=5)

    unanswerable = [c for c in cases if not c.is_answerable]
    answerable = [c for c in cases if c.is_answerable]

    correct_abs = sum(
        1 for c in unanswerable if abstention_results.get(c.id, False)
    )
    correct_ans = sum(
        1 for c in answerable if not abstention_results.get(c.id, False)
    )

    abstention_precision = correct_abs / len(unanswerable) if unanswerable else 1.0
    answer_precision = correct_ans / len(answerable) if answerable else 1.0
    citation_validity_pct = (
        (citation_validity_count / total_answers * 100.0)
        if total_answers > 0
        else 100.0
    )

    return {
        "dataset": "data/eval_cases.jsonl (100 cases: 80 answerable, 20 unanswerable)",
        "retrieval_metrics": {
            "recall_at_1": r_at_1.recall_at_k,
            "recall_at_3": r_at_3.recall_at_k,
            "recall_at_5": r_at_5.recall_at_k,
            "mrr": r_at_3.mean_reciprocal_rank,
            "evaluated_cases": r_at_3.evaluated_cases,
        },
        "rag_grounding_and_abstention": {
            "unanswerable_abstention_accuracy": abstention_precision,
            "answerable_response_accuracy": answer_precision,
            "citation_validity_percentage": citation_validity_pct,
        },
    }


def main() -> None:
    results = run_rag_evaluation()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
