"""End-to-end tests for chunking, retrieval, and deterministic answering."""

import numpy as np

from vectordb_rag.rag import AnswerProvider, Chunk, RagPipeline, chunk_text


class FakeEmbeddingProvider:
    """Map known strings to deterministic vectors without a real model."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed(self, text: str) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return np.array(self._vectors[text], dtype=np.float64)


class FakeAnswerProvider(AnswerProvider):
    """Return the first retrieved context so the test can inspect the path."""

    def answer(self, question: str, contexts: list[str]) -> str:
        return f"Answer: {contexts[0]}"


def test_chunk_text_assigns_stable_ids() -> None:
    chunks = chunk_text("guide", "one two three four five", words_per_chunk=2)

    assert chunks == [
        Chunk(id="guide:0", text="one two"),
        Chunk(id="guide:1", text="three four"),
        Chunk(id="guide:2", text="five"),
    ]


def test_rag_pipeline_retrieves_context_and_returns_its_source() -> None:
    chunks = [
        Chunk(id="facts:0", text="The library is open on Monday."),
        Chunk(id="facts:1", text="The cafe closes at six."),
    ]
    embeddings = FakeEmbeddingProvider(
        {
            chunks[0].text: [1.0, 0.0],
            chunks[1].text: [0.0, 1.0],
            "When is the library open?": [1.0, 0.0],
        }
    )
    pipeline = RagPipeline(chunks, embeddings, FakeAnswerProvider())

    result = pipeline.answer("When is the library open?", limit=1)

    assert result.text == "Answer: The library is open on Monday."
    assert result.sources == [chunks[0]]
