"""A small, dependency-injected retrieval-augmented generation pipeline."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from vectordb_rag.search import exact_search

FloatVector = NDArray[np.float64]


@dataclass(frozen=True)
class Chunk:
    """A searchable piece of source text."""

    id: str
    text: str


@dataclass(frozen=True)
class RagAnswer:
    """Generated answer together with the retrieved supporting chunks."""

    text: str
    sources: list[Chunk]
    abstained: bool = False


class EmbeddingProvider(Protocol):
    """Turns text into vectors in one shared vector space."""

    def embed(self, text: str) -> FloatVector:
        """Embed one string."""


class AnswerProvider(Protocol):
    """Generates an answer from a question and retrieved source text."""

    def answer(self, question: str, contexts: list[str]) -> str:
        """Answer using only the supplied contexts."""


def chunk_text(document_id: str, text: str, words_per_chunk: int = 80) -> list[Chunk]:
    """Split text into fixed-size word chunks with stable source identifiers."""
    if words_per_chunk < 1:
        raise ValueError("words_per_chunk must be at least 1")

    words = text.split()
    return [
        Chunk(
            id=f"{document_id}:{offset // words_per_chunk}",
            text=" ".join(words[offset : offset + words_per_chunk]),
        )
        for offset in range(0, len(words), words_per_chunk)
    ]


class RagPipeline:
    """Retrieve the closest chunks and hand them to an answer provider."""

    def __init__(
        self,
        chunks: list[Chunk],
        embedding_provider: EmbeddingProvider,
        answer_provider: AnswerProvider,
    ) -> None:
        if not chunks:
            raise ValueError("at least one chunk is required")

        self._chunks = chunks
        self._embedding_provider = embedding_provider
        self._answer_provider = answer_provider
        chunk_vectors = [embedding_provider.embed(chunk.text) for chunk in chunks]
        self._vectors = np.vstack(chunk_vectors)

    def answer(
        self, question: str, limit: int = 3, minimum_score: float = 0.2
    ) -> RagAnswer:
        """Answer one question and return the source chunks used as context."""
        results = exact_search(
            self._vectors,
            self._embedding_provider.embed(question),
            limit,
        )
        if not results or results[0].score < minimum_score:
            return RagAnswer(
                text="I do not have enough supporting context to answer that.",
                sources=[],
                abstained=True,
            )
        sources = [self._chunks[result.index] for result in results]
        text = self._answer_provider.answer(question, [chunk.text for chunk in sources])
        return RagAnswer(text=text, sources=sources)


def verify_citation(quoted_span: str, source: Chunk) -> bool:
    """Confirm that a citation quote occurs verbatim in its source chunk."""
    return bool(quoted_span) and quoted_span in source.text
