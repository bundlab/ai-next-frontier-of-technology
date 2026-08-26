#!/usr/bin/env python3
"""
Chapter 04: Context Engineering & RAG Systems
Example: Hybrid RAG Pipeline & Semantic Search

This executable script demonstrates text document chunking, pseudo-vector
embedding generation, cosine similarity computation, and top-k context retrieval.
"""

import logging
import math
import re
from typing import Any

from pydantic import BaseModel, Field

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RAGPipeline")


# ==============================================================================
# 1. Domain Models & Data Structures
# ==============================================================================


class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    score: float


# ==============================================================================
# 2. Mock Embedding & Vector Engine
# ==============================================================================


class SimpleEmbeddingEngine:
    """Generates deterministic pseudo-embeddings for demonstration."""

    def embed_text(self, text: str, dim: int = 16) -> list[float]:
        """Maps token frequencies into a normalized dense vector."""
        tokens = re.findall(r"\w+", text.lower())
        vec = [0.0] * dim
        for token in tokens:
            idx = sum(ord(c) for c in token) % dim
            vec[idx] += 1.0

        # L2 Normalization
        magnitude = math.sqrt(sum(v * v for v in vec))
        if magnitude == 0:
            return vec
        return [round(v / magnitude, 4) for v in vec]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Computes cosine similarity between two normalized vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    return round(dot_product, 4)


# ==============================================================================
# 3. Document Chunking & RAG Retriever
# ==============================================================================


class RAGRetriever:
    """Handles text chunking, indexing, and semantic retrieval."""

    def __init__(self, chunk_size: int = 120, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedder = SimpleEmbeddingEngine()
        self.vector_store: list[DocumentChunk] = []

    def chunk_document(self, doc_id: str, text: str) -> list[DocumentChunk]:
        """Splits document text into overlapping chunks."""
        words = text.split()
        chunks: list[DocumentChunk] = []
        start = 0
        chunk_idx = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_text = " ".join(words[start:end])
            embedding = self.embedder.embed_text(chunk_text)

            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                content=chunk_text,
                metadata={
                    "start_word": start,
                    "end_word": min(end, len(words)),
                },
                embedding=embedding,
            )
            chunks.append(chunk)
            chunk_idx += 1
            start += self.chunk_size - self.overlap

        return chunks

    def index_document(self, doc_id: str, text: str) -> None:
        """Chunks and stores document vectors in memory."""
        chunks = self.chunk_document(doc_id, text)
        self.vector_store.extend(chunks)
        logger.info(f"Indexed document '{doc_id}' into {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = 2) -> list[RetrievalResult]:
        """Searches vector store for top_k most similar chunks."""
        query_vec = self.embedder.embed_text(query)
        results: list[RetrievalResult] = []

        for chunk in self.vector_store:
            score = cosine_similarity(query_vec, chunk.embedding)
            results.append(RetrievalResult(chunk=chunk, score=score))

        # Sort descending by similarity score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# ==============================================================================
# 4. Execution Loop
# ==============================================================================


def main() -> None:
    print("\n--- Running Chapter 04: Hybrid RAG Pipeline Example ---\n")

    knowledge_base = (
        "Retrieval-Augmented Generation (RAG) improves LLM responses by"
        " injecting external knowledge into the prompt context window."
        " Modern RAG systems combine dense vector search with sparse keyword"
        " indexes like BM25 to build a hybrid retrieval engine."
        " Cross-encoders are frequently used as a re-ranking stage to filter"
        " top context chunks."
    )

    retriever = RAGRetriever(chunk_size=20, overlap=5)
    retriever.index_document(doc_id="arch_doc_01", text=knowledge_base)

    query = "How do cross-encoders and hybrid search work in RAG?"
    logger.info(f"Executing Query: '{query}'")

    results = retriever.search(query=query, top_k=2)

    print("\n=======================================================")
    print("RETRIEVED CONTEXT CHUNKS")
    print("=======================================================")
    for rank, res in enumerate(results, start=1):
        print(f"Rank {rank} (Score: {res.score}) [{res.chunk.chunk_id}]:")
        print(f'  "{res.chunk.content}"\n')
    print("=======================================================\n")


if __name__ == "__main__":
    main()
