# Chapter 04: Context Engineering & Retrieval-Augmented Generation (RAG)

## Overview

Large Language Models (LLMs) are bound by fixed knowledge cutoffs and context window constraints. To build production systems capable of querying private enterprise data, real-time metrics, and domain-specific knowledge bases, system architects employ **Retrieval-Augmented Generation (RAG)**.

RAG bridges parametric memory (model weights) and non-parametric memory (external vector indexes, databases, and document stores). Rather than retraining or fine-tuning models, context engineering dynamically constructs the optimal prompt context window at runtime, maximizing answer accuracy while minimizing retrieval noise and hallucination.

This chapter details the architecture of modern RAG pipelines, chunking strategies, vector embedding retrieval, hybrid keyword-semantic search, and context re-ranking patterns.

---

## Core Concepts

### 1. The Production RAG Pipeline Architecture

A production-grade RAG system decouples ingestion from query-time retrieval and generation:


```

[ Ingestion Pipeline ]
Documents -> Structural Chunking -> Embedding Model -> Vector Store Index
|
[ Query Execution Pipeline ]                                 v
User Query -> Query Expansion -> Hybrid Retrieval (Vector + BM25)
|
v
Context Re-Ranking (Cross-Encoder)
|
v
LLM Generation (Prompt Assembly)

```

### 2. Chunking & Overlap Strategies

How documents are segmented directly impacts retrieval accuracy:

| Chunking Strategy | Description | Best Used For |
| :--- | :--- | :--- |
| **Fixed-Size (Token/Char)** | Hard limit on character or token counts with fixed overlap. | Quick prototypes, uniform text blocks. |
| **Recursive Character** | Splitting along natural language boundaries (`\n\n`, `\n`, `.`, ` `). | Technical documentation, markdown, articles. |
| **Semantic Chunking** | Splitting based on embedding distance shifts between adjacent sentences. | Unstructured books, transcribed audio, essays. |
| **Hierarchical / Parent-Child** | Small chunks for precise vector matching linked to larger parent chunks for LLM context. | Complex enterprise policy manuals and PDFs. |

### 3. Hybrid Search and Cross-Encoder Re-Ranking

Dense vector retrieval (cosine similarity over embeddings) excels at capturing high-level semantic intent but frequently misses exact keyword matches (such as product IDs, code symbols, or error codes). Production systems pair dense retrieval with sparse keyword search (BM25):

$$\text{Hybrid Score} = \alpha \cdot \text{Dense Score} + (1 - \alpha) \cdot \text{Sparse Score}$$

Following initial hybrid retrieval (e.g., fetching top 50 candidates), a **Cross-Encoder Re-ranker** scores the query jointly against each candidate chunk to return the top $N$ most relevant context items, drastically reducing context window noise.

---

## Implementation & Code Patterns

Below is a self-contained Python implementation of a **Hybrid RAG Pipeline with Document Chunking and Cosine Similarity Retrieval**.

```python
#!/usr/bin/env python3
"""
Chapter 04: Context Engineering & RAG Systems
Example: Hybrid RAG Pipeline & Semantic Search
"""

import math
import re
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field

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
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: List[float] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    score: float


# ==============================================================================
# 2. Mock Embedding & Vector Engine
# ==============================================================================

class SimpleEmbeddingEngine:
    """Generates deterministic pseudo-embeddings for demonstration."""
    
    def embed_text(self, text: str, dim: int = 16) -> List[float]:
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


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two normalized vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
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
        self.vector_store: List[DocumentChunk] = []

    def chunk_document(self, doc_id: str, text: str) -> List[DocumentChunk]:
        """Splits document text into overlapping chunks."""
        words = text.split()
        chunks: List[DocumentChunk] = []
        start = 0
        chunk_idx = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_text = " ".join(words[start:end])
            embedding = self.embedder.embed_text(chunk_text)
            
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                content=chunk_text,
                metadata={"start_word": start, "end_word": min(end, len(words))},
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

    def search(self, query: str, top_k: int = 2) -> List[RetrievalResult]:
        """Searches vector store for top_k most similar chunks."""
        query_vec = self.embedder.embed_text(query)
        results: List[RetrievalResult] = []

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
        "Retrieval-Augmented Generation (RAG) improves LLM responses by injecting external "
        "knowledge into the prompt context window. Modern RAG systems combine dense vector search "
        "with sparse keyword indexes like BM25 to build a hybrid retrieval engine. "
        "Cross-encoders are frequently used as a re-ranking stage to filter top context chunks."
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
        print(f"  \"{res.chunk.content}\"\n")
    print("=======================================================\n")


if __name__ == "__main__":
    main()

```

---

## Architecture & System Design Guidelines

1. **Avoid Over-Chunking:** Extremely small chunks (e.g., under 50 words) destroy context coherence, while massive chunks (e.g., over 2,000 words) dilute embedding specificity. Target 250–500 tokens for general text.
2. **Metadata Enrichment:** Store structural document metadata (author, document type, access permissions, creation timestamp) alongside vectors to enable metadata filtering prior to vector similarity computation.
3. **Monitor Lost-in-the-Middle Phenomenon:** LLMs pay most attention to tokens at the very beginning and very end of their context window. Position the most relevant retrieved context at the top or bottom of your assembled prompt rather than buried in the middle.

---

## Summary & Key Takeaways

* **RAG Extends Knowledge Safely:** Provides models with up-to-date and domain-specific knowledge without expensive model fine-tuning.
* **Hybrid Retrieval is Mandatory:** Combining vector search with BM25 sparse keyword search balances high-level semantic intent with exact term match accuracy.
* **Context Quality over Quantity:** Use re-ranking mechanisms to inject only high-precision chunks into the model's context window.

