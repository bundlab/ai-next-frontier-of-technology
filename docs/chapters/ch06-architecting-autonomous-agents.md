# Chapter 06: LLM Evaluation, Observability & Tracing

## Overview

Deploying Large Language Model (LLM) applications into production requires moving beyond subjective "vibe checks" toward rigorous, deterministic metrics and real-time observability. Non-deterministic generation, variable latency, and dynamic context retrieval make debugging LLM pipelines fundamentally different from traditional software systems.

This chapter details key evaluation techniques—focusing on **LLM-as-a-Judge**, **RAG Triad metrics** (Faithfulness, Answer Relevance, Context Relevance), telemetry via **OpenTelemetry tracing**, and real-time operational monitoring for cost, latency, and token consumption.

---

## Core Concepts

### 1. The RAG Evaluation Triad

Evaluating Retrieval-Augmented Generation systems requires decoupling retrieval performance from generation quality. The RAG Triad addresses this by measuring three core relationships:

| Metric | Measured Target | Core Question | Failure Mode |
| :--- | :--- | :--- | :--- |
| **Context Relevance** | `Query` $\to$ `Retrieved Context` | Is the retrieved information relevant to the user's question? | Noise in retrieval, vector search mismatch, high token costs. |
| **Faithfulness (Groundedness)** | `Retrieved Context` $\to$ `Generated Answer` | Is the answer strictly grounded *only* in the retrieved context? | Hallucination, model introducing unverified external knowledge. |
| **Answer Relevance** | `Query` $\to$ `Generated Answer` | Does the generated answer directly address the user's input? | Irrelevant digressions, incomplete answers, evasion. |

### 2. LLM-as-a-Judge Architecture

Using fine-tuned or larger LLMs (e.g., GPT-4o, Claude 3.5 Sonnet) as automated judges allows teams to score subjective qualities (e.g., tone, hallucination, safety) at scale:

- **Structured Output Schemas:** Force judges to output JSON with explicit numerical scores ($0.0 - 1.0$) alongside step-by-step chain-of-thought rationale.
- **Reference-Based vs. Reference-Free:** Reference-free scoring evaluates quality purely on context/prompt alignment; reference-based compares against human ground-truth answers.
- **Position & Consensus Bias:** Mitigate judge bias by swapping prompt order or running multi-judge consensus loops.

### 3. Distributed Tracing & Telemetry

Observability in AI applications requires full execution visibility across multi-step chains, tool invocations, and retriever calls:

- **Spans & Traces:** Wrap individual LLM calls, vector DB lookups, and Python agent tools into hierarchical OpenTelemetry spans.
- **Token Tracking:** Record prompt tokens, completion tokens, and estimated cost per span.
- **Latency Breakdown:** Isolate network transport time from time-to-first-token (TTFT) and token generation speed.

---

## Implementation & Code Patterns

Below is a complete Python module demonstrating an automated evaluation pipeline incorporating **RAG Triad scoring** and **LLM-as-a-Judge validation** using Pydantic.

```python
#!/usr/bin/env python3
"""
Chapter 06: LLM Evaluation, Observability & Tracing
Example: RAG Triad Evaluator & LLM-as-a-Judge Framework
"""

import json
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EvalPipeline")


# ==============================================================================
# 1. Evaluation Data Models
# ==============================================================================

class RAGPipelineSample(BaseModel):
    sample_id: str
    user_query: str
    retrieved_contexts: List[str]
    generated_answer: str


class MetricScore(BaseModel):
    metric_name: str
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class EvaluationResult(BaseModel):
    sample_id: str
    faithfulness: MetricScore
    answer_relevance: MetricScore
    context_relevance: MetricScore
    overall_score: float


# ==============================================================================
# 2. Mock LLM-as-a-Judge Engine
# ==============================================================================

class LLMJudgeEngine:
    """Simulates structured evaluation using LLM-as-a-Judge prompts."""

    def evaluate_faithfulness(self, context: str, answer: str) -> MetricScore:
        """Checks if the answer is strictly derived from the context."""
        logger.info("Evaluating Faithfulness...")
        
        # Heuristic check simulating LLM judgment logic
        context_words = set(context.lower().split())
        answer_words = answer.lower().split()
        grounded_count = sum(1 for word in answer_words if word in context_words)
        ratio = min(1.0, grounded_count / max(1, len(answer_words)))

        if ratio > 0.6:
            return MetricScore(
                metric_name="Faithfulness",
                score=0.95,
                reasoning="All claims in the generated response are backed by retrieved context.",
            )
        return MetricScore(
            metric_name="Faithfulness",
            score=0.40,
            reasoning="Answer contains assertions not supported by the provided context.",
        )

    def evaluate_answer_relevance(self, query: str, answer: str) -> MetricScore:
        """Checks if the answer directly addresses the query."""
        logger.info("Evaluating Answer Relevance...")
        
        if len(answer.strip()) > 10 and ("rag" in answer.lower() or "eval" in answer.lower()):
            return MetricScore(
                metric_name="Answer Relevance",
                score=0.90,
                reasoning="Response directly answers the query with target domain concepts.",
            )
        return MetricScore(
            metric_name="Answer Relevance",
            score=0.50,
            reasoning="Response is vague or partially misses the core prompt intent.",
        )

    def evaluate_context_relevance(self, query: str, contexts: List[str]) -> MetricScore:
        """Checks if retrieved context chunks contain information needed for query."""
        logger.info("Evaluating Context Relevance...")
        combined_context = " ".join(contexts).lower()
        
        if "eval" in combined_context or "metrics" in combined_context:
            return MetricScore(
                metric_name="Context Relevance",
                score=0.88,
                reasoning="Retrieved passages contain high-density target keywords.",
            )
        return MetricScore(
            metric_name="Context Relevance",
            score=0.30,
            reasoning="Retrieved context contains high amounts of irrelevant noise.",
        )


# ==============================================================================
# 3. Evaluator Pipeline Runner
# ==============================================================================

class RAGEvaluationPipeline:
    """Orchestrates RAG Triad evaluation metrics across pipeline samples."""

    def __init__(self):
        self.judge = LLMJudgeEngine()

    def evaluate_sample(self, sample: RAGPipelineSample) -> EvaluationResult:
        logger.info(f"Running evaluation on sample ID: {sample.sample_id}")
        combined_context = " ".join(sample.retrieved_contexts)

        faithfulness = self.judge.evaluate_faithfulness(
            context=combined_context, answer=sample.generated_answer
        )
        answer_rel = self.judge.evaluate_answer_relevance(
            query=sample.user_query, answer=sample.generated_answer
        )
        context_rel = self.judge.evaluate_context_relevance(
            query=sample.user_query, contexts=sample.retrieved_contexts
        )

        overall = round(
            (faithfulness.score + answer_rel.score + context_rel.score) / 3.0, 2
        )

        return EvaluationResult(
            sample_id=sample.sample_id,
            faithfulness=faithfulness,
            answer_relevance=answer_rel,
            context_relevance=context_rel,
            overall_score=overall,
        )


# ==============================================================================
# 4. Execution Loop
# ==============================================================================

def main() -> None:
    print("\n--- Running Chapter 06: RAG Triad & LLM Evaluation Engine ---\n")

    sample = RAGPipelineSample(
        sample_id="eval_001",
        user_query="How do we measure context relevance in RAG evaluation?",
        retrieved_contexts=[
            "RAG evaluation metrics measure three pillars: faithfulness, answer relevance, and context relevance.",
            "Context relevance checks whether retrieved context chunks contain signal needed to answer the prompt.",
        ],
        generated_answer="Context relevance evaluates if the retrieved contexts contain the signal required to answer the query effectively.",
    )

    pipeline = RAGEvaluationPipeline()
    result = pipeline.evaluate_sample(sample)

    print("\n=======================================================")
    print("EVALUATION PIPELINE REPORT")
    print("=======================================================")
    print(f"Sample ID:        {result.sample_id}")
    print(f"Overall Score:    {result.overall_score} / 1.0\n")
    print(f"- {result.faithfulness.metric_name}: {result.faithfulness.score}")
    print(f"  Reason: {result.faithfulness.reasoning}\n")
    print(f"- {result.answer_relevance.metric_name}: {result.answer_relevance.score}")
    print(f"  Reason: {result.answer_relevance.reasoning}\n")
    print(f"- {result.context_relevance.metric_name}: {result.context_relevance.score}")
    print(f"  Reason: {result.context_relevance.reasoning}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()

```

---

## System Design & Production Best Practices

1. **Continuous Evaluation (CI/CD Integration):** Run regression evaluation suites against benchmark datasets on every model prompt update or retriever parameter change.
2. **Asynchronous Tracing:** Collect spans asynchronously (e.g., OpenTelemetry / OpenInference background workers) to prevent telemetry overhead from inflating request latency.
3. **Monitor Token Cost & Latency:** Set alert thresholds for token count spikes, high time-to-first-token (TTFT), and model response drift in live environments.

---

## Summary & Key Takeaways

* **Decouple Retrieval & Generation:** Evaluate context quality separately from generator groundedness to isolate failure modes quickly.
* **Automate with LLM-as-a-Judge:** Use structured Pydantic schemas and chain-of-thought prompting to convert qualitative LLM evaluation into actionable quantitative scores.
* **Instrument Every Step:** End-to-end tracing across vector retrievers, prompt builders, and model calls is non-negotiable for operating resilient production AI systems.

