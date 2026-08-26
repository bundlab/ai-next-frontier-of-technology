#!/usr/bin/env python3
"""
Chapter 06: LLM Evaluation, Observability & Tracing
Example: RAG Triad Evaluator & LLM-as-a-Judge Framework

This executable script demonstrates automated RAG evaluation using the RAG Triad
metrics (Faithfulness, Answer Relevance, and Context Relevance) with structured
Pydantic scoring outputs.
"""

import logging

from pydantic import BaseModel, Field

# Configure structured logging
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
    retrieved_contexts: list[str]
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
    """Simulates structured evaluation using LLM-as-a-Judge logic."""

    def evaluate_faithfulness(self, context: str, answer: str) -> MetricScore:
        """Checks if the answer is strictly derived from the retrieved context."""
        logger.info("Evaluating Faithfulness...")

        context_words = set(context.lower().split())
        answer_words = answer.lower().split()
        grounded_count = sum(1 for word in answer_words if word in context_words)
        ratio = grounded_count / max(1, len(answer_words))

        if ratio > 0.5:
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
        """Checks if the answer directly addresses the user query."""
        logger.info("Evaluating Answer Relevance...")

        if len(answer.strip()) > 10 and (
            "rag" in answer.lower() or "context" in answer.lower()
        ):
            return MetricScore(
                metric_name="Answer Relevance",
                score=0.90,
                reasoning="Response directly answers the query using relevant domain concepts.",
            )
        return MetricScore(
            metric_name="Answer Relevance",
            score=0.50,
            reasoning="Response is vague or misses the core prompt intent.",
        )

    def evaluate_context_relevance(
        self, query: str, contexts: list[str]
    ) -> MetricScore:
        """Checks if retrieved context chunks contain information needed for the query."""
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
            reasoning="Retrieved context contains excessive noise.",
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
