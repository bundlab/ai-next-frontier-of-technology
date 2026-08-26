#!/usr/bin/env python3
"""
Chapter 02: The Modern AI Runtime & Tech Stack
Example: Dynamic Model Router Pattern

This executable script demonstrates an asynchronous Model Router that evaluates request
complexity and token estimates to dynamically dispatch prompts to the optimal runtime tier.
"""

import asyncio
import logging
import time
from enum import Enum

from pydantic import BaseModel, Field

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TechStackRouter")


# ==============================================================================
# 1. Execution Tiers & Domain Models
# ==============================================================================


class ModelTier(str, Enum):
    FAST_LOCAL = "fast_local"  # Edge/Local SLMs for low latency & zero cost
    BALANCED = "balanced"  # Mid-tier models for standard processing
    HEAVY_REASONING = "heavy_reason"  # Frontier LLMs for complex reasoning


class RoutingDecision(BaseModel):
    selected_tier: ModelTier
    rationale: str
    estimated_tokens: int = Field(..., ge=1)


class CompletionResult(BaseModel):
    prompt: str
    tier_used: ModelTier
    response: str
    latency_ms: float


# ==============================================================================
# 2. Model Router Engine
# ==============================================================================


class ModelRouter:
    """Evaluates prompts and dispatches execution to appropriate model runtimes."""

    def __init__(self) -> None:
        self._mock_runtimes = {
            ModelTier.FAST_LOCAL: "Ollama / Llama-3-8B Local",
            ModelTier.BALANCED: "Standard Cloud LLM API",
            ModelTier.HEAVY_REASONING: "Frontier Reasoning Model API",
        }

    def evaluate_route(self, prompt: str) -> RoutingDecision:
        """Determines execution tier based on prompt length and complexity keywords."""
        token_estimate = len(prompt.split()) * 2
        prompt_lower = prompt.lower()

        complex_keywords = [
            "architect",
            "analyze",
            "refactor",
            "algorithm",
            "design pattern",
        ]

        if (
            any(keyword in prompt_lower for keyword in complex_keywords)
            or token_estimate > 500
        ):
            return RoutingDecision(
                selected_tier=ModelTier.HEAVY_REASONING,
                rationale="Prompt requires deep reasoning or complex system design.",
                estimated_tokens=token_estimate,
            )
        elif len(prompt) < 100:
            return RoutingDecision(
                selected_tier=ModelTier.FAST_LOCAL,
                rationale="Short query. Routing to fast local edge runtime.",
                estimated_tokens=token_estimate,
            )
        else:
            return RoutingDecision(
                selected_tier=ModelTier.BALANCED,
                rationale="Standard query. Routing to mid-tier cloud endpoint.",
                estimated_tokens=token_estimate,
            )

    async def execute(self, prompt: str) -> CompletionResult:
        """Asynchronously dispatches prompt to selected model tier."""
        start_time = time.perf_counter()
        decision = self.evaluate_route(prompt)

        logger.info(
            f"Routing prompt to [{decision.selected_tier.value.upper()}] | Rationale: {decision.rationale}"
        )

        # Simulate tier-specific processing latency
        if decision.selected_tier == ModelTier.FAST_LOCAL:
            await asyncio.sleep(0.05)
            response_text = f"[Local SLM Output] Task completed via {self._mock_runtimes[decision.selected_tier]}."
        elif decision.selected_tier == ModelTier.BALANCED:
            await asyncio.sleep(0.20)
            response_text = f"[Cloud LLM Output] Task completed via {self._mock_runtimes[decision.selected_tier]}."
        else:
            await asyncio.sleep(0.50)
            response_text = f"[Frontier LLM Output] Reasoning complete via {self._mock_runtimes[decision.selected_tier]}."

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return CompletionResult(
            prompt=prompt,
            tier_used=decision.selected_tier,
            response=response_text,
            latency_ms=elapsed_ms,
        )


# ==============================================================================
# 3. Execution Loop
# ==============================================================================


async def main() -> None:
    router = ModelRouter()

    prompts = [
        "What is the CLI command to check disk space on Linux?",
        "Write a standard README description for an open-source repository.",
        "Design a distributed saga pattern for microservices with database fallbacks.",
    ]

    print("\n--- Running Chapter 02: Dynamic Model Router Pattern Example ---\n")

    for idx, prompt in enumerate(prompts, start=1):
        logger.info(f"Processing Prompt #{idx}: '{prompt[:45]}...'")
        result = await router.execute(prompt)
        print(f"  Selected Tier : {result.tier_used.value}")
        print(f"  Response      : {result.response}")
        print(f"  Latency       : {result.latency_ms} ms\n")


if __name__ == "__main__":
    asyncio.run(main())