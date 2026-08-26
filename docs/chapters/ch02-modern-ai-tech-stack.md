# Chapter 02: The Modern AI Runtime & Tech Stack

## Overview

Architecting modern AI applications requires moving beyond isolated API calls into building a multi-layered ecosystem. The modern AI tech stack coordinates foundation models, high-performance vector databases, orchestration layers, context retrieval pipelines, and runtime execution environments.

This chapter breaks down the key components of the modern AI software architecture, detailing how model orchestration, structured outputs, memory management, and asynchronous execution interface with traditional software infrastructure.

---

## Core Concepts

### 1. The 5-Layer AI Architecture Stack

Modern production AI systems are structured into five distinct architectural layers:

| Layer | Component | Function & Responsibilities |
| :--- | :--- | :--- |
| **Layer 5: Interface** | Web / Mobile / API | User interaction, streaming UI components, client-side state. |
| **Layer 4: Orchestration** | Frameworks & Routers | Agent execution loops, prompt templates, tool binding, state persistence. |
| **Layer 3: Memory & Context** | Vector Stores & Caches | Semantic indexing, document embeddings, hybrid search, session memory. |
| **Layer 2: Evaluation & Ops** | Telemetry & Guardrails | Tracing, cost tracking, hallucination checks, rate limiting, output parsing. |
| **Layer 1: Inference Runtime** | LLMs / SLMs / APIs | Open-weights model execution, cloud provider endpoints, edge models. |

### 2. Model Routing & Execution Tiering
Rather than using a single model for all tasks, production systems implement a **Model Router**. Requests are dynamically routed based on task complexity, context length, latency constraints, and operational cost:


```

+-------------------------------------------------------------------------+
|                              USER REQUEST                               |
+-------------------------------------------------------------------------+
|
v
+-------------------------------------------------------------------------+
|                              MODEL ROUTER                               |
|                  (Analyzes intent, SLA & cost budget)                   |
+-------------------------------------------------------------------------+
|                                |                               |
| Simple classification          | Complex reasoning             | Fast / Offline
v                                v                               v
+--------------------+        +-------------------+           +-------------------+
|  Small Model (SLM) |        |  Large Model      |           |  Local / Edge     |
|  (e.g., Llama 3B)  |        |  (e.g., GPT-4o)   |           |  Runtime (Ollama) |
+--------------------+        +-------------------+           +-------------------+

```

---

## Implementation & Code Patterns

Below is a production-ready Python pattern demonstrating an **Asynchronous Model Router with Structured Routing Contracts**. It dynamically evaluates request complexity and dispatches prompts to the optimal runtime tier.

```python
#!/usr/bin/env python3
"""
Chapter 02: Modern AI Tech Stack
Example: Dynamic Model Router Pattern
"""

import asyncio
import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TechStackRouter")


# 1. Define Execution Tiers
class ModelTier(str, Enum):
    FAST_LOCAL = "fast_local"       # Edge/Local SLMs for low latency & zero cost
    BALANCED = "balanced"           # Mid-tier models for standard processing
    HEAVY_REASONING = "heavy_reason" # Frontier LLMs for complex reasoning


# 2. Domain Models for Routing Logic
class RoutingDecision(BaseModel):
    selected_tier: ModelTier
    rationale: str
    estimated_tokens: int = Field(..., ge=1)


class CompletionResult(BaseModel):
    prompt: str
    tier_used: ModelTier
    response: str
    latency_ms: float


# 3. Model Router Engine
class ModelRouter:
    """Evaluates prompts and dispatches execution to appropriate model runtimes."""

    def __init__(self):
        self._mock_runtimes = {
            ModelTier.FAST_LOCAL: "Ollama / Llama-3-8B Local",
            ModelTier.BALANCED: "Standard Cloud LLM API",
            ModelTier.HEAVY_REASONING: "Frontier Reasoning Model API",
        }

    def evaluate_route(self, prompt: str) -> RoutingDecision:
        """Determines the execution tier based on prompt length and complexity keywords."""
        token_estimate = len(prompt.split()) * 2
        prompt_lower = prompt.lower()

        complex_keywords = ["architect", "analyze", "refactor", "algorithm", "design pattern"]
        
        if any(keyword in prompt_lower for keyword in complex_keywords) or token_estimate > 500:
            return RoutingDecision(
                selected_tier=ModelTier.HEAVY_REASONING,
                rationale="Prompt requires complex system design or deep architectural reasoning.",
                estimated_tokens=token_estimate,
            )
        elif len(prompt) < 100:
            return RoutingDecision(
                selected_tier=ModelTier.FAST_LOCAL,
                rationale="Short, straightforward query. Routing to local edge runtime.",
                estimated_tokens=token_estimate,
            )
        else:
            return RoutingDecision(
                selected_tier=ModelTier.BALANCED,
                rationale="Standard query complexity. Routing to mid-tier cloud endpoint.",
                estimated_tokens=token_estimate,
            )

    async def execute(self, prompt: str) -> CompletionResult:
        """Asynchronously dispatches the prompt to the selected model tier."""
        start_time = asyncio.get_event_loop().time()
        decision = self.evaluate_route(prompt)

        logger.info(
            f"Routing prompt to [{decision.selected_tier.value.upper()}] | Rationale: {decision.rationale}"
        )

        # Simulate tier-specific latency
        if decision.selected_tier == ModelTier.FAST_LOCAL:
            await asyncio.sleep(0.05)
            response_text = f"[Local SLM Output] Processed short task using {self._mock_runtimes[decision.selected_tier]}."
        elif decision.selected_tier == ModelTier.BALANCED:
            await asyncio.sleep(0.20)
            response_text = f"[Cloud LLM Output] Completed task via {self._mock_runtimes[decision.selected_tier]}."
        else:
            await asyncio.sleep(0.50)
            response_text = f"[Frontier LLM Output] Deep reasoning complete via {self._mock_runtimes[decision.selected_tier]}."

        elapsed_ms = round((asyncio.get_event_loop().time() - start_time) * 1000, 2)

        return CompletionResult(
            prompt=prompt,
            tier_used=decision.selected_tier,
            response=response_text,
            latency_ms=elapsed_ms,
        )


# --- Execution Example ---
async def main():
    router = ModelRouter()

    prompts = [
        "What is the CLI command to check disk space on Linux?",
        "Write a standard README description for an open-source project.",
        "Design a distributed saga pattern for multi-region microservices with database fallbacks.",
    ]

    print("\n--- Running Chapter 02: Tech Stack Model Router Example ---\n")
    
    for idx, prompt in enumerate(prompts, start=1):
        logger.info(f"\nProcessing Prompt #{idx}: '{prompt[:40]}...'")
        result = await router.execute(prompt)
        print(f"  Result Response : {result.response}")
        print(f"  Latency Taken   : {result.latency_ms} ms\n")


if __name__ == "__main__":
    asyncio.run(main())

```

---

## Architecture & System Design Guidelines

1. **Decouple Provider SDKs Behind Standard Interfaces:** Never couple application logic directly to a single provider's proprietary client library. Use abstraction layers (such as LiteLLM, LangChain, or custom interfaces) to enable seamless model swapping.
2. **Implement Hybrid Vector & Keyword Indexing:** Vector embeddings capture semantic intent but struggle with exact matches (e.g., serial numbers, SKU codes, exact function names). Always pair vector indices with full-text search engines (BM25/Postgres tsvector).
3. **Architect for Observability First:** Ensure every LLM call emits structured telemetry: input/output token counts, latency breakdowns, model identifiers, prompt versions, and trace IDs.

---

## Summary & Key Takeaways

* **Modular Stack Architecture:** Build systems with clear separation between UI, orchestration, memory, telemetry, and model execution layers.
* **Dynamic Model Routing:** Optimize performance and cost by dynamically matching task complexity to the cheapest and fastest viable model tier.
* **Standardized Interfaces:** Insulate application code from model provider volatility using abstract orchestration contracts.

---