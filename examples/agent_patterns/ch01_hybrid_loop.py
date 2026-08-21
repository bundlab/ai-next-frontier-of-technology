#!/usr/bin/env python3
"""
Chapter 01: The AI-First Architecture Paradigm
Example: Hybrid Control Loop Pattern

This executable script demonstrates wrapping a probabilistic AI/LLM decision engine
inside a deterministic boundary using Pydantic schema validation, automatic retries,
and a safe fallback path.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HybridControlLoop")


# ==============================================================================
# 1. Deterministic Domain Model / Output Schema
# ==============================================================================


class TaskStep(BaseModel):
    step_id: int = Field(..., ge=1, description="Sequential step index")
    action: str = Field(..., min_length=3, description="Action to perform")
    target_resource: str = Field(..., description="Affected system or resource")


class TaskExecutionPlan(BaseModel):
    task_name: str = Field(..., min_length=3, description="Name of the task")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence rating"
    )
    requires_human_approval: bool = Field(
        default=False, description="Manual gate trigger"
    )
    steps: List[TaskStep] = Field(
        ..., min_items=1, description="List of execution steps"
    )


# ==============================================================================
# 2. Simulated Probabilistic Engine (Model Provider)
# ==============================================================================


class SimulatedLLMProvider:
    """
    Simulates a non-deterministic LLM output, including malformed responses
    to demonstrate error recovery within the deterministic wrapper.
    """

    def __init__(self):
        self._call_count = 0

    def generate_completion(self, prompt: str) -> str:
        self._call_count += 1
        logger.info(
            f"LLM API Invoked (Call #{self._call_count}) with prompt: '{prompt}'"
        )

        # Simulate initial failures to demonstrate schema validation & recovery
        if self._call_count == 1:
            logger.warning(
                "Simulating LLM output with invalid types (string instead of float for confidence)..."
            )
            return json.dumps(
                {
                    "task_name": "Database Schema Migration",
                    "confidence_score": "INVALID_HIGH_CONFIDENCE",  # Causes Pydantic ValidationError
                    "requires_human_approval": False,
                    "steps": [
                        {
                            "step_id": 1,
                            "action": "Run alembic upgrade head",
                            "target_resource": "PostgreSQL",
                        }
                    ],
                }
            )

        elif self._call_count == 2:
            logger.warning("Simulating LLM output with missing required fields...")
            return json.dumps(
                {
                    "task_name": "Database Schema Migration",
                    # Missing steps key causes validation failure
                    "confidence_score": 0.88,
                }
            )

        # Successful deterministic match
        logger.info("Simulating valid structural JSON completion...")
        return json.dumps(
            {
                "task_name": "Database Schema Migration",
                "confidence_score": 0.95,
                "requires_human_approval": False,
                "steps": [
                    {
                        "step_id": 1,
                        "action": "Create read-only replica snapshot",
                        "target_resource": "RDS Cluster",
                    },
                    {
                        "step_id": 2,
                        "action": "Execute non-blocking schema migration",
                        "target_resource": "PostgreSQL Primary",
                    },
                    {
                        "step_id": 3,
                        "action": "Validate table index integrity",
                        "target_resource": "PostgreSQL Primary",
                    },
                ],
            }
        )


# ==============================================================================
# 3. Hybrid Orchestrator
# ==============================================================================


class HybridOrchestrator:
    """
    Wraps the LLM provider within strict validation rules, backoff logic,
    and deterministic fallback execution paths.
    """

    def __init__(self, llm_provider: SimulatedLLMProvider, max_retries: int = 3):
        self.llm_provider = llm_provider
        self.max_retries = max_retries

    def execute_task(self, prompt: str) -> TaskExecutionPlan:
        """Executes probabilistic completion inside deterministic safety boundaries."""
        for attempt in range(1, self.max_retries + 1):
            logger.info(
                f"Attempt {attempt}/{self.max_retries}: Requesting execution plan..."
            )

            raw_response = self.llm_provider.generate_completion(prompt)

            try:
                # Step A: Parse raw string into JSON structure
                parsed_json = json.loads(raw_response)

                # Step B: Enforce schema validation via Pydantic
                validated_plan = TaskExecutionPlan(**parsed_json)
                logger.info("✅ Schema validation successful.")

                # Step C: Business logic evaluation
                if validated_plan.confidence_score < 0.70:
                    logger.warning(
                        "Confidence below threshold (0.70). Flagging for human approval."
                    )
                    validated_plan.requires_human_approval = True

                return validated_plan

            except json.JSONDecodeError as e:
                logger.error(
                    f"Attempt {attempt} failed: Invalid JSON format from model. ({e})"
                )
            except ValidationError as e:
                logger.error(
                    f"Attempt {attempt} failed: Schema validation error. ({e.error_count()} errors)"
                )

            # Backoff delay prior to retrying
            time.sleep(0.5)

        logger.critical(
            "All probabilistic generation retries exhausted. Initiating safe fallback logic."
        )
        return self._deterministic_fallback(prompt)

    def _deterministic_fallback(self, prompt: str) -> TaskExecutionPlan:
        """Safe execution path when model output repeatedly fails validation."""
        return TaskExecutionPlan(
            task_name=f"Manual Override: {prompt[:30]}...",
            confidence_score=0.0,
            requires_human_approval=True,
            steps=[
                TaskStep(
                    step_id=1,
                    action="Route prompt to human review queue",
                    target_resource="OpsDesk Ticket System",
                ),
                TaskStep(
                    step_id=2,
                    action="Log raw execution anomaly",
                    target_resource="Datadog Telemetry",
                ),
            ],
        )


# ==============================================================================
# 4. Entrypoint / Execution Loop
# ==============================================================================


def main():
    print("\n--- Running Chapter 01: Hybrid Control Loop Pattern Example ---\n")

    provider = SimulatedLLMProvider()
    orchestrator = HybridOrchestrator(llm_provider=provider, max_retries=3)

    user_prompt = "Generate a zero-downtime database migration plan for production."

    # Execute orchestrator loop
    final_plan = orchestrator.execute_task(user_prompt)

    # Print final structured results
    print("\n=======================================================")
    print("FINAL EXECUTED PLAN")
    print("=======================================================")
    print(f"Task Name               : {final_plan.task_name}")
    print(f"Model Confidence Score  : {final_plan.confidence_score}")
    print(f"Requires Human Approval : {final_plan.requires_human_approval}")
    print("Execution Steps         :")
    for step in final_plan.steps:
        print(f"  [{step.step_id}] {step.action} -> ({step.target_resource})")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
