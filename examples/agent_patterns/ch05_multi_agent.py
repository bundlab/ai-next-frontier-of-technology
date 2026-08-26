#!/usr/bin/env python3
"""
Chapter 05: Multi-Agent Orchestration & Communication Patterns
Example: Supervisor-Specialist Agent Loop with State Handoff

This executable script demonstrates hierarchical multi-agent orchestration,
typed payload handoffs via Pydantic, state isolation, and execution bounds.
"""

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MultiAgentOrchestrator")


# ==============================================================================
# 1. Domain Models & Shared State
# ==============================================================================


class AgentRole(str, Enum):
    SUPERVISOR = "Supervisor"
    RESEARCHER = "Researcher"
    WRITER = "Writer"
    REVIEWER = "Reviewer"


class AgentTask(BaseModel):
    task_id: str
    target_role: AgentRole
    description: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    task_id: str
    source_role: AgentRole
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class OrchestratorState(BaseModel):
    user_query: str
    step_count: int = 0
    max_steps: int = 5
    task_history: list[AgentResponse] = Field(default_factory=list)
    final_output: str | None = None


# ==============================================================================
# 2. Specialist Agents
# ==============================================================================


class ResearchAgent:
    """Specialist agent focused on gathering domain facts."""

    role: AgentRole = AgentRole.RESEARCHER

    def execute(self, task: AgentTask) -> AgentResponse:
        logger.info(f"[{self.role.value}] Researching topic: {task.description}")
        query = task.payload.get("query", "")

        # Simulated research output payload
        facts = [
            f"Fact 1: Multi-agent systems decouple tasks for domain '{query}'.",
            "Fact 2: Shared blackboard patterns enable asynchronous peer communication.",
            "Fact 3: Explicit state handoffs reduce prompt context bloat.",
        ]
        return AgentResponse(
            task_id=task.task_id,
            source_role=self.role,
            success=True,
            data={"facts": facts},
        )


class WriterAgent:
    """Specialist agent focused on content synthesis."""

    role: AgentRole = AgentRole.WRITER

    def execute(self, task: AgentTask) -> AgentResponse:
        logger.info(f"[{self.role.value}] Synthesizing draft for task: {task.task_id}")
        facts = task.payload.get("facts", [])

        if not facts:
            return AgentResponse(
                task_id=task.task_id,
                source_role=self.role,
                success=False,
                error_message="No research facts provided to writer.",
            )

        draft = "Executive Summary:\n" + "\n".join(f"- {fact}" for fact in facts)
        return AgentResponse(
            task_id=task.task_id,
            source_role=self.role,
            success=True,
            data={"draft": draft},
        )


class ReviewerAgent:
    """Specialist agent focused on draft verification and quality checks."""

    role: AgentRole = AgentRole.REVIEWER

    def execute(self, task: AgentTask) -> AgentResponse:
        logger.info(
            f"[{self.role.value}] Auditing content quality for task: {task.task_id}"
        )
        draft = task.payload.get("draft", "")

        is_approved = len(draft.strip()) > 20
        feedback = "Approved for output." if is_approved else "Draft too concise."

        return AgentResponse(
            task_id=task.task_id,
            source_role=self.role,
            success=is_approved,
            data={"approved": is_approved, "feedback": feedback},
        )


# ==============================================================================
# 3. Supervisor Orchestrator
# ==============================================================================


class AgentSupervisor:
    """Central Orchestrator managing agent execution flow and shared state."""

    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps
        self.researcher = ResearchAgent()
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()

    def run(self, user_query: str) -> OrchestratorState:
        state = OrchestratorState(user_query=user_query, max_steps=self.max_steps)
        logger.info(f"[Supervisor] Starting workflow for: '{user_query}'")

        # Step 1: Dispatch Research Task
        if state.step_count >= state.max_steps:
            logger.warning("[Supervisor] Max steps limit reached before research.")
            return state

        state.step_count += 1
        res_task = AgentTask(
            task_id="task_01_research",
            target_role=AgentRole.RESEARCHER,
            description="Gather structural facts",
            payload={"query": user_query},
        )
        res_response = self.researcher.execute(res_task)
        state.task_history.append(res_response)

        if not res_response.success:
            logger.error("[Supervisor] Research phase failed. Aborting.")
            return state

        # Step 2: Dispatch Writer Task
        if state.step_count >= state.max_steps:
            logger.warning("[Supervisor] Max steps limit reached before synthesis.")
            return state

        state.step_count += 1
        write_task = AgentTask(
            task_id="task_02_write",
            target_role=AgentRole.WRITER,
            description="Synthesize research facts",
            payload={"facts": res_response.data.get("facts", [])},
        )
        write_response = self.writer.execute(write_task)
        state.task_history.append(write_response)

        if not write_response.success:
            logger.error("[Supervisor] Writing phase failed. Aborting.")
            return state

        # Step 3: Dispatch Reviewer Task
        if state.step_count >= state.max_steps:
            logger.warning("[Supervisor] Max steps limit reached before review.")
            return state

        state.step_count += 1
        review_task = AgentTask(
            task_id="task_03_review",
            target_role=AgentRole.REVIEWER,
            description="Review draft quality",
            payload={"draft": write_response.data.get("draft", "")},
        )
        review_response = self.reviewer.execute(review_task)
        state.task_history.append(review_response)

        if review_response.success:
            state.final_output = write_response.data.get("draft")
            logger.info(
                f"[Supervisor] Workflow completed successfully in {state.step_count} steps."
            )
        else:
            logger.error(
                f"[Supervisor] Review failed: {review_response.data.get('feedback')}"
            )

        return state


# ==============================================================================
# 4. Execution Loop
# ==============================================================================


def main() -> None:
    print("\n--- Running Chapter 05: Multi-Agent Orchestration Engine ---\n")

    supervisor = AgentSupervisor(max_steps=5)
    result_state = supervisor.run(user_query="Distributed Agent Topologies")

    print("\n=======================================================")
    print("FINAL ORCHESTRATION RESULT")
    print("=======================================================")
    print(f"Status:            {'SUCCESS' if result_state.final_output else 'FAILED'}")
    print(f"Total Steps Taken: {result_state.step_count} / {result_state.max_steps}")
    print(f"Tasks Executed:    {len(result_state.task_history)}")
    print("\nFinal Output:")
    print(
        result_state.final_output
        if result_state.final_output
        else "No output generated."
    )
    print("=======================================================\n")


if __name__ == "__main__":
    main()
