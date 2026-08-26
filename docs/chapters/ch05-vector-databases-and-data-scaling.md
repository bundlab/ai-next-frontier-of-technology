# Chapter 05: Multi-Agent Orchestration & Communication Patterns

## Overview

Single-agent systems struggle when tasked with complex, multi-domain workflows. As prompt context grows and responsibility expands, single agents suffer from prompt degradation, tool selection confusion, and context fragmentation. 

**Multi-Agent Systems (MAS)** mitigate these limitations by decoupling monolithic agent loops into specialized, autonomous units. Each agent operates with a narrow domain role, isolated state memory, and a dedicated toolset.

This chapter covers multi-agent interaction topologies, state routing patterns, consensus protocols, and asynchronous agent communication models for building fault-tolerant agent networks.

---

## Core Concepts

### 1. Multi-Agent Topologies

Choosing the right communication topology depends on workflow complexity, dynamic branching needs, and task determinism:

| Topology Pattern | Description | Best Used For |
| :--- | :--- | :--- |
| **Hierarchical Router (Supervisor)** | A central coordinator delegates sub-tasks to downstream specialist agents and synthesizes final responses. | Complex analytical requests, dynamic workflow routing, software development systems. |
| **Sequential Chain (Pipeline)** | Agents pass output state strictly linear to the next agent (`Agent A` $\to$ `Agent B` $\to$ `Agent C`). | Content editing pipelines, data transform pipelines, fixed multi-stage audits. |
| **Collaborative Swarm (Peer-to-Peer)** | Agents publish and subscribe to a shared event bus or blackboard, handoffs occurring autonomously. | Open-ended exploration, decentralized consensus building, research assistance. |

### 2. State Handoff & Memory Isolation

To prevent prompt pollution, sub-agents should never inherit the full conversation transcript of their supervisor. Instead, orchestrators use **Explicit State Handoff**:

- **Input Serialization:** The supervisor extracts only task-relevant variables into a structured payload passed to the specialist.
- **Output Aggregation:** The specialist returns a typed execution output (`Pydantic` schema) back to the supervisor state store.
- **Context Boundaries:** Each agent maintains an isolated system prompt and short-term execution memory.

### 3. Loop Guardrails & Deadlock Prevention

Autonomous loops run the risk of infinite regression (e.g., two agents endlessly passing execution back and forth). Production systems enforce strict bounds:

- **Maximum Step Counter:** Terminate orchestration loops after $N$ state transitions.
- **Cycle Detection:** Track previous state hashes; abort if an identical state reoccurs.
- **Fallback Handlers:** Default to human-in-the-loop (HITL) or graceful error responses upon step exhaustion.

---

## Implementation & Code Patterns

Below is a complete Python module demonstrating a **Hierarchical Multi-Agent System** using Pydantic, structured state passing, and a central supervisor orchestrator.

```python
#!/usr/bin/env python3
"""
Chapter 05: Multi-Agent Orchestration & Communication Patterns
Example: Supervisor-Specialist Agent Loop
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

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
    payload: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    task_id: str
    source_role: AgentRole
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None


class OrchestratorState(BaseModel):
    user_query: str
    step_count: int = 0
    max_steps: int = 5
    task_history: List[AgentResponse] = Field(default_factory=list)
    final_output: Optional[str] = None


# ==============================================================================
# 2. Specialist Agents
# ==============================================================================

class ResearchAgent:
    """Specialist agent focused on gathering domain facts."""
    
    role: AgentRole = AgentRole.RESEARCHER

    def execute(self, task: AgentTask) -> AgentResponse:
        logger.info(f"[{self.role.value}] Researching query: {task.description}")
        query = task.payload.get("query", "")
        
        # Simulated research output payload
        facts = [
            f"Fact 1: Multi-agent systems decouple tasks for domain '{query}'.",
            "Fact 2: Shared blackboard patterns enable asynchronous peer communication.",
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
        logger.info(f"[{self.role.value}] Synthesizing content for task: {task.task_id}")
        facts = task.payload.get("facts", [])
        draft = "Summary Report:\n" + "\n".join(f"- {fact}" for fact in facts)
        
        return AgentResponse(
            task_id=task.task_id,
            source_role=self.role,
            success=True,
            data={"draft": draft},
        )


# ==============================================================================
# 3. Supervisor Orchestrator
# ==============================================================================

class AgentSupervisor:
    """Central Orchestrator managing agent execution flow and shared state."""

    def __init__(self):
        self.researcher = ResearchAgent()
        self.writer = WriterAgent()

    def run(self, user_query: str) -> OrchestratorState:
        state = OrchestratorState(user_query=user_query)
        logger.info(f"[Supervisor] Starting multi-agent workflow for: '{user_query}'")

        # Step 1: Dispatch Research Task
        state.step_count += 1
        research_task = AgentTask(
            task_id="task_01",
            target_role=AgentRole.RESEARCHER,
            description="Gather structural facts",
            payload={"query": user_query},
        )
        research_res = self.researcher.execute(research_task)
        state.task_history.append(research_res)

        # Step 2: Dispatch Writer Task using outputs from Research Stage
        if research_res.success:
            state.step_count += 1
            writer_task = AgentTask(
                task_id="task_02",
                target_role=AgentRole.WRITER,
                description="Synthesize research output",
                payload={"facts": research_res.data.get("facts", [])},
            )
            writer_res = self.writer.execute(writer_task)
            state.task_history.append(writer_res)

            if writer_res.success:
                state.final_output = writer_res.data.get("draft")

        logger.info(f"[Supervisor] Workflow completed in {state.step_count} steps.")
        return state


# ==============================================================================
# 4. Execution Loop
# ==============================================================================

def main() -> None:
    print("\n--- Running Chapter 05: Multi-Agent Orchestration Engine ---\n")

    supervisor = AgentSupervisor()
    result_state = supervisor.run(user_query="Distributed Agent Topologies")

    print("\n=======================================================")
    print("FINAL ORCHESTRATION RESULT")
    print("=======================================================")
    print(f"Status: {'SUCCESS' if result_state.final_output else 'FAILED'}")
    print(f"Total State Steps: {result_state.step_count}")
    print("\nSynthesized Output:")
    print(result_state.final_output)
    print("=======================================================\n")


if __name__ == "__main__":
    main()

```

---

## Architecture & System Design Guidelines

1. **Keep Agent Payloads Strictly Typed:** Enforce schema validation (`Pydantic`) across all agent communication boundaries to prevent unexpected input formats from degrading agent execution.
2. **Implement Idempotent Agent Actions:** Specialist agents that trigger external APIs or execute database writes should handle retry loops idempotently using unique execution tokens (`task_id`).
3. **Log Handoff Telemetry:** Log state transfers between sub-agents alongside latency metrics, token consumption counts, and agent state transitions to trace workflow errors easily.

---

## Summary & Key Takeaways

* **Decouple Complex Responsibilities:** Multi-agent architectures improve systemic reliability by restricting sub-agents to isolated domain prompts and targeted toolsets.
* **Topologies Control Predictability:** Use hierarchical supervisor patterns when deterministic execution routing is required, and save P2P swarm patterns for exploratory tasks.
* **State Hygiene is Vital:** Pass explicit, minimal state dictionaries between agents rather than dumping raw chat histories across boundaries.

