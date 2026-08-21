#!/usr/bin/env bash

set -euo pipefail

DOCS_DIR="docs/chapters"
EXAMPLES_DIR="examples"

declare -A CHAPTERS=(
  ["ch01-ai-first-architecture.md"]="Chapter 01: The AI-First Architecture Paradigm"
  ["ch02-modern-ai-tech-stack.md"]="Chapter 02: The Modern AI Runtime & Tech Stack"
  ["ch03-ml-foundations-and-mechanics.md"]="Chapter 03: Foundations of Machine Learning & Model Mechanics"
  ["ch04-context-engineering-and-rag.md"]="Chapter 04: Context Engineering & Retrieval-Augmented Generation"
  ["ch05-vector-databases-and-data-scaling.md"]="Chapter 05: Vector Databases & High-Dimensional Data Scaling"
  ["ch06-architecting-autonomous-agents.md"]="Chapter 06: Architecting Autonomous Agents"
  ["ch07-multi-agent-systems-and-orchestration.md"]="Chapter 07: Multi-Agent Systems & Orchestration Patterns"
  ["ch08-observability-tracing-and-evals.md"]="Chapter 08: Observability, Tracing, and Evaluation Frameworks"
  ["ch09-reliability-fallbacks-and-guardrails.md"]="Chapter 09: System Reliability, Fallbacks, and Guardrails"
  ["ch10-fine-tuning-quantization-distillation.md"]="Chapter 10: Optimization: Fine-Tuning, Quantization & Distillation"
  ["ch11-edge-ai-and-local-runtimes.md"]="Chapter 11: Edge AI, Local Runtimes, and Hybrid Deployment"
  ["ch12-future-proof-system-architect.md"]="Chapter 12: The Future-Proof System Architect"
)

echo "🚀 Initializing repository structure..."

mkdir -p "${DOCS_DIR}"
mkdir -p "${EXAMPLES_DIR}/agent_patterns"
mkdir -p "${EXAMPLES_DIR}/rag_pipelines"
mkdir -p "${EXAMPLES_DIR}/eval_frameworks"

echo "📂 Directories created successfully."
echo "📝 Generating chapter files..."

for FILE in "${!CHAPTERS[@]}"; do
  TITLE="${CHAPTERS[$FILE]}"
  FILE_PATH="${DOCS_DIR}/${FILE}"

  if [ ! -f "${FILE_PATH}" ]; then
    cat < "${FILE_PATH}"
# ${TITLE}

## Overview
*Brief introduction to the architectural patterns and concepts covered in this chapter.*

## Core Concepts
- Key concept 1
- Key concept 2

## Implementation & Code Patterns
\`\`\`python
# Code examples and architectural implementations go here
\`\`\`

## Architecture & System Design
- Diagrams and flowcharts

## Summary & Key Takeaways
- Summary point 1
- Summary point 2
EOT
    echo "  + Created: ${FILE_PATH}"
  else
    echo "  ~ Skipped (already exists): ${FILE_PATH}"
  fi
done

touch "${EXAMPLES_DIR}/agent_patterns/.gitkeep"
touch "${EXAMPLES_DIR}/rag_pipelines/.gitkeep"
touch "${EXAMPLES_DIR}/eval_frameworks/.gitkeep"

echo "✅ Setup complete! All folders and 12 chapter files are ready."
