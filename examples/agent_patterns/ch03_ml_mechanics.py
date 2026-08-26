#!/usr/bin/env python3
"""
Chapter 03: Foundations of Machine Learning & Model Mechanics
Example: Transformer Mechanics & KV Cache Calculator

This executable script demonstrates tokenization estimation, KV cache memory footprint
calculations, and temperature-adjusted logit Softmax probability sampling.
"""

import logging
import math

from pydantic import BaseModel, Field

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MLMechanics")


# ==============================================================================
# 1. Models & Calculations
# ==============================================================================


class KVCacheRequirement(BaseModel):
    num_layers: int = Field(..., ge=1)
    hidden_heads: int = Field(..., ge=1)
    head_dim: int = Field(..., ge=1)
    sequence_length: int = Field(..., ge=1)
    batch_size: int = Field(..., ge=1)
    bytes_per_param: int = Field(default=2, description="2 bytes for FP16/BF16")

    def calculate_bytes(self) -> int:
        """Calculates total KV Cache memory consumption in bytes."""
        # Factor of 2 represents Key and Value matrices
        return (
            2
            * self.num_layers
            * self.hidden_heads
            * self.head_dim
            * self.sequence_length
            * self.batch_size
            * self.bytes_per_param
        )

    def calculate_gigabytes(self) -> float:
        """Converts memory consumption to Gigabytes (GB)."""
        return round(self.calculate_bytes() / (1024**3), 4)


# ==============================================================================
# 2. Temperature Logit Softmax Engine
# ==============================================================================


def apply_temperature_softmax(logits: list[float], temperature: float) -> list[float]:
    """Applies temperature scaling to raw logits and computes Softmax probabilities."""
    if temperature <= 0.0:
        # Argmax / Greedy selection simulation
        max_idx = logits.index(max(logits))
        return [1.0 if i == max_idx else 0.0 for i in range(len(logits))]

    # Scale logits by temperature
    scaled_logits = [l / temperature for l in logits]

    # Subtract max for numerical stability
    max_logit = max(scaled_logits)
    exp_logits = [math.exp(l - max_logit) for l in scaled_logits]
    sum_exp_logits = sum(exp_logits)

    return [round(exp_l / sum_exp_logits, 4) for exp_l in exp_logits]


# ==============================================================================
# 3. Execution Example
# ==============================================================================


def main() -> None:
    print("\n--- Running Chapter 03: ML Mechanics & KV Cache Estimator ---\n")

    # A. Calculate KV Cache footprint for an 8B Parameter Model (e.g., Llama 3 8B)
    # 32 layers, 32 heads, 128 head dimension, 8192 context length, batch size 4
    kv_config = KVCacheRequirement(
        num_layers=32,
        hidden_heads=32,
        head_dim=128,
        sequence_length=8192,
        batch_size=4,
        bytes_per_param=2,  # FP16
    )

    gb_required = kv_config.calculate_gigabytes()
    logger.info("Model Spec: 32 Layers, 8k Context, Batch Size 4 (FP16)")
    logger.info(f"Required KV Cache Memory Footprint: {gb_required} GB VRAM")

    # B. Demonstrate Temperature Scaling on Logits
    raw_logits = [2.0, 1.5, 5.0, 0.5]
    tokens = ["code", "script", "function", "variable"]

    print("\nLogit Temperature Comparison:")
    for temp in [0.1, 0.7, 1.5]:
        probs = apply_temperature_softmax(raw_logits, temperature=temp)
        prob_dist = dict(zip(tokens, probs, strict=False))
        print(f"  Temperature T={temp:<3} -> Probabilities: {prob_dist}")
    print()


if __name__ == "__main__":
    main()
