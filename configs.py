"""Model and training configurations for replicating arXiv:2505.06633.

Gerber (2025), "Attention Is Not All You Need: The Importance of Feedforward
Networks in Transformer Models."

The paper uses a GPT-3 medium-scale baseline:
  d_model=1024, n_layers=24, n_heads=16, ffn_layers=2, dropout=0.1

Four FFN variants are compared (Architectures A-D in Figure 1):
  A (FFN2): standard 2-layer FFN   — ~323M total parameters
  B (FFN3): 3-layer FFN, 24 blocks — ~726M total parameters
  C (FFN1): 1-layer FFN, 24 blocks
  D (FFN0): no FFN,      24 blocks

Additional experiments match parameter counts across configurations:
  3-layer FFN, ~10 blocks  → comparable parameter budget to 2-layer 24-block
  0-layer, larger d_model  → comparable parameter budget
  1-layer, larger d_model  → comparable parameter budget

For development/testing a small configuration is provided that can be trained
in minutes on a CPU or a single GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Transformer model hyper-parameters."""

    # Vocabulary / sequence length
    vocab_size: int = 50257          # GPT-2 BPE tokenizer
    max_seq_len: int = 1024          # context window

    # Architecture
    d_model: int = 1024              # embedding / model dimension
    n_layers: int = 24               # number of transformer blocks
    n_heads: int = 16                # number of attention heads
    ffn_layers: int = 2              # 0 | 1 | 2 | 3 linear layers in FFN
    d_ff: Optional[int] = None       # FFN inner dim (default: 4 * d_model)
    dropout: float = 0.1
    act_type: str = "gelu"           # activation function in FFN layers

    _VALID_ACT_TYPES = frozenset(
        ["gelu", "gelusine", "gelusincperturbation", "gmtu", "turbulent"]
    )

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.ffn_layers not in (0, 1, 2, 3):
            raise ValueError(f"ffn_layers must be 0, 1, 2, or 3; got {self.ffn_layers}")
        key = self.act_type.lower().replace("-", "").replace("_", "")
        if key not in self._VALID_ACT_TYPES:
            raise ValueError(
                f"act_type '{self.act_type}' is not recognised. "
                f"Valid options: {sorted(self._VALID_ACT_TYPES)}"
            )


@dataclass
class TrainingConfig:
    """Training hyper-parameters."""

    # Optimiser
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Scheduler (linear warm-up + cosine decay)
    warmup_steps: int = 2000
    max_steps: int = 100_000

    # Data
    batch_size: int = 32
    grad_accum_steps: int = 1        # effective batch = batch_size * grad_accum_steps

    # Compute
    mixed_precision: bool = True     # use torch.autocast (bf16 on Ampere+, fp16 otherwise)
    compile_model: bool = False      # torch.compile (PyTorch 2.0+)

    # Logging / checkpointing
    eval_interval: int = 500         # evaluate every N steps
    eval_batches: int = 50           # number of batches for evaluation
    log_interval: int = 50           # log training loss every N steps
    checkpoint_dir: str = "checkpoints"
    run_name: str = "run"


# ---------------------------------------------------------------------------
# Pre-defined model configurations
# ---------------------------------------------------------------------------

# Small configurations for quick experimentation (trains in minutes)
SMALL_CONFIGS: dict[str, ModelConfig] = {
    "small_0layer": ModelConfig(d_model=256, n_layers=4, n_heads=4, ffn_layers=0),
    "small_1layer": ModelConfig(d_model=256, n_layers=4, n_heads=4, ffn_layers=1),
    "small_2layer": ModelConfig(d_model=256, n_layers=4, n_heads=4, ffn_layers=2),
    "small_3layer": ModelConfig(d_model=256, n_layers=4, n_heads=4, ffn_layers=3),
}

# Medium configurations for single-GPU experimentation
MEDIUM_CONFIGS: dict[str, ModelConfig] = {
    "medium_0layer": ModelConfig(d_model=512, n_layers=8, n_heads=8, ffn_layers=0),
    "medium_1layer": ModelConfig(d_model=512, n_layers=8, n_heads=8, ffn_layers=1),
    "medium_2layer": ModelConfig(d_model=512, n_layers=8, n_heads=8, ffn_layers=2),
    "medium_3layer": ModelConfig(d_model=512, n_layers=8, n_heads=8, ffn_layers=3),
    # 3-layer with parameter-matched fewer blocks (~same params as medium_2layer 8-block)
    "medium_3layer_matched": ModelConfig(d_model=512, n_layers=4, n_heads=8, ffn_layers=3),
}

# Paper-scale configurations (GPT-3 medium; requires A100 and ~66 hours compute)
# Approximate parameter counts (excluding LM head weight tying):
#   paper_2layer : ~329M  (baseline)
#   paper_3layer : ~726M  (3-layer FFN, same n_layers)
#   paper_3layer_matched : ~329M  (3-layer FFN, ~10 blocks to match 2-layer 24-block params)
#   paper_1layer : ~277M
#   paper_0layer : ~253M
#   paper_1layer_hd: ~329M  (higher d_model to match baseline params)
#   paper_0layer_hd: ~329M  (higher d_model to match baseline params)
PAPER_CONFIGS: dict[str, ModelConfig] = {
    # Primary comparison: same n_layers=24, vary ffn_layers
    "paper_0layer": ModelConfig(d_model=1024, n_layers=24, n_heads=16, ffn_layers=0),
    "paper_1layer": ModelConfig(d_model=1024, n_layers=24, n_heads=16, ffn_layers=1),
    "paper_2layer": ModelConfig(d_model=1024, n_layers=24, n_heads=16, ffn_layers=2),
    "paper_3layer": ModelConfig(d_model=1024, n_layers=24, n_heads=16, ffn_layers=3),

    # 3-layer FFN with fewer blocks: parameter-matched to 2-layer 24-block baseline
    # 24 blocks × (4+8) × 1024² ≈ 302M; 10 blocks × (4+24) × 1024² ≈ 302M
    "paper_3layer_matched": ModelConfig(d_model=1024, n_layers=10, n_heads=16, ffn_layers=3),

    # Higher-dimension 0-layer and 1-layer to match baseline parameter count
    # 0-layer: 24 × 4 × d² ≈ 302M → d ≈ 1792 (1792/16 = 112)
    "paper_0layer_hd": ModelConfig(d_model=1792, n_layers=24, n_heads=16, ffn_layers=0),
    # 1-layer: 24 × 5 × d² ≈ 302M → d ≈ 1600 (1600/16 = 100)
    "paper_1layer_hd": ModelConfig(d_model=1600, n_layers=24, n_heads=16, ffn_layers=1),
}

ALL_CONFIGS = {**SMALL_CONFIGS, **MEDIUM_CONFIGS, **PAPER_CONFIGS}


def get_config(name: str) -> ModelConfig:
    """Retrieve a named ModelConfig.

    Args:
        name: One of the keys in ALL_CONFIGS.

    Returns:
        The corresponding ModelConfig instance.
    """
    if name not in ALL_CONFIGS:
        available = ", ".join(sorted(ALL_CONFIGS.keys()))
        raise KeyError(f"Unknown config '{name}'. Available: {available}")
    return ALL_CONFIGS[name]
