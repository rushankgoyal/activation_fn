"""Feed-forward network (FFN) variants for transformer blocks.

Implements the four FFN architectures studied in:
  Gerber (2025), "Attention Is Not All You Need: The Importance of Feedforward
  Networks in Transformer Models", arXiv:2505.06633.

The paper compares:
  - FFN0: No FFN layers  (Architecture D in Figure 1)
  - FFN1: One linear layer  (Architecture C in Figure 1)
  - FFN2: Two linear layers, the standard GPT-style FFN  (Architecture A in Figure 1)
  - FFN3: Three linear layers  (Architecture B in Figure 1)

The baseline configuration (FFN2) matches the MLP component of GPT-3 medium:
  d_model=1024, intermediate dimension = 4 * d_model = 4096, GELU activation.

Parameter counts per block (no biases):
  FFN0: 0 parameters
  FFN1: d_model^2 parameters          (single projection d -> d)
  FFN2: 8 * d_model^2 parameters      (d -> 4d -> d)
  FFN3: 24 * d_model^2 parameters     (d -> 4d -> 4d -> d)

MHA : FFN parameter ratio for the two-layer baseline = 8:3 when the
output projection is excluded, or 8:4 when included.
"""

import torch
import torch.nn as nn


class FFN0(nn.Module):
    """Architecture D: Zero linear layers — no FFN sub-layer.

    The residual branch carries the input unchanged; this module always
    returns a zero tensor so the residual connection is equivalent to
    skipping the sub-layer entirely.

    Args:
        d_model: Model dimension (unused, kept for uniform interface).
        dropout: Dropout probability (unused, kept for uniform interface).
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def count_parameters(self) -> int:
        return 0


class FFN1(nn.Module):
    """Architecture C: One linear layer.

    A single learnable projection from d_model -> d_model followed by a
    GELU non-linearity.  This is the minimal FFN that can learn a
    non-trivial per-position transformation while preserving the residual
    dimension.

    Parameter count: d_model^2  (+ d_model bias if bias=True)

    Args:
        d_model: Model dimension.
        dropout: Dropout probability applied after the activation.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.linear(x)))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class FFN2(nn.Module):
    """Architecture A: Two linear layers — standard GPT-style FFN.

    Projects input from d_model to d_ff (default: 4 * d_model), applies
    GELU, then projects back.  This matches the MLP sub-layer used in
    the original GPT-2/GPT-3 models.

    Parameter count: 2 * d_model * d_ff  (≈ 8 * d_model^2 for d_ff=4*d_model)

    Args:
        d_model: Model dimension.
        d_ff: Inner dimension of the FFN (default: 4 * d_model).
        dropout: Dropout probability applied between the two layers.
    """

    def __init__(
        self, d_model: int, d_ff: int | None = None, dropout: float = 0.1
    ) -> None:
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class FFN3(nn.Module):
    """Architecture B: Three linear layers.

    Extends the standard two-layer FFN with an additional hidden layer,
    giving the architecture d_model -> d_ff -> d_ff -> d_model.  GELU is
    applied after each of the first two projections.

    The paper shows this configuration outperforms the two-layer baseline
    even when using fewer transformer blocks (fewer total parameters).

    Parameter count: d_model*d_ff + d_ff^2 + d_ff*d_model
                   ≈ 24 * d_model^2  for d_ff = 4 * d_model

    Args:
        d_model: Model dimension.
        d_ff: Inner dimension (default: 4 * d_model).
        dropout: Dropout probability applied after each activation.
    """

    def __init__(
        self, d_model: int, d_ff: int | None = None, dropout: float = 0.1
    ) -> None:
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_ff)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_ffn(ffn_layers: int, d_model: int, d_ff: int | None = None, dropout: float = 0.1) -> nn.Module:
    """Factory function to build an FFN module by the number of linear layers.

    Args:
        ffn_layers: Number of linear layers in the FFN (0, 1, 2, or 3).
        d_model: Model dimension.
        d_ff: Inner FFN dimension (only used for ffn_layers >= 2).
        dropout: Dropout probability.

    Returns:
        An FFN module instance.
    """
    if ffn_layers == 0:
        return FFN0(d_model, dropout)
    elif ffn_layers == 1:
        return FFN1(d_model, dropout)
    elif ffn_layers == 2:
        return FFN2(d_model, d_ff, dropout)
    elif ffn_layers == 3:
        return FFN3(d_model, d_ff, dropout)
    else:
        raise ValueError(f"ffn_layers must be 0, 1, 2, or 3; got {ffn_layers}")
