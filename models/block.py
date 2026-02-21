"""Transformer block with pre-layer-norm architecture.

Implements the decoder transformer block studied in:
  Gerber (2025), "Attention Is Not All You Need: The Importance of Feedforward
  Networks in Transformer Models", arXiv:2505.06633.

Following GPT-3 and modern LLM conventions the paper uses pre-norm (LayerNorm
applied *before* each sub-layer) rather than the post-norm of the original
Transformer paper.  The overall structure is:

    x = x + MHA(LayerNorm(x))          # attention sub-layer
    x = x + FFN(LayerNorm(x))          # FFN sub-layer (may be a no-op for FFN0)

The FFN sub-layer is one of the four variants (FFN0..FFN3) configured via
the `ffn_layers` argument.
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .ffn import build_ffn


class TransformerBlock(nn.Module):
    """Single decoder-only transformer block.

    Args:
        d_model: Model (embedding) dimension.
        n_heads: Number of attention heads.
        ffn_layers: Number of linear layers in the FFN sub-layer (0, 1, 2, or 3).
        d_ff: Inner FFN dimension (defaults to 4 * d_model).
        dropout: Dropout probability applied in both MHA and FFN.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_layers: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Pre-norm layers
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Sub-layers
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = build_ffn(ffn_layers, d_model, d_ff, dropout)
        self.res_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Attention sub-layer with pre-norm and residual
        x = x + self.res_dropout(self.attn(self.ln1(x)))

        # FFN sub-layer with pre-norm and residual
        x = x + self.res_dropout(self.ffn(self.ln2(x)))

        return x
