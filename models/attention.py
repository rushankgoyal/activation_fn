"""Multi-head causal self-attention for decoder-only transformers.

Implements the standard multi-head attention (MHA) mechanism used in GPT-style
decoder-only transformers as described in:
  Gerber (2025), "Attention Is Not All You Need: The Importance of Feedforward
  Networks in Transformer Models", arXiv:2505.06633.

The MHA sub-layer contains Q, K, V, and output projection matrices, using
causal (left-to-right) masking suitable for autoregressive language modeling.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads. Must divide d_model evenly.
        dropout: Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout

        # Q, K, V and output projections (no bias, following GPT-3 convention)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal masking.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        B, T, C = x.shape

        # Project to Q, K, V and split into heads
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Use PyTorch 2.0 efficient scaled dot-product attention when available
        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual causal attention for older PyTorch versions
            scale = math.sqrt(self.d_head)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
            causal_mask = torch.tril(
                torch.ones(T, T, device=x.device, dtype=torch.bool)
            )
            attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_out = torch.matmul(attn_weights, v)

        # Merge heads and project back to d_model
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)
