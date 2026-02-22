"""Decoder-only transformer language model.

Full autoregressive language model built from stacked TransformerBlocks as
described in:
  Gerber (2025), "Attention Is Not All You Need: The Importance of Feedforward
  Networks in Transformer Models", arXiv:2505.06633.

Architecture overview:
    token_embeddings + positional_embeddings
      -> dropout
      -> N x TransformerBlock (with configurable FFN depth)
      -> LayerNorm
      -> linear projection to vocabulary (weight-tied with token embeddings)

The weight-tying between the input embedding and the LM head follows the
approach used in GPT-2/GPT-3 and saves a significant number of parameters
for large vocabulary sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock


class DecoderTransformer(nn.Module):
    """Decoder-only transformer for causal language modelling.

    Args:
        vocab_size: Vocabulary size (e.g. 50257 for GPT-2 BPE tokenizer).
        d_model: Model (embedding) dimension.
        n_layers: Number of stacked transformer blocks.
        n_heads: Number of attention heads per block.
        ffn_layers: Number of linear layers in each FFN sub-layer (0-3).
        max_seq_len: Maximum sequence length (used to size positional embeddings).
        d_ff: Inner FFN dimension (defaults to 4 * d_model).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_layers: int,
        max_seq_len: int = 1024,
        d_ff: int | None = None,
        dropout: float = 0.1,
        act_type: str = "gelu",
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.ffn_layers = ffn_layers
        self.act_type = act_type

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, ffn_layers, d_ff, dropout, act_type)
                for _ in range(n_layers)
            ]
        )

        # Final layer norm and LM head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share token embedding weights with the LM head
        self.lm_head.weight = self.token_emb.weight

        # Initialise weights (GPT-2 style)
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layers) as in GPT-2
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("fc2.weight") or pn.endswith("fc3.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * n_layers) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            input_ids: Token indices, shape (batch, seq_len).
            targets: Target token indices for loss computation, shape (batch, seq_len).
                     If None, only logits are returned.

        Returns:
            (logits, loss) where logits has shape (batch, seq_len, vocab_size)
            and loss is a scalar cross-entropy value (or None if targets is None).
        """
        B, T = input_ids.shape

        # Embeddings
        tok_emb = self.token_emb(input_ids)                           # (B, T, d_model)
        positions = torch.arange(T, device=input_ids.device)
        pos_emb = self.pos_emb(positions)                             # (T, d_model)

        x = self.emb_dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)                                      # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts broken down by component."""
        emb_params = sum(p.numel() for p in self.token_emb.parameters())
        emb_params += sum(p.numel() for p in self.pos_emb.parameters())

        attn_params = 0
        ffn_params = 0
        ln_params = 0
        for block in self.blocks:
            attn_params += sum(p.numel() for p in block.attn.parameters())
            ffn_params += sum(p.numel() for p in block.ffn.parameters())
            ln_params += sum(p.numel() for p in block.ln1.parameters())
            ln_params += sum(p.numel() for p in block.ln2.parameters())

        lm_head_params = 0  # weight-tied, not counted separately
        ln_f_params = sum(p.numel() for p in self.ln_f.parameters())

        total = emb_params + attn_params + ffn_params + ln_params + ln_f_params

        return {
            "embeddings": emb_params,
            "attention": attn_params,
            "ffn": ffn_params,
            "layer_norm": ln_params + ln_f_params,
            "total": total,
        }

    def describe(self) -> str:
        """Return a human-readable model description."""
        counts = self.count_parameters()
        lines = [
            f"DecoderTransformer (ffn_layers={self.ffn_layers}, act={self.act_type})",
            f"  n_layers   : {self.n_layers}",
            f"  d_model    : {self.d_model}",
            f"  ffn_layers : {self.ffn_layers}",
            f"  act_type   : {self.act_type}",
            "",
            "  Parameters:",
            f"    embeddings  : {counts['embeddings']:>12,}",
            f"    attention   : {counts['attention']:>12,}",
            f"    ffn         : {counts['ffn']:>12,}",
            f"    layer_norm  : {counts['layer_norm']:>12,}",
            f"    total       : {counts['total']:>12,}",
        ]
        return "\n".join(lines)
