"""
model/transformer/encoder.py
==============================
Transformer Encoder Block for the AttenSum Transformer.

Implements a single encoder layer as described in:

    "Attention Is All You Need" — Vaswani et al., 2017  (§3.1)
    https://arxiv.org/abs/1706.03762

Architecture
------------
A single encoder block consists of two sub-layers, each wrapped with a
**post-norm residual connection** (LayerNorm applied after the addition)::

    ┌─────────────────────────────────────────┐
    │              Encoder Block              │
    │                                         │
    │  x ──────────────────────────┐          │
    │  │                           │          │
    │  ▼                           │          │
    │  Multi-Head Self-Attention   │          │
    │  │                           │          │
    │  └──────── + ────────────────┘          │
    │            │                            │
    │            ▼                            │
    │       LayerNorm           (residual 1)  │
    │            │  x1                        │
    │  x1 ───────────────────────┐            │
    │  │                         │            │
    │  ▼                         │            │
    │  PositionwiseFeedForward   │            │
    │  │                         │            │
    │  └──────── + ──────────────┘            │
    │            │                            │
    │            ▼                            │
    │       LayerNorm           (residual 2)  │
    │            │  output                    │
    └────────────┼────────────────────────────┘
                 ▼

Post-norm equations
-------------------
::

    x1     = LayerNorm(x  + MultiHeadSelfAttention(x, x, x, mask))
    output = LayerNorm(x1 + PositionwiseFeedForward(x1))

Why self-attention?
-------------------
In the encoder, every position attends to every other position in the
*same* sequence — query, key, and value all come from ``x``.  There is no
separate "source" sequence.  This allows every token to gather contextual
information from all other tokens before passing the representation to the
decoder.

Why post-norm?
--------------
The original Transformer applies LayerNorm *after* the residual addition
(post-norm), which is what ``ResidualConnection`` implements.  This module
composes two ``ResidualConnection`` instances accordingly.

Composing existing components
------------------------------
``EncoderBlock`` does **not** reimplement any mathematical logic.  It
wires together the independently implemented and tested modules:

- :class:`~model.transformer.attention.MultiHeadAttention`
- :class:`~model.transformer.feed_forward.PositionwiseFeedForward`
- :class:`~model.transformer.normalization.ResidualConnection`
"""

from typing import Optional

import torch
import torch.nn as nn

from model.transformer.attention import MultiHeadAttention
from model.transformer.feed_forward import PositionwiseFeedForward
from model.transformer.normalization import ResidualConnection


class EncoderBlock(nn.Module):
    """A single Transformer Encoder block.

    Applies two sub-layers in sequence, each followed by a post-norm
    residual connection:

    1. **Multi-Head Self-Attention** — every position attends to every
       other position in the same sequence.
    2. **Position-wise Feed-Forward Network** — a two-layer MLP applied
       independently to each position.

    Both sub-layers are wrapped with ``ResidualConnection``, which
    implements::

        output = LayerNorm(input + sublayer(input))

    Tensor shapes at each stage::

        Input             : (B, seq_len, d_model)
              ↓  Multi-Head Self-Attention (Q=K=V=x)
        Attention output  : (B, seq_len, d_model)
              ↓  ResidualConnection 1:  LayerNorm(x + attn_output)
        x1                : (B, seq_len, d_model)
              ↓  PositionwiseFeedForward
        FFN output        : (B, seq_len, d_model)
              ↓  ResidualConnection 2:  LayerNorm(x1 + ffn_output)
        Output            : (B, seq_len, d_model)

    Attention weights are also returned so that AttenSum's visualization
    layer can inspect per-head attention patterns without a second forward
    pass.

    Args:
        d_model   (int):   Model width.  Must be divisible by ``num_heads``.
        num_heads (int):   Number of parallel attention heads.
        d_ff      (int):   Inner dimension of the feed-forward network.
                           Typically ``4 × d_model`` as in the original paper.
        dropout   (float): Dropout probability forwarded to both the
                           attention weight matrix (inside
                           ``MultiHeadAttention``) and the inner
                           representation of the feed-forward network.
                           Defaults to ``0.0`` (no dropout).

    Returns (forward):
        tuple[torch.Tensor, torch.Tensor]:
            * ``output``       — Shape ``(B, seq_len, d_model)``
            * ``attn_weights`` — Shape ``(B, num_heads, seq_len, seq_len)``
              per-head self-attention distributions.

    Raises:
        ValueError: If ``d_model % num_heads != 0``, or if any dimension
            argument is non-positive, or if ``dropout`` is not in
            ``[0.0, 1.0)``.

    Example::

        >>> enc = EncoderBlock(d_model=512, num_heads=8, d_ff=2048, dropout=0.0)
        >>> x   = torch.randn(2, 30, 512)   # (batch=2, seq=30, d_model=512)
        >>> out, weights = enc(x)
        >>> out.shape
        torch.Size([2, 30, 512])
        >>> weights.shape
        torch.Size([2, 8, 30, 30])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Validation is delegated to the sub-modules; they each raise
        # descriptive ValueError for bad arguments.  MultiHeadAttention
        # validates d_model % num_heads == 0.

        # ------------------------------------------------------------------ #
        # Sub-layer 1: Multi-Head Self-Attention
        # Q, K, V all receive the same input x (self-attention).
        # ------------------------------------------------------------------ #
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ------------------------------------------------------------------ #
        # Sub-layer 2: Position-wise Feed-Forward Network
        # ------------------------------------------------------------------ #
        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        # ------------------------------------------------------------------ #
        # Two independent ResidualConnections, each with its own LayerNorm.
        # Using separate instances keeps gradient flow and parameters clean.
        # ------------------------------------------------------------------ #
        self.residual_1 = ResidualConnection(d_model=d_model)
        self.residual_2 = ResidualConnection(d_model=d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single encoder block.

        Args:
            x    (torch.Tensor): Input of shape ``(B, seq_len, d_model)``.
            mask (torch.Tensor | None):
                Optional self-attention mask, broadcastable to
                ``(B, num_heads, seq_len, seq_len)``.
                Accepted shapes: ``(seq_len, seq_len)``,
                ``(B, seq_len, seq_len)``,
                ``(B, num_heads, seq_len, seq_len)``.
                Bool masks use ``True = mask out`` convention.
                Defaults to ``None`` (attend everywhere).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                * ``output``       — ``(B, seq_len, d_model)``
                * ``attn_weights`` — ``(B, num_heads, seq_len, seq_len)``
        """
        # ------------------------------------------------------------------ #
        # Stage 1 — Multi-Head Self-Attention
        #
        # Q = K = V = x  →  every position attends to every other position
        # in the same sequence.
        #
        # attn_out    : (B, seq_len, d_model)
        # attn_weights: (B, num_heads, seq_len, seq_len)
        # ------------------------------------------------------------------ #
        attn_out, attn_weights = self.self_attn(x, x, x, mask=mask)

        # ------------------------------------------------------------------ #
        # Stage 2 — First post-norm residual connection
        #
        # x1 = LayerNorm(x + attn_out)      shape: (B, seq_len, d_model)
        # ------------------------------------------------------------------ #
        x1 = self.residual_1(x, attn_out)

        # ------------------------------------------------------------------ #
        # Stage 3 — Position-wise Feed-Forward Network
        #
        # ffn_out = FFN(x1)                 shape: (B, seq_len, d_model)
        # ------------------------------------------------------------------ #
        ffn_out = self.ffn(x1)

        # ------------------------------------------------------------------ #
        # Stage 4 — Second post-norm residual connection
        #
        # output = LayerNorm(x1 + ffn_out)  shape: (B, seq_len, d_model)
        # ------------------------------------------------------------------ #
        output = self.residual_2(x1, ffn_out)

        return output, attn_weights
