"""
model/transformer/decoder.py
==============================
Transformer Decoder Block for the AttenSum Transformer.

Implements a single decoder layer as described in:

    "Attention Is All You Need" — Vaswani et al., 2017  (§3.1)
    https://arxiv.org/abs/1706.03762

Architecture
------------
A single decoder block contains **three** sub-layers, each wrapped with a
post-norm residual connection (LayerNorm applied after the addition)::

    ┌──────────────────────────────────────────────────────────┐
    │                      Decoder Block                       │
    │                                                          │
    │  x ─────────────────────────────────────┐               │
    │  │                                       │               │
    │  ▼                                       │               │
    │  Masked Multi-Head Self-Attention        │               │
    │  (Q=x, K=x, V=x,  mask=tgt_mask)        │               │
    │  │                                       │               │
    │  └──────── + ────────────────────────────┘               │
    │            │                                             │
    │            ▼                                             │
    │       LayerNorm                  (residual_1)            │
    │            │  x1                                         │
    │                                                          │
    │  x1 ────────────────────────────────────┐               │
    │  │                                       │               │
    │  ▼                                       │               │
    │  Encoder–Decoder Cross-Attention         │               │
    │  (Q=x1, K=memory, V=memory,              │               │
    │         mask=memory_mask)                │               │
    │  │                                       │               │
    │  └──────── + ────────────────────────────┘               │
    │            │                                             │
    │            ▼                                             │
    │       LayerNorm                  (residual_2)            │
    │            │  x2                                         │
    │                                                          │
    │  x2 ────────────────────────────────────┐               │
    │  │                                       │               │
    │  ▼                                       │               │
    │  PositionwiseFeedForward                 │               │
    │  │                                       │               │
    │  └──────── + ────────────────────────────┘               │
    │            │                                             │
    │            ▼                                             │
    │       LayerNorm                  (residual_3)            │
    │            │  output                                     │
    └────────────┼────────────────────────────────────────────-┘
                 ▼

Post-norm equations
-------------------
::

    self_attn_out              = MaskedMHA(x, x, x, tgt_mask)
    x1                         = LayerNorm(x  + self_attn_out)

    cross_attn_out             = MHA(x1, memory, memory, memory_mask)
    x2                         = LayerNorm(x1 + cross_attn_out)

    ffn_out                    = FFN(x2)
    output                     = LayerNorm(x2 + ffn_out)

Why masked self-attention?
--------------------------
During training the decoder receives the full target sequence shifted
right.  The causal (upper-triangular) mask on the self-attention layer
prevents each target position from "seeing" future tokens — simulating
the left-to-right auto-regressive generation at inference time.

Why encoder-decoder cross-attention?
--------------------------------------
Cross-attention lets the decoder query the encoder's output
(``memory``).  The *queries* come from the current decoder
representation ``x1``, while the *keys* and *values* come from
``memory``.  This allows every target position to gather information
from any source position, which is the fundamental mechanism for
sequence-to-sequence transduction.

    Q  ← decoder representation   (B, target_len, d_model)
    K  ← encoder output / memory  (B, source_len, d_model)
    V  ← encoder output / memory  (B, source_len, d_model)

The resulting attention weight matrix is **rectangular**:
``(B, num_heads, target_len, source_len)``.

Composing existing components
------------------------------
``DecoderBlock`` does **not** reimplement any mathematical logic.  It
wires together independently implemented and tested modules:

- :class:`~model.transformer.attention.MultiHeadAttention`  × 2
- :class:`~model.transformer.feed_forward.PositionwiseFeedForward`
- :class:`~model.transformer.normalization.ResidualConnection`  × 3
"""

from typing import Optional

import torch
import torch.nn as nn

from model.transformer.attention import MultiHeadAttention
from model.transformer.feed_forward import PositionwiseFeedForward
from model.transformer.normalization import ResidualConnection


class DecoderBlock(nn.Module):
    """A single Transformer Decoder block.

    Applies three sub-layers in sequence, each followed by a post-norm
    residual connection:

    1. **Masked Multi-Head Self-Attention** — each target position attends
       to all *preceding* target positions (past and itself). A causal
       (upper-triangular) mask is typically supplied as ``tgt_mask`` to
       prevent attention to future tokens.

    2. **Encoder-Decoder Cross-Attention** — the decoder queries the
       encoder output (``memory``).  Queries come from the decoder
       representation; keys and values come from the encoder.

    3. **Position-wise Feed-Forward Network** — a two-layer MLP applied
       independently to each target position.

    All three sub-layers are wrapped with ``ResidualConnection``, which
    implements the post-norm formula::

        output = LayerNorm(input + sublayer(input))

    Tensor shapes at each stage::

        Input x               : (B, tgt_len, d_model)
        memory (encoder out)  : (B, src_len, d_model)
              ↓  Masked Self-Attention (Q=K=V=x)
        self_attn_out         : (B, tgt_len, d_model)
        self_attn_weights     : (B, num_heads, tgt_len, tgt_len)
              ↓  ResidualConnection 1: LayerNorm(x + self_attn_out)
        x1                    : (B, tgt_len, d_model)
              ↓  Cross-Attention (Q=x1, K=memory, V=memory)
        cross_attn_out        : (B, tgt_len, d_model)
        cross_attn_weights    : (B, num_heads, tgt_len, src_len)  ← rectangular
              ↓  ResidualConnection 2: LayerNorm(x1 + cross_attn_out)
        x2                    : (B, tgt_len, d_model)
              ↓  PositionwiseFeedForward
        ffn_out               : (B, tgt_len, d_model)
              ↓  ResidualConnection 3: LayerNorm(x2 + ffn_out)
        output                : (B, tgt_len, d_model)

    Args:
        d_model   (int):   Model width.  Must be divisible by ``num_heads``.
        num_heads (int):   Number of parallel attention heads.
        d_ff      (int):   Inner dimension of the feed-forward network.
                           Typically ``4 × d_model`` as in the original paper.
        dropout   (float): Dropout probability forwarded to both attention
                           modules and the feed-forward network.
                           Defaults to ``0.0``.

    Returns (forward):
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            * ``output``             — ``(B, tgt_len, d_model)``
            * ``self_attn_weights``  — ``(B, num_heads, tgt_len, tgt_len)``
              per-head self-attention distributions.
            * ``cross_attn_weights`` — ``(B, num_heads, tgt_len, src_len)``
              per-head cross-attention distributions (rectangular).
            All three are returned so AttenSum's visualization layer can
            inspect attention patterns without a second forward pass.

    Raises:
        ValueError: If ``d_model % num_heads != 0``, or if any dimension
            argument is non-positive, or if ``dropout`` is not in
            ``[0.0, 1.0)``.

    Example::

        >>> dec    = DecoderBlock(d_model=512, num_heads=8, d_ff=2048)
        >>> x      = torch.randn(2, 20, 512)    # target: (B=2, tgt=20, d=512)
        >>> memory = torch.randn(2, 30, 512)    # encoder out: (B=2, src=30, d=512)
        >>> out, sw, cw = dec(x, memory)
        >>> out.shape
        torch.Size([2, 20, 512])
        >>> sw.shape
        torch.Size([2, 8, 20, 20])
        >>> cw.shape
        torch.Size([2, 8, 20, 30])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Validation is delegated to the sub-modules; MultiHeadAttention
        # raises ValueError for d_model % num_heads != 0 and other bad args.

        # ------------------------------------------------------------------ #
        # Sub-layer 1: Masked Multi-Head Self-Attention
        # Q = K = V = x  (same decoder sequence for all three)
        # ------------------------------------------------------------------ #
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ------------------------------------------------------------------ #
        # Sub-layer 2: Encoder-Decoder Cross-Attention
        # Q = decoder representation (x1)
        # K = V = encoder output (memory)
        # ------------------------------------------------------------------ #
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ------------------------------------------------------------------ #
        # Sub-layer 3: Position-wise Feed-Forward Network
        # ------------------------------------------------------------------ #
        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        # ------------------------------------------------------------------ #
        # Three independent ResidualConnections, each with its own LayerNorm.
        # ------------------------------------------------------------------ #
        self.residual_1 = ResidualConnection(d_model=d_model)   # after self-attn
        self.residual_2 = ResidualConnection(d_model=d_model)   # after cross-attn
        self.residual_3 = ResidualConnection(d_model=d_model)   # after FFN

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a single decoder block.

        Args:
            x           (torch.Tensor): Target sequence input,
                shape ``(B, tgt_len, d_model)``.
            memory      (torch.Tensor): Encoder output (source context),
                shape ``(B, src_len, d_model)``.
            tgt_mask    (torch.Tensor | None):
                Self-attention mask for the target sequence.  Typically a
                causal upper-triangular bool mask of shape
                ``(tgt_len, tgt_len)`` or ``(B, tgt_len, tgt_len)`` where
                ``True`` means "mask out" (do not attend).
                Defaults to ``None`` (attend to all target positions).
            memory_mask (torch.Tensor | None):
                Cross-attention mask that controls which source positions
                the decoder may attend to.  Shape broadcastable to
                ``(B, num_heads, tgt_len, src_len)`` — typically a
                padding mask for the encoder output.
                Defaults to ``None`` (attend to all source positions).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                * ``output``             — ``(B, tgt_len, d_model)``
                * ``self_attn_weights``  — ``(B, num_heads, tgt_len, tgt_len)``
                * ``cross_attn_weights`` — ``(B, num_heads, tgt_len, src_len)``
        """
        # ------------------------------------------------------------------ #
        # Stage 1 — Masked Multi-Head Self-Attention
        #
        # Q = K = V = x   →   every target position attends to every
        # (non-masked) target position.
        # A causal mask on tgt_mask is the typical choice for auto-regressive
        # decoding.
        #
        # self_attn_out    : (B, tgt_len, d_model)
        # self_attn_weights: (B, num_heads, tgt_len, tgt_len)
        # ------------------------------------------------------------------ #
        self_attn_out, self_attn_weights = self.self_attn(
            x, x, x, mask=tgt_mask
        )

        # ------------------------------------------------------------------ #
        # Stage 2 — First post-norm residual connection
        #
        # x1 = LayerNorm(x + self_attn_out)     shape: (B, tgt_len, d_model)
        # ------------------------------------------------------------------ #
        x1 = self.residual_1(x, self_attn_out)

        # ------------------------------------------------------------------ #
        # Stage 3 — Encoder-Decoder Cross-Attention
        #
        # Q ← x1      (decoder representation,   shape: B, tgt_len, d_model)
        # K ← memory  (encoder output,            shape: B, src_len, d_model)
        # V ← memory  (encoder output,            shape: B, src_len, d_model)
        #
        # cross_attn_out    : (B, tgt_len, d_model)
        # cross_attn_weights: (B, num_heads, tgt_len, src_len)  ← rectangular
        # ------------------------------------------------------------------ #
        cross_attn_out, cross_attn_weights = self.cross_attn(
            x1, memory, memory, mask=memory_mask
        )

        # ------------------------------------------------------------------ #
        # Stage 4 — Second post-norm residual connection
        #
        # x2 = LayerNorm(x1 + cross_attn_out)   shape: (B, tgt_len, d_model)
        # ------------------------------------------------------------------ #
        x2 = self.residual_2(x1, cross_attn_out)

        # ------------------------------------------------------------------ #
        # Stage 5 — Position-wise Feed-Forward Network
        #
        # ffn_out = FFN(x2)                      shape: (B, tgt_len, d_model)
        # ------------------------------------------------------------------ #
        ffn_out = self.ffn(x2)

        # ------------------------------------------------------------------ #
        # Stage 6 — Third post-norm residual connection
        #
        # output = LayerNorm(x2 + ffn_out)       shape: (B, tgt_len, d_model)
        # ------------------------------------------------------------------ #
        output = self.residual_3(x2, ffn_out)

        return output, self_attn_weights, cross_attn_weights
