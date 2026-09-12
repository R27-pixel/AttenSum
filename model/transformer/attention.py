"""
model/transformer/attention.py
================================
Scaled Dot-Product Attention for the AttenSum Transformer.

Implements the canonical attention function from:

    "Attention Is All You Need" — Vaswani et al., 2017
    https://arxiv.org/abs/1706.03762  (§3.2.1)

The equation is::

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

Design notes
------------
* This module is intentionally **single-head only**.  Multi-Head Attention
  will be a separate module that calls this one once per head (or once on
  the reshaped batch of heads).

* ``forward`` returns **both** the context tensor and the attention weight
  matrix.  The weight matrix is exposed so AttenSum's visualization layer
  can display per-position attention distributions without a second forward
  pass.

* Masking uses the **additive** convention: the caller passes a float tensor
  (or a bool tensor that is converted internally) whose ``-inf`` / ``True``
  entries mark positions that must not be attended to.  After adding the mask
  to the raw scores, softmax maps ``-inf`` to ~0.  This convention is
  compatible with both padding masks and causal (look-ahead) masks.

* No project-specific hyperparameters are hard-coded.  ``d_k`` is inferred
  from the last dimension of the query tensor at runtime.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention.

    Computes::

        scores   = Q @ K^T / sqrt(d_k)          # (B, seq_q, seq_k)
        scores   = scores + mask                 # optional masking
        weights  = softmax(scores, dim=-1)       # (B, seq_q, seq_k)
        weights  = dropout(weights)              # optional regularisation
        output   = weights @ V                   # (B, seq_q, d_v)

    Tensor dimension legend
    -----------------------
    B     — batch size
    seq_q — query sequence length
    seq_k — key/value sequence length  (= seq_q for self-attention)
    d_k   — query/key depth (inferred from Q and K at runtime)
    d_v   — value depth (inferred from V at runtime)

    Args:
        dropout (float): Dropout probability applied to the attention
            weight matrix **before** multiplying by V.  Set to ``0.0``
            to disable.  Defaults to ``0.0``.

    Inputs:
        query  (torch.Tensor): Shape ``(B, seq_q, d_k)``
        key    (torch.Tensor): Shape ``(B, seq_k, d_k)``
        value  (torch.Tensor): Shape ``(B, seq_k, d_v)``
        mask   (torch.Tensor | None):
            Optional mask of shape broadcastable to ``(B, seq_q, seq_k)``.

            * **Bool mask**: ``True`` → position is masked out (ignored).
            * **Float mask**: values are *added* directly to the raw
              scores; use ``-inf`` (``float("-inf")``) to mask positions.

            Defaults to ``None`` (no masking).

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            * ``output``       — Shape ``(B, seq_q, d_v)``
            * ``attn_weights`` — Shape ``(B, seq_q, seq_k)``,
              the post-softmax attention distribution.
              Each row sums to 1.0 (modulo dropout at train time).
              Returned even when ``mask`` is ``None`` for visualization.

    Raises:
        ValueError: If ``query`` and ``key`` have different ``d_k``
            dimensions, or if ``key`` and ``value`` have different
            sequence lengths.

    Example::

        >>> attn = ScaledDotProductAttention(dropout=0.0)
        >>> B, seq_q, seq_k, d_k, d_v = 2, 10, 10, 64, 64
        >>> Q = torch.randn(B, seq_q, d_k)
        >>> K = torch.randn(B, seq_k, d_k)
        >>> V = torch.randn(B, seq_k, d_v)
        >>> output, weights = attn(Q, K, V)
        >>> output.shape
        torch.Size([2, 10, 64])
        >>> weights.shape
        torch.Size([2, 10, 10])
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()

        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"dropout must be in the range [0.0, 1.0), got {dropout}."
            )

        self.attn_dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run scaled dot-product attention.

        Args:
            query (torch.Tensor): Shape ``(B, seq_q, d_k)``.
            key   (torch.Tensor): Shape ``(B, seq_k, d_k)``.
            value (torch.Tensor): Shape ``(B, seq_k, d_v)``.
            mask  (torch.Tensor | None):
                Optional mask broadcastable to ``(B, seq_q, seq_k)``.
                Bool tensors are supported (``True`` = ignore position).
                Float tensors are added directly to the raw logit scores.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                ``(output, attn_weights)`` where
                ``output`` has shape ``(B, seq_q, d_v)`` and
                ``attn_weights`` has shape ``(B, seq_q, seq_k)``.

        Raises:
            ValueError: On d_k mismatch between Q and K, or seq_k
                mismatch between K and V.
        """
        # ------------------------------------------------------------------ #
        # Dimension validation
        # ------------------------------------------------------------------ #
        d_k_q = query.size(-1)   # depth of query
        d_k_k = key.size(-1)     # depth of key  — must match d_k_q
        seq_k_k = key.size(1)    # key sequence length
        seq_k_v = value.size(1)  # value sequence length — must match seq_k_k

        if d_k_q != d_k_k:
            raise ValueError(
                f"query d_k ({d_k_q}) must equal key d_k ({d_k_k})."
            )
        if seq_k_k != seq_k_v:
            raise ValueError(
                f"key seq_k ({seq_k_k}) must equal value seq_k ({seq_k_v})."
            )

        d_k: int = d_k_q  # confirmed equal

        # ------------------------------------------------------------------ #
        # Step 1 — Compute raw dot-product scores
        #
        #   query : (B, seq_q, d_k)
        #   key^T : (B, d_k, seq_k)   — transpose last two dims
        #   scores: (B, seq_q, seq_k)
        # ------------------------------------------------------------------ #
        scores = torch.matmul(query, key.transpose(-2, -1))
        # scores: (B, seq_q, seq_k)

        # ------------------------------------------------------------------ #
        # Step 2 — Scale by 1 / sqrt(d_k)
        #
        # Without scaling, large d_k pushes dot products into regions where
        # softmax has near-zero gradients (vanishing gradients).
        # ------------------------------------------------------------------ #
        scores = scores / math.sqrt(d_k)
        # scores: (B, seq_q, seq_k)

        # ------------------------------------------------------------------ #
        # Step 3 — Apply optional mask
        #
        # Bool mask  → True means "this position must be ignored".
        #              Converted to -inf so softmax produces ~0 weight.
        # Float mask → Added directly; caller uses -inf for positions to mask.
        # ------------------------------------------------------------------ #
        if mask is not None:
            if mask.dtype == torch.bool:
                # Expand mask to match score shape if needed, then fill
                scores = scores.masked_fill(mask, float("-inf"))
            else:
                # Additive float mask — caller is responsible for -inf values
                scores = scores + mask
        # scores: (B, seq_q, seq_k)

        # ------------------------------------------------------------------ #
        # Step 4 — Softmax along the key dimension
        #
        # dim=-1 normalises over seq_k so each query position produces a
        # probability distribution over all key positions.
        # ------------------------------------------------------------------ #
        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights: (B, seq_q, seq_k) — rows sum to 1.0

        # ------------------------------------------------------------------ #
        # Step 5 — Optional dropout on attention weights
        #
        # Applied *after* softmax and *before* the weighted sum.
        # Only active during training (nn.Dropout is a no-op in eval mode).
        # ------------------------------------------------------------------ #
        attn_weights_dropped = self.attn_dropout(attn_weights)
        # attn_weights_dropped: (B, seq_q, seq_k)

        # ------------------------------------------------------------------ #
        # Step 6 — Weighted sum over value vectors
        #
        #   attn_weights_dropped: (B, seq_q, seq_k)
        #   value               : (B, seq_k, d_v)
        #   output              : (B, seq_q, d_v)
        # ------------------------------------------------------------------ #
        output = torch.matmul(attn_weights_dropped, value)
        # output: (B, seq_q, d_v)

        # Return the pre-dropout weights for visualization — they represent
        # the true attention distribution before stochastic zeroing.
        return output, attn_weights
