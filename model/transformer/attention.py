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


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention.

    Implements the multi-head extension of scaled dot-product attention from
    "Attention Is All You Need" (Vaswani et al., 2017) §3.2.2::

        MultiHead(Q, K, V) = Concat(head_1, …, head_h) W_O

        where head_i = Attention(Q W_Q_i, K W_K_i, V W_V_i)

    Rather than running ``num_heads`` independent attention modules, the
    implementation projects once to ``d_model``, reshapes into heads, and
    merges the head axis into the batch axis before delegating to a single
    :class:`ScaledDotProductAttention` instance.  This reuses all of that
    module's masking and dropout logic without any duplication.

    Tensor dimension legend
    -----------------------
    B        — batch size
    seq_q    — query sequence length
    seq_k    — key / value sequence length  (= seq_q for self-attention)
    d_model  — model width (must be divisible by num_heads)
    H        — num_heads
    head_dim — d_model // num_heads  (depth per head)

    Shape pipeline::

        Input Q / K / V : (B, seq_q/k, d_model)
            ↓  Linear projection (W_Q / W_K / W_V)
        Projected        : (B, seq_q/k, d_model)
            ↓  view + transpose
        Per-head tensors : (B, H, seq_q/k, head_dim)
            ↓  merge B and H into one axis
        Flat for SDPA    : (B*H, seq_q/k, head_dim)
            ↓  ScaledDotProductAttention
        SDPA output      : (B*H, seq_q, head_dim)
        SDPA weights     : (B*H, seq_q, seq_k)
            ↓  split B*H back into (B, H)
        Context          : (B, H, seq_q, head_dim)
        Attn weights     : (B, H, seq_q, seq_k)
            ↓  transpose + contiguous + view
        Concatenated     : (B, seq_q, d_model)
            ↓  Linear projection (W_O)
        Output           : (B, seq_q, d_model)

    Args:
        d_model   (int):   Model width.  Must be divisible by ``num_heads``.
        num_heads (int):   Number of parallel attention heads.
        dropout   (float): Dropout probability applied to the attention
                           weight matrix inside each head.  Defaults to ``0.0``.
        bias      (bool):  Whether to include bias terms in the four linear
                           projection layers.  Defaults to ``True``.

    Returns (forward):
        tuple[torch.Tensor, torch.Tensor]:
            * ``output``       — Shape ``(B, seq_q, d_model)``
            * ``attn_weights`` — Shape ``(B, H, seq_q, seq_k)``,
              the per-head, post-softmax attention distributions.
              Each ``[b, h, q, :]`` row sums to 1.0.
              Exposed for AttenSum's visualization layer.

    Raises:
        ValueError: If ``d_model`` is not divisible by ``num_heads``, or if
            dropout is outside ``[0.0, 1.0)``.

    Example::

        >>> mha = MultiHeadAttention(d_model=64, num_heads=4, dropout=0.0)
        >>> B, seq, d = 2, 10, 64
        >>> x = torch.randn(B, seq, d)
        >>> out, weights = mha(x, x, x)        # self-attention
        >>> out.shape
        torch.Size([2, 10, 64])
        >>> weights.shape
        torch.Size([2, 4, 10, 10])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {d_model}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, got {num_heads}.")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"dropout must be in the range [0.0, 1.0), got {dropout}."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads   # depth per head

        # Four learned linear projections — no Transformer logic lives here,
        # only weight matrices that map d_model → d_model.
        #
        # W_Q, W_K, W_V: project the raw Q/K/V inputs into the attention space.
        # W_O          : project the concatenated multi-head output back to d_model.
        self.w_q = nn.Linear(d_model, d_model, bias=bias)  # (d_model → d_model)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)  # (d_model → d_model)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)  # (d_model → d_model)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)  # (d_model → d_model)

        # Delegate per-head attention to the module we already built.
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Reshape a projected tensor into the per-head layout.

        Args:
            x       (torch.Tensor): Shape ``(B, seq_len, d_model)``.
            seq_len (int):          The sequence length of ``x``.

        Returns:
            torch.Tensor: Shape ``(B * num_heads, seq_len, head_dim)``.
                The batch and head axes are merged so a single call to
                :class:`ScaledDotProductAttention` handles all heads at once.
        """
        b = x.size(0)
        # (B, seq_len, d_model)
        #   → (B, seq_len, num_heads, head_dim)   via view
        #   → (B, num_heads, seq_len, head_dim)   via transpose
        #   → (B * num_heads, seq_len, head_dim)  via reshape
        return (
            x.view(b, seq_len, self.num_heads, self.head_dim)
             .transpose(1, 2)
             .reshape(b * self.num_heads, seq_len, self.head_dim)
        )

    def _merge_heads(self, x: torch.Tensor, b: int, seq_len: int) -> torch.Tensor:
        """Reverse ``_split_heads``: merge all head outputs into d_model.

        Args:
            x       (torch.Tensor): Shape ``(B * num_heads, seq_len, head_dim)``.
            b       (int):          Original batch size ``B``.
            seq_len (int):          Query sequence length.

        Returns:
            torch.Tensor: Shape ``(B, seq_len, d_model)``.
        """
        # (B * num_heads, seq_len, head_dim)
        #   → (B, num_heads, seq_len, head_dim)   via view
        #   → (B, seq_len, num_heads, head_dim)   via transpose
        #   → (B, seq_len, d_model)               via contiguous + view
        return (
            x.view(b, self.num_heads, seq_len, self.head_dim)
             .transpose(1, 2)
             .contiguous()
             .view(b, seq_len, self.d_model)
        )

    def _prepare_mask(
        self,
        mask: torch.Tensor,
        b: int,
        seq_q: int,
        seq_k: int,
    ) -> torch.Tensor:
        """Expand and reshape a mask to match the flat ``(B*H, seq_q, seq_k)`` layout.

        Accepted input shapes (all are broadcast-expanded to ``(B, H, seq_q, seq_k)``):

        * ``(seq_q, seq_k)``             — shared across all batches and heads
        * ``(B, seq_q, seq_k)``          — per-batch, shared across heads
        * ``(B, num_heads, seq_q, seq_k)`` — fully specified per-head mask

        Args:
            mask   (torch.Tensor): Input mask (bool or float).
            b      (int):          Batch size.
            seq_q  (int):          Query sequence length.
            seq_k  (int):          Key sequence length.

        Returns:
            torch.Tensor: Shape ``(B * num_heads, seq_q, seq_k)``,
                ready to pass straight into :class:`ScaledDotProductAttention`.

        Raises:
            ValueError: If the mask has an unsupported number of dimensions.
        """
        if mask.dim() == 2:
            # (seq_q, seq_k) → (1, 1, seq_q, seq_k)
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            # (B, seq_q, seq_k) → (B, 1, seq_q, seq_k)
            mask = mask.unsqueeze(1)
        elif mask.dim() == 4:
            # (B, H, seq_q, seq_k) — already the right shape
            pass
        else:
            raise ValueError(
                f"mask must be 2-D, 3-D, or 4-D, got {mask.dim()}-D."
            )

        # Broadcast over batch and head dimensions, then flatten B*H into axis 0.
        # .expand does not copy data; .reshape will copy only if needed.
        mask = mask.expand(b, self.num_heads, seq_q, seq_k)
        return mask.reshape(b * self.num_heads, seq_q, seq_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run multi-head attention.

        Supports both **self-attention** (``query is key is value``) and
        **cross-attention** (``query`` from one sequence, ``key`` / ``value``
        from another — e.g. decoder queries attending to encoder output).

        Args:
            query (torch.Tensor): Shape ``(B, seq_q, d_model)``.
            key   (torch.Tensor): Shape ``(B, seq_k, d_model)``.
            value (torch.Tensor): Shape ``(B, seq_k, d_model)``.
            mask  (torch.Tensor | None):
                Optional mask broadcastable to ``(B, H, seq_q, seq_k)``.
                Accepted shapes: ``(seq_q, seq_k)``, ``(B, seq_q, seq_k)``,
                ``(B, H, seq_q, seq_k)``.  Bool masks follow the convention
                ``True = mask out``.  Float masks are added directly to raw
                attention scores.  Defaults to ``None``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                * ``output``       — Shape ``(B, seq_q, d_model)``
                * ``attn_weights`` — Shape ``(B, H, seq_q, seq_k)``
                  per-head attention distributions.
        """
        b    = query.size(0)
        seq_q = query.size(1)
        seq_k = key.size(1)

        # ------------------------------------------------------------------ #
        # Step 1 — Linear projections
        #
        #   query → W_Q → Q_proj : (B, seq_q, d_model)
        #   key   → W_K → K_proj : (B, seq_k, d_model)
        #   value → W_V → V_proj : (B, seq_k, d_model)
        # ------------------------------------------------------------------ #
        q_proj = self.w_q(query)   # (B, seq_q, d_model)
        k_proj = self.w_k(key)     # (B, seq_k, d_model)
        v_proj = self.w_v(value)   # (B, seq_k, d_model)

        # ------------------------------------------------------------------ #
        # Step 2 — Split into heads and flatten batch + head into axis 0
        #
        #   (B, seq, d_model) → (B*H, seq, head_dim)
        # ------------------------------------------------------------------ #
        q_heads = self._split_heads(q_proj, seq_q)   # (B*H, seq_q, head_dim)
        k_heads = self._split_heads(k_proj, seq_k)   # (B*H, seq_k, head_dim)
        v_heads = self._split_heads(v_proj, seq_k)   # (B*H, seq_k, head_dim)

        # ------------------------------------------------------------------ #
        # Step 3 — Expand mask to cover all heads (if provided)
        #
        #   Resulting mask shape: (B*H, seq_q, seq_k)
        # ------------------------------------------------------------------ #
        flat_mask: Optional[torch.Tensor] = None
        if mask is not None:
            flat_mask = self._prepare_mask(mask, b, seq_q, seq_k)

        # ------------------------------------------------------------------ #
        # Step 4 — Scaled dot-product attention (all heads in one call)
        #
        #   context_flat : (B*H, seq_q, head_dim)
        #   weights_flat : (B*H, seq_q, seq_k)
        # ------------------------------------------------------------------ #
        context_flat, weights_flat = self.attention(
            q_heads, k_heads, v_heads, mask=flat_mask
        )

        # ------------------------------------------------------------------ #
        # Step 5 — Restore head and batch dimensions for weights
        #
        #   (B*H, seq_q, seq_k) → (B, H, seq_q, seq_k)
        # ------------------------------------------------------------------ #
        attn_weights = weights_flat.view(b, self.num_heads, seq_q, seq_k)
        # attn_weights: (B, H, seq_q, seq_k)

        # ------------------------------------------------------------------ #
        # Step 6 — Merge heads
        #
        #   (B*H, seq_q, head_dim) → (B, seq_q, d_model)
        # ------------------------------------------------------------------ #
        context = self._merge_heads(context_flat, b, seq_q)
        # context: (B, seq_q, d_model)

        # ------------------------------------------------------------------ #
        # Step 7 — Output projection
        #
        #   (B, seq_q, d_model) → (B, seq_q, d_model)
        # ------------------------------------------------------------------ #
        output = self.w_o(context)
        # output: (B, seq_q, d_model)

        return output, attn_weights
