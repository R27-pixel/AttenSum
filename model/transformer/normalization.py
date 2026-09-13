"""
model/transformer/normalization.py
====================================
Layer Normalization and Post-Norm Residual Connection for the AttenSum
Transformer.

Implements the two normalization / connection components required by each
Transformer encoder and decoder sub-layer, based on:

    "Attention Is All You Need" — Vaswani et al., 2017  (§3.1)
    https://arxiv.org/abs/1706.03762

Components
----------
LayerNorm
    Thin, self-documenting wrapper around ``nn.LayerNorm`` that normalises
    each position's ``d_model``-dimensional representation independently.

ResidualConnection
    Combines the residual (skip) connection with LayerNorm in the
    **post-norm** ordering used by the original Transformer::

        output = LayerNorm(x + sublayer_output)

Background — why LayerNorm?
----------------------------
Batch Normalization normalises across the *batch* dimension and is
therefore sensitive to batch size and awkward for variable-length
sequences.  **Layer Normalization** (Ba et al., 2016) normalises across
the *feature* dimension (``d_model``) for each individual (batch, position)
pair, making it batch-size-independent and well-suited to NLP tasks.

For each position the computation is::

    μ  = mean(x, dim=-1)           # scalar per position
    σ² = var(x, dim=-1)            # scalar per position
    x̂  = (x − μ) / √(σ² + ε)     # normalised, shape (d_model,)
    y  = γ ⊙ x̂ + β               # learned scale (γ) and shift (β)

where ``γ`` and ``β`` are learnable parameters of shape ``(d_model,)``,
initialised to 1 and 0 respectively.

Background — why post-norm residual?
--------------------------------------
The original Transformer applies LayerNorm *after* the residual addition::

    output = LayerNorm(x + Sublayer(x))

This is called **Post-LN** (post-norm).  A later variant, **Pre-LN**
(pre-norm), applies LayerNorm *before* the sublayer::

    output = x + Sublayer(LayerNorm(x))

Pre-LN has better training stability for very deep networks, but this
project implements Post-LN to match the original paper exactly.  The
``ResidualConnection`` class documents this decision explicitly.

Modular design
--------------
Neither ``LayerNorm`` nor ``ResidualConnection`` knows anything about
Multi-Head Attention or Feed-Forward Networks.  The Encoder / Decoder
blocks will pass in sublayer *outputs* directly, keeping each component
single-responsibility.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer Normalization over the ``d_model`` (last) dimension.

    Wraps ``nn.LayerNorm`` with explicit documentation of what is being
    normalised and why.  All Transformer sub-layers share the same
    normalization scheme.

    **What it does**

    For each ``(batch, position)`` pair, the ``d_model``-dimensional
    vector is normalised to have zero mean and unit variance, then
    rescaled by learned parameters ``γ`` (weight) and ``β`` (bias)::

        y = γ ⊙ (x − μ) / √(σ² + ε) + β

    **Which dimension is normalised**

    The normalisation is applied across the *last* dimension (``d_model``).
    The batch (``B``) and sequence (``seq_len``) dimensions are untouched,
    so each position is normalised independently.

    Tensor shapes::

        Input  : (B, seq_len, d_model)
              ↓  normalise across d_model per (B, seq_len) position
        Output : (B, seq_len, d_model)

    Args:
        d_model (int): Size of the last dimension to normalise over.
            Must be a positive integer.
        eps (float): Small constant added to the variance for numerical
            stability.  Defaults to ``1e-6``.

    Raises:
        ValueError: If ``d_model`` is not a positive integer.

    Example::

        >>> ln = LayerNorm(d_model=512)
        >>> x  = torch.randn(2, 30, 512)
        >>> ln(x).shape
        torch.Size([2, 30, 512])
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(
                f"d_model must be a positive integer, got {d_model}."
            )

        self.d_model = d_model

        # nn.LayerNorm is a PyTorch primitive; it is NOT a Transformer block.
        # It performs the normalisation arithmetic and holds the two learnable
        # parameters γ (weight) and β (bias), each of shape (d_model,).
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalisation across the d_model dimension.

        Args:
            x (torch.Tensor): Shape ``(B, seq_len, d_model)``.

        Returns:
            torch.Tensor: Same shape ``(B, seq_len, d_model)``.
                Each ``(b, t)`` slice has approximately zero mean and unit
                variance before the learned affine transform is applied.
        """
        # Delegate to nn.LayerNorm which normalises the last `d_model`
        # dimensions — matching exactly our normalised_shape=(d_model,).
        return self.norm(x)


class ResidualConnection(nn.Module):
    """Post-Norm Residual Connection.

    Combines a residual (skip) connection with Layer Normalization in the
    **post-norm** ordering from the original Transformer paper::

        output = LayerNorm(x + sublayer_output)

    **What a residual connection does**

    The residual (or skip) connection adds the module's input ``x`` directly
    to its output ``sublayer_output``, creating a shortcut that lets
    gradients flow back through the network without vanishing.  This is the
    same idea as ResNet (He et al., 2016), applied here to each sub-layer
    of the Transformer.

    **Why post-norm?**

    The original "Attention Is All You Need" paper applies LayerNorm
    *after* the addition::

        output = LayerNorm(x + Sublayer(x))   ← post-norm  ✓ (this class)

    An alternative pre-norm formulation applies LayerNorm *before* the
    sublayer::

        output = x + Sublayer(LayerNorm(x))   ← pre-norm   ✗ (not this class)

    This module implements post-norm to match the original paper.

    **API contract**

    ``ResidualConnection`` does *not* run the sublayer itself.  The caller
    is responsible for computing ``sublayer_output`` (e.g. the result of
    Multi-Head Attention or the Feed-Forward Network) and passing it in.
    This keeps the class reusable across different sublayer types and avoids
    coupling it to any specific Transformer component.

    Tensor shapes::

        x              : (B, seq_len, d_model)
        sublayer_output: (B, seq_len, d_model)   ← same shape required
             ↓  add
        sum            : (B, seq_len, d_model)
             ↓  LayerNorm (across d_model)
        output         : (B, seq_len, d_model)

    Args:
        d_model (int): Model width — must match the last dimension of both
            ``x`` and ``sublayer_output``.  Must be a positive integer.
        eps (float): Epsilon passed through to ``LayerNorm``.
            Defaults to ``1e-6``.

    Raises:
        ValueError: If ``d_model`` is not a positive integer (propagated
            from ``LayerNorm``).

    Example::

        >>> rc  = ResidualConnection(d_model=512)
        >>> x   = torch.randn(2, 30, 512)
        >>> sub = torch.randn(2, 30, 512)   # output of some sublayer
        >>> rc(x, sub).shape
        torch.Size([2, 30, 512])
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()

        # LayerNorm validates d_model; no duplicated validation needed here.
        self.layer_norm = LayerNorm(d_model=d_model, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        sublayer_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the post-norm residual connection.

        Computes::

            LayerNorm(x + sublayer_output)

        Args:
            x (torch.Tensor): The sub-layer's input, shape
                ``(B, seq_len, d_model)``.
            sublayer_output (torch.Tensor): The sub-layer's output, shape
                ``(B, seq_len, d_model)``.  Must match ``x`` in shape.

        Returns:
            torch.Tensor: Shape ``(B, seq_len, d_model)``.
                The layer-normalised sum of input and sublayer output.
        """
        # Post-norm: add first, then normalise.
        # Explicitly written as a single expression to make the formula
        # unambiguous and prevent accidental pre-norm rewrites.
        return self.layer_norm(x + sublayer_output)
