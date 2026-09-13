"""
model/transformer/feed_forward.py
===================================
Position-wise Feed-Forward Network for the AttenSum Transformer.

Implements the feed-forward sub-layer described in:

    "Attention Is All You Need" — Vaswani et al., 2017  (§3.3)
    https://arxiv.org/abs/1706.03762

The equation is::

    FFN(x) = activation(x W_1 + b_1) W_2 + b_2

where ``W_1 ∈ ℝ^{d_model × d_ff}`` expands the representation,
the activation introduces non-linearity, and
``W_2 ∈ ℝ^{d_ff × d_model}`` projects back to ``d_model``.

Key properties
--------------
* **Position-wise** — the same two-layer MLP is applied independently
  to every sequence position.  ``nn.Linear`` achieves this naturally
  because it operates on the last tensor dimension, leaving the batch
  and sequence dimensions untouched.

* **Shape-preserving** — input and output share the shape
  ``(B, seq_len, d_model)``.  The expansion to ``d_ff`` is internal
  and invisible to the caller.

* **Modular** — this module does *not* include a residual connection
  or layer normalisation.  Those belong to the surrounding encoder /
  decoder layer, following the principle of separation of concerns.

Activation default
------------------
The original Transformer uses ReLU.  More recent architectures (BERT,
GPT-2) prefer GELU.  The default is ``nn.ReLU()``; callers may pass
any ``nn.Module`` activation to override.
"""

from typing import Optional

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network.

    Applies the two-layer MLP::

        x → Linear(d_model → d_ff) → activation → Dropout → Linear(d_ff → d_model)

    independently to each position in the sequence.

    Tensor shapes
    -------------
    ::

        Input  : (B, seq_len, d_model)
             ↓  Linear  W_1 : (d_model → d_ff)
        Inner  : (B, seq_len, d_ff)
             ↓  activation
        Inner  : (B, seq_len, d_ff)
             ↓  Dropout (p=dropout, no-op when dropout=0.0 or eval mode)
        Inner  : (B, seq_len, d_ff)
             ↓  Linear  W_2 : (d_ff → d_model)
        Output : (B, seq_len, d_model)

    Dimension legend
    ----------------
    B       — batch size
    seq_len — sequence length (arbitrary; this module is length-agnostic)
    d_model — model width shared across all Transformer sub-layers
    d_ff    — inner (expanded) dimension, typically 4 × d_model

    Args:
        d_model    (int):        Model width — must be a positive integer.
        d_ff       (int):        Inner dimension — must be a positive integer.
                                 The original paper uses d_ff = 4 × d_model.
        dropout    (float):      Dropout probability applied to the inner
                                 representation after the activation.
                                 Set to ``0.0`` to disable (default).
        activation (nn.Module | None):
                                 Activation function applied between the two
                                 linear layers.  Any ``nn.Module`` that
                                 accepts a tensor and returns a tensor of the
                                 same shape is accepted (e.g. ``nn.ReLU()``,
                                 ``nn.GELU()``, ``nn.SiLU()``).
                                 Defaults to ``nn.ReLU()`` as in the
                                 original Transformer paper.
        bias       (bool):       Whether to include bias terms in both
                                 linear layers.  Defaults to ``True``.

    Raises:
        ValueError: If ``d_model`` or ``d_ff`` is not a positive integer,
            or if ``dropout`` is not in ``[0.0, 1.0)``.

    Example::

        >>> ff = PositionwiseFeedForward(d_model=512, d_ff=2048, dropout=0.1)
        >>> x  = torch.randn(2, 30, 512)   # (batch=2, seq=30, d_model=512)
        >>> ff(x).shape
        torch.Size([2, 30, 512])
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: Optional[nn.Module] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # Constructor validation
        # ------------------------------------------------------------------ #
        if d_model <= 0:
            raise ValueError(
                f"d_model must be a positive integer, got {d_model}."
            )
        if d_ff <= 0:
            raise ValueError(
                f"d_ff must be a positive integer, got {d_ff}."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0.0, 1.0), got {dropout}."
            )

        self.d_model = d_model
        self.d_ff = d_ff

        # ------------------------------------------------------------------ #
        # Sub-layers
        # ------------------------------------------------------------------ #

        # W_1: expands each position's representation from d_model → d_ff.
        # Applied to the last dimension, so batch and seq dimensions are
        # untouched — this is what makes the network "position-wise".
        self.linear_1 = nn.Linear(d_model, d_ff, bias=bias)

        # Activation — default is ReLU per the original Transformer paper.
        # A fresh nn.ReLU() is created per instance to avoid shared state
        # issues if the caller reuses the same activation object.
        self.activation: nn.Module = activation if activation is not None else nn.ReLU()

        # Dropout applied to the inner (post-activation) representation.
        # nn.Dropout is a no-op in eval mode regardless of p.
        self.dropout = nn.Dropout(p=dropout)

        # W_2: projects the expanded representation back to d_model.
        self.linear_2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the position-wise feed-forward network.

        The same two-layer MLP is applied to every position independently.
        PyTorch's ``nn.Linear`` naturally handles this because it applies
        the transformation to the last dimension of any input tensor,
        regardless of how many leading dimensions exist.

        Args:
            x (torch.Tensor): Input of shape ``(B, seq_len, d_model)``.

        Returns:
            torch.Tensor: Output of shape ``(B, seq_len, d_model)``.
                Same shape as the input.
        """
        # x       : (B, seq_len, d_model)

        x = self.linear_1(x)
        # x       : (B, seq_len, d_ff)     — expanded representation

        x = self.activation(x)
        # x       : (B, seq_len, d_ff)     — non-linearity applied per position

        x = self.dropout(x)
        # x       : (B, seq_len, d_ff)     — regularisation (no-op when p=0)

        x = self.linear_2(x)
        # x       : (B, seq_len, d_model)  — projected back to model width

        return x
