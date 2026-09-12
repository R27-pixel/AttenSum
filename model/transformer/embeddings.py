"""
model/transformer/embeddings.py
================================
Foundational embedding components for the AttenSum Transformer.

Implements:
    - TokenEmbedding   : learnable lookup table mapping token ids → d_model vectors.
    - PositionalEncoding: fixed sinusoidal positional encoding that adds position
                          information to the token embeddings, as described in
                          "Attention Is All You Need" (Vaswani et al., 2017).

These modules are designed to be consumed by both the Encoder and the Decoder;
neither class is coupled to a specific vocabulary size, sequence length, or
model dimension — all such values are passed as constructor arguments.
"""

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Learnable token embedding layer.

    Maps a sequence of integer token ids to dense vectors of dimension
    ``d_model``.  The embedding weights are scaled by ``sqrt(d_model)``
    following the convention in the original Transformer paper, which
    prevents the positional encoding signal from being swamped by the
    embedding magnitudes.

    Args:
        vocab_size (int): Total number of tokens in the vocabulary.
            Determines the number of rows in the embedding lookup table.
        d_model (int): Dimensionality of each token embedding vector.
            Must match the model width used throughout the Transformer.
        padding_idx (int | None): If given, the embedding vector at this
            index is kept as an all-zero vector and receives no gradient
            updates.  Defaults to ``None`` (no padding token).

    Shape:
        - Input:  ``(batch_size, seq_len)`` — ``torch.long``
        - Output: ``(batch_size, seq_len, d_model)`` — ``torch.float``

    Example::

        >>> embed = TokenEmbedding(vocab_size=1000, d_model=128)
        >>> ids   = torch.randint(0, 1000, (2, 20))   # batch=2, seq=20
        >>> out   = embed(ids)
        >>> out.shape
        torch.Size([2, 20, 128])
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive integer, got {vocab_size}.")
        if d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {d_model}.")

        self.d_model = d_model
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed a batch of token id sequences.

        Args:
            token_ids (torch.Tensor): Integer tensor of shape
                ``(batch_size, seq_len)`` containing token indices.

        Returns:
            torch.Tensor: Float tensor of shape
                ``(batch_size, seq_len, d_model)`` where each id has
                been replaced by its scaled embedding vector.
        """
        # Scale embeddings by sqrt(d_model) — §3.4 of Vaswani et al. (2017)
        return self.embedding(token_ids) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    Adds a deterministic position signal to a sequence of embeddings so
    the model can distinguish tokens by their position.  The encoding is
    *not* learned; it is pre-computed once at construction time and
    stored as a buffer.

    For position ``pos`` and dimension ``i`` the encoding is:

    .. math::

        PE(pos, 2i)   &= \\sin\\!\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right) \\\\
        PE(pos, 2i+1) &= \\cos\\!\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)

    The buffer is pre-allocated up to ``max_seq_len`` positions.  At
    forward time only the first ``seq_len`` rows are sliced and added to
    the input.

    Args:
        d_model (int): Model dimensionality — must match ``TokenEmbedding.d_model``.
        max_seq_len (int): Maximum sequence length the module will ever
            be called with.  Pre-allocates the encoding table up to this
            length.  Sequences shorter than ``max_seq_len`` are handled
            automatically via slicing.  Defaults to ``5000``.
        dropout (float): Dropout probability applied to the combined
            (embedding + positional encoding) tensor.  Set to ``0.0`` to
            disable.  Defaults to ``0.1``.

    Shape:
        - Input:  ``(batch_size, seq_len, d_model)``
        - Output: ``(batch_size, seq_len, d_model)``

    Example::

        >>> pe  = PositionalEncoding(d_model=128, max_seq_len=512, dropout=0.0)
        >>> x   = torch.zeros(2, 20, 128)
        >>> out = pe(x)
        >>> out.shape
        torch.Size([2, 20, 128])
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {d_model}.")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be a positive integer, got {max_seq_len}.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")

        self.dropout = nn.Dropout(p=dropout)

        # Build the sinusoidal table: shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)  # (max_seq_len, d_model)

        # Column vector of position indices: shape (max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Divisor term: 10000^(2i/d_model) expressed in log-space for
        # numerical stability — one value per even dimension index.
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension: (1, max_seq_len, d_model) so broadcasting
        # with (batch_size, seq_len, d_model) works automatically.
        pe = pe.unsqueeze(0)

        # Register as a buffer — it will be part of the module state
        # (e.g. saved with the checkpoint) but will NOT be updated by
        # the optimizer.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input embeddings.

        Args:
            x (torch.Tensor): Embedding tensor of shape
                ``(batch_size, seq_len, d_model)``.

        Returns:
            torch.Tensor: Tensor of shape ``(batch_size, seq_len, d_model)``
                with positional information added and dropout applied.

        Raises:
            ValueError: If ``seq_len`` exceeds ``max_seq_len``.
        """
        seq_len = x.size(1)
        max_len = self.pe.size(1)  # type: ignore[union-attr]
        if seq_len > max_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds the pre-allocated "
                f"max_seq_len ({max_len}).  Increase max_seq_len at construction."
            )

        # self.pe is (1, max_seq_len, d_model); slice to actual seq_len
        x = x + self.pe[:, :seq_len, :]  # type: ignore[index]
        return self.dropout(x)
