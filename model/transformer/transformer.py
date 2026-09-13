"""
model/transformer/transformer.py
=================================
Full Encoder-Decoder Transformer for the AttenSum project.

Implements the complete sequence-to-sequence Transformer architecture as described in:

    "Attention Is All You Need" — Vaswani et al., 2017 (§3.1)
    https://arxiv.org/abs/1706.03762

Architecture
------------
The model is composed of a source Encoder stack and a target Decoder stack,
with linear embedding, positional encoding, and output logit projection layers::

    SOURCE PIPELINE                              TARGET PIPELINE
    ===============                              ===============

    src_tokens (B, src_len)                      tgt_tokens (B, tgt_len)
          │                                            │
          ▼                                            ▼
    TokenEmbedding (src_vocab → d_model)         TokenEmbedding (tgt_vocab → d_model)
          │                                            │
          ▼                                            ▼
    PositionalEncoding                           PositionalEncoding
          │                                            │
          ▼                                            ▼
    Encoder Stack (N × EncoderBlock)             Decoder Stack (N × DecoderBlock)
          │                                            │
          │                                 (memory) ──┼───┐ (cross-attention)
          ▼                                            │   │
        memory (B, src_len, d_model) ──────────────────┘   │
                                                           ▼
                                                     Decoder Stack Output (B, tgt_len, d_model)
                                                           │
                                                           ▼
                                                     Linear Projection (d_model → tgt_vocab)
                                                           │
                                                           ▼
                                                     logits (B, tgt_len, tgt_vocab_size)

Composing Existing Components
-----------------------------
``Transformer`` does **not** reimplement lower-level mathematical logic.
It composes the existing, independently tested building blocks:

- :class:`~model.transformer.embeddings.TokenEmbedding`
- :class:`~model.transformer.embeddings.PositionalEncoding`
- :class:`~model.transformer.encoder.EncoderBlock`
- :class:`~model.transformer.decoder.DecoderBlock`
"""

from typing import Optional, Union

import torch
import torch.nn as nn

from model.transformer.decoder import DecoderBlock
from model.transformer.embeddings import PositionalEncoding, TokenEmbedding
from model.transformer.encoder import EncoderBlock


class Transformer(nn.Module):
    """Full Encoder-Decoder Transformer for sequence-to-sequence learning.

    Combines source embedding, positional encoding, an N-layer Encoder stack,
    target embedding, positional encoding, an N-layer Decoder stack, and a
    final linear output projection.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        d_model (int): Model dimensionality (width across all layers).
            Must be divisible by ``num_heads``.
        num_heads (int): Number of attention heads in each attention block.
        num_encoder_layers (int): Number of stacked ``EncoderBlock`` layers.
        num_decoder_layers (int): Number of stacked ``DecoderBlock`` layers.
        d_ff (int): Inner dimension of position-wise feed-forward networks.
        max_seq_len (int): Maximum sequence length supported by positional
            encoding. Defaults to ``5000``.
        dropout (float): Dropout probability used throughout embeddings,
            positional encoding, attention, and feed-forward sub-layers.
            Defaults to ``0.1``.
        src_pad_idx (int | None): Padding token index for source vocabulary.
            If provided, embedding vector at this index remains zero and
            receives no gradients. Defaults to ``None``.
        tgt_pad_idx (int | None): Padding token index for target vocabulary.
            Defaults to ``None``.

    Raises:
        ValueError: If any size or layer count argument is non-positive, if
            ``d_model % num_heads != 0``, or if ``dropout`` is not in ``[0.0, 1.0)``.

    Example::

        >>> model = Transformer(
        ...     src_vocab_size=1000,
        ...     tgt_vocab_size=1000,
        ...     d_model=256,
        ...     num_heads=8,
        ...     num_encoder_layers=4,
        ...     num_decoder_layers=4,
        ...     d_ff=1024,
        ... )
        >>> src = torch.randint(0, 1000, (2, 30))
        >>> tgt = torch.randint(0, 1000, (2, 20))
        >>> logits = model(src, tgt)
        >>> logits.shape
        torch.Size([2, 20, 1000])
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        d_ff: int,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
        src_pad_idx: Optional[int] = None,
        tgt_pad_idx: Optional[int] = None,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # Argument Validation
        # ------------------------------------------------------------------ #
        if src_vocab_size <= 0:
            raise ValueError(
                f"src_vocab_size must be a positive integer, got {src_vocab_size}."
            )
        if tgt_vocab_size <= 0:
            raise ValueError(
                f"tgt_vocab_size must be a positive integer, got {tgt_vocab_size}."
            )
        if d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {d_model}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, got {num_heads}.")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )
        if num_encoder_layers <= 0:
            raise ValueError(
                f"num_encoder_layers must be a positive integer, got {num_encoder_layers}."
            )
        if num_decoder_layers <= 0:
            raise ValueError(
                f"num_decoder_layers must be a positive integer, got {num_decoder_layers}."
            )
        if d_ff <= 0:
            raise ValueError(f"d_ff must be a positive integer, got {d_ff}.")
        if max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be a positive integer, got {max_seq_len}."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"dropout must be in the range [0.0, 1.0), got {dropout}."
            )

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout_p = dropout
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        # ------------------------------------------------------------------ #
        # Source Sub-network (Encoder components)
        # ------------------------------------------------------------------ #
        self.src_embedding = TokenEmbedding(
            vocab_size=src_vocab_size,
            d_model=d_model,
            padding_idx=src_pad_idx,
        )
        self.src_pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.encoder_layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # ------------------------------------------------------------------ #
        # Target Sub-network (Decoder components)
        # ------------------------------------------------------------------ #
        self.tgt_embedding = TokenEmbedding(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            padding_idx=tgt_pad_idx,
        )
        self.tgt_pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        # ------------------------------------------------------------------ #
        # Final Output Projection (Linear generator d_model → tgt_vocab_size)
        # ------------------------------------------------------------------ #
        self.projection = nn.Linear(d_model, tgt_vocab_size)

    def encode(
        self,
        src_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode a source sequence into contextual memory representations.

        Args:
            src_tokens (torch.Tensor): Integer token ids of shape
                ``(batch_size, src_len)``.
            src_mask (torch.Tensor | None): Optional encoder self-attention
                mask, broadcastable to ``(batch_size, num_heads, src_len, src_len)``.
                Defaults to ``None``.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]:
                * ``memory`` — Encoded representations of shape
                  ``(batch_size, src_len, d_model)``.
                * ``encoder_attentions`` — List of length ``num_encoder_layers``
                  containing per-layer self-attention weight tensors of shape
                  ``(batch_size, num_heads, src_len, src_len)``.
        """
        # Embed and add positional encoding: (B, src_len) → (B, src_len, d_model)
        x = self.src_embedding(src_tokens)
        x = self.src_pos_encoding(x)

        encoder_attentions = []
        for layer in self.encoder_layers:
            x, attn = layer(x, mask=src_mask)
            encoder_attentions.append(attn)

        return x, encoder_attentions

    def decode(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Decode target tokens given encoder memory representations.

        Args:
            tgt_tokens (torch.Tensor): Integer token ids of shape
                ``(batch_size, tgt_len)``.
            memory (torch.Tensor): Encoder output tensor of shape
                ``(batch_size, src_len, d_model)``.
            tgt_mask (torch.Tensor | None): Target self-attention (causal) mask.
                Defaults to ``None``.
            memory_mask (torch.Tensor | None): Cross-attention mask for memory.
                Defaults to ``None``.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
                * ``output`` — Decoder hidden states of shape
                  ``(batch_size, tgt_len, d_model)``.
                * ``decoder_self_attentions`` — List of length ``num_decoder_layers``
                  with tensors of shape ``(batch_size, num_heads, tgt_len, tgt_len)``.
                * ``decoder_cross_attentions`` — List of length ``num_decoder_layers``
                  with tensors of shape ``(batch_size, num_heads, tgt_len, src_len)``.
        """
        # Embed and add positional encoding: (B, tgt_len) → (B, tgt_len, d_model)
        x = self.tgt_embedding(tgt_tokens)
        x = self.tgt_pos_encoding(x)

        decoder_self_attentions = []
        decoder_cross_attentions = []

        for layer in self.decoder_layers:
            x, self_attn, cross_attn = layer(
                x,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )
            decoder_self_attentions.append(self_attn)
            decoder_cross_attentions.append(cross_attn)

        return x, decoder_self_attentions, decoder_cross_attentions

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> Union[
        torch.Tensor,
        tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
    ]:
        """Execute full end-to-end Transformer encoder-decoder forward pass.

        Args:
            src_tokens (torch.Tensor): Integer token ids of shape ``(B, src_len)``.
            tgt_tokens (torch.Tensor): Integer token ids of shape ``(B, tgt_len)``.
            src_mask (torch.Tensor | None): Optional encoder self-attention mask.
            tgt_mask (torch.Tensor | None): Optional target self-attention / causal mask.
            memory_mask (torch.Tensor | None): Optional encoder-decoder cross-attention mask.
            return_attentions (bool): If ``True``, returns attention weight matrices
                along with logits. Defaults to ``False``.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list, list, list]:
                * If ``return_attentions=False``:
                  ``logits`` tensor of shape ``(B, tgt_len, tgt_vocab_size)``.
                * If ``return_attentions=True``:
                  tuple ``(logits, encoder_attentions, decoder_self_attentions, decoder_cross_attentions)``.
        """
        memory, enc_attns = self.encode(src_tokens, src_mask=src_mask)
        dec_out, dec_self_attns, dec_cross_attns = self.decode(
            tgt_tokens,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )

        logits = self.projection(dec_out)

        if return_attentions:
            return logits, enc_attns, dec_self_attns, dec_cross_attns

        return logits
