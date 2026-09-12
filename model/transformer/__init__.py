"""
model/transformer/__init__.py
==============================
Public API for the ``model.transformer`` package.

Higher-level modules (Encoder, Decoder, full Transformer) can import
any component directly from this package::

    from model.transformer import TokenEmbedding, PositionalEncoding
    from model.transformer import ScaledDotProductAttention
"""

from model.transformer.attention import MultiHeadAttention, ScaledDotProductAttention
from model.transformer.embeddings import PositionalEncoding, TokenEmbedding

__all__ = [
    "TokenEmbedding",
    "PositionalEncoding",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
]
