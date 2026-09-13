"""
tests/test_transformer.py
==========================
Unit tests for ``model.transformer.transformer.Transformer``.

All tests run on CPU (no CUDA required).

Test coverage
-------------
Output shapes
    - Basic forward pass produces logits of shape (B, tgt_len, tgt_vocab_size).
    - Batched inputs with different batch sizes (B=1, B=8).
    - Mismatched source and target sequence lengths (src_len != tgt_len).
    - `encode()` method returns memory of shape (B, src_len, d_model).
    - `decode()` method returns hidden state of shape (B, tgt_len, d_model).

Attention weight collection
    - `return_attentions=True` returns logits and attention lists for encoder self-attention,
      decoder self-attention, and decoder cross-attention.
    - Attention tensor list lengths match layer counts (`num_encoder_layers`, `num_decoder_layers`).
    - Rectangular cross-attention weights shape: (B, num_heads, tgt_len, src_len).

Mask propagation
    - Encoder source mask (`src_mask`) zeroes attention to masked source positions.
    - Decoder target mask (`tgt_mask`) restricts attention (e.g. causal upper triangle).
    - Cross-attention memory mask (`memory_mask`) zeroes cross-attention to masked source positions.

Gradient flow
    - Loss computation and `loss.backward()` updates parameters across source embeddings,
      target embeddings, encoder blocks, decoder blocks, and final output projection.

Determinism & numerical stability
    - Model outputs are deterministic in `eval()` mode.
    - Output logits contain only finite float values (no NaN or Inf).

Constructor validation
    - Validates positive integers for vocab sizes, dimensions, head count, layer counts.
    - Validates `d_model % num_heads == 0`.
    - Validates valid dropout range [0, 1).
"""

import pytest
import torch
import torch.nn as nn

from model.transformer import Transformer


@pytest.fixture
def sample_config():
    """Default lightweight Transformer model configuration for fast testing."""
    return {
        "src_vocab_size": 100,
        "tgt_vocab_size": 120,
        "d_model": 64,
        "num_heads": 4,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "d_ff": 128,
        "max_seq_len": 200,
        "dropout": 0.0,
    }


@pytest.fixture
def transformer_model(sample_config):
    """Instantiate a lightweight Transformer instance with dropout=0.0."""
    return Transformer(**sample_config)


# ============================================================================ #
# 1. Output Shape & Forward Pass Tests
# ============================================================================ #

class TestTransformerShapes:
    """Tests for output tensor dimensions across various batch and sequence sizes."""

    def test_forward_logits_shape(self, transformer_model):
        """Basic forward pass produces logits of shape (B, tgt_len, tgt_vocab_size)."""
        transformer_model.eval()
        B, src_len, tgt_len = 2, 15, 10
        src_tokens = torch.randint(0, 100, (B, src_len))
        tgt_tokens = torch.randint(0, 120, (B, tgt_len))

        logits = transformer_model(src_tokens, tgt_tokens)

        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (B, tgt_len, 120)

    def test_single_batch(self, transformer_model):
        """Forward pass works with batch size = 1."""
        transformer_model.eval()
        src_tokens = torch.randint(0, 100, (1, 8))
        tgt_tokens = torch.randint(0, 120, (1, 5))

        logits = transformer_model(src_tokens, tgt_tokens)
        assert logits.shape == (1, 5, 120)

    def test_larger_batch(self, transformer_model):
        """Forward pass works with larger batch size B = 8."""
        transformer_model.eval()
        src_tokens = torch.randint(0, 100, (8, 12))
        tgt_tokens = torch.randint(0, 120, (8, 14))

        logits = transformer_model(src_tokens, tgt_tokens)
        assert logits.shape == (8, 14, 120)

    def test_mismatched_sequence_lengths(self, transformer_model):
        """Source and target sequences can have dramatically different lengths."""
        transformer_model.eval()
        src_tokens = torch.randint(0, 100, (3, 50))  # src_len = 50
        tgt_tokens = torch.randint(0, 120, (3, 7))   # tgt_len = 7

        logits = transformer_model(src_tokens, tgt_tokens)
        assert logits.shape == (3, 7, 120)

    def test_encode_standalone_method(self, transformer_model):
        """encode() returns memory of shape (B, src_len, d_model) and encoder attentions."""
        transformer_model.eval()
        B, src_len = 4, 16
        src_tokens = torch.randint(0, 100, (B, src_len))

        memory, enc_attns = transformer_model.encode(src_tokens)

        assert memory.shape == (B, src_len, 64)
        assert len(enc_attns) == 2  # num_encoder_layers = 2
        for attn in enc_attns:
            assert attn.shape == (B, 4, src_len, src_len)

    def test_decode_standalone_method(self, transformer_model):
        """decode() returns target representations and self/cross attention weight lists."""
        transformer_model.eval()
        B, src_len, tgt_len = 3, 20, 12
        tgt_tokens = torch.randint(0, 120, (B, tgt_len))
        memory = torch.randn(B, src_len, 64)

        dec_out, self_attns, cross_attns = transformer_model.decode(tgt_tokens, memory)

        assert dec_out.shape == (B, tgt_len, 64)
        assert len(self_attns) == 2  # num_decoder_layers = 2
        assert len(cross_attns) == 2

        for sa in self_attns:
            assert sa.shape == (B, 4, tgt_len, tgt_len)
        for ca in cross_attns:
            assert ca.shape == (B, 4, tgt_len, src_len)


# ============================================================================ #
# 2. Attention Weight Collection Tests
# ============================================================================ #

class TestAttentionCollection:
    """Tests for extracting per-head attention weight matrices from all layers."""

    def test_return_attentions_flag(self, transformer_model):
        """return_attentions=True returns (logits, enc_attns, dec_self_attns, dec_cross_attns)."""
        transformer_model.eval()
        B, src_len, tgt_len = 2, 10, 8
        src_tokens = torch.randint(0, 100, (B, src_len))
        tgt_tokens = torch.randint(0, 120, (B, tgt_len))

        logits, enc_attns, dec_self_attns, dec_cross_attns = transformer_model(
            src_tokens, tgt_tokens, return_attentions=True
        )

        assert logits.shape == (B, tgt_len, 120)
        assert len(enc_attns) == transformer_model.num_encoder_layers
        assert len(dec_self_attns) == transformer_model.num_decoder_layers
        assert len(dec_cross_attns) == transformer_model.num_decoder_layers

        for ea in enc_attns:
            assert ea.shape == (B, 4, src_len, src_len)
        for dsa in dec_self_attns:
            assert dsa.shape == (B, 4, tgt_len, tgt_len)
        for dca in dec_cross_attns:
            assert dca.shape == (B, 4, tgt_len, src_len)

    def test_attention_weights_sum_to_one(self, transformer_model):
        """Attention weights along key dimension sum to 1.0 for unmasked positions."""
        transformer_model.eval()
        src_tokens = torch.randint(0, 100, (2, 6))
        tgt_tokens = torch.randint(0, 120, (2, 5))

        _, enc_attns, dec_self_attns, dec_cross_attns = transformer_model(
            src_tokens, tgt_tokens, return_attentions=True
        )

        for ea in enc_attns:
            sum_keys = ea.sum(dim=-1)
            assert torch.allclose(sum_keys, torch.ones_like(sum_keys), atol=1e-5)

        for dsa in dec_self_attns:
            sum_keys = dsa.sum(dim=-1)
            assert torch.allclose(sum_keys, torch.ones_like(sum_keys), atol=1e-5)

        for dca in dec_cross_attns:
            sum_keys = dca.sum(dim=-1)
            assert torch.allclose(sum_keys, torch.ones_like(sum_keys), atol=1e-5)


# ============================================================================ #
# 3. Mask Propagation Tests
# ============================================================================ #

class TestMaskPropagation:
    """Tests that masks propagate correctly through encoder and decoder stacks."""

    def test_src_mask_propagation(self, transformer_model):
        """src_mask masks out designated source tokens in encoder self-attention."""
        transformer_model.eval()
        B, src_len, tgt_len = 1, 4, 3
        src_tokens = torch.randint(0, 100, (B, src_len))
        tgt_tokens = torch.randint(0, 120, (B, tgt_len))

        # Mask out position 3 (last token) in source
        src_mask = torch.tensor([[
            [False, False, False, True],
            [False, False, False, True],
            [False, False, False, True],
            [False, False, False, True],
        ]])  # (1, 4, 4)

        _, enc_attns, _, _ = transformer_model(
            src_tokens, tgt_tokens, src_mask=src_mask, return_attentions=True
        )

        for ea in enc_attns:
            # Column 3 (key=3) should receive near-zero attention from all queries
            masked_weights = ea[0, :, :, 3]
            assert torch.all(masked_weights < 1e-4)

    def test_tgt_causal_mask_propagation(self, transformer_model):
        """tgt_mask applies causal masking to decoder self-attention."""
        transformer_model.eval()
        B, src_len, tgt_len = 1, 3, 4
        src_tokens = torch.randint(0, 100, (B, src_len))
        tgt_tokens = torch.randint(0, 120, (B, tgt_len))

        # Upper triangular causal mask (True = masked out)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool), diagonal=1)

        _, _, dec_self_attns, _ = transformer_model(
            src_tokens, tgt_tokens, tgt_mask=tgt_mask, return_attentions=True
        )

        for dsa in dec_self_attns:
            # Upper triangle of key dimension should be near zero
            for i in range(tgt_len):
                for j in range(i + 1, tgt_len):
                    assert torch.all(dsa[0, :, i, j] < 1e-4)

    def test_memory_cross_attn_mask_propagation(self, transformer_model):
        """memory_mask masks out padded source positions during cross-attention."""
        transformer_model.eval()
        B, src_len, tgt_len = 1, 4, 3
        src_tokens = torch.randint(0, 100, (B, src_len))
        tgt_tokens = torch.randint(0, 120, (B, tgt_len))

        # Mask out source positions 2 and 3 (padding tokens)
        # Shape: (1, 1, tgt_len, src_len) or broadcastable
        memory_mask = torch.tensor([[
            [False, False, True, True],
            [False, False, True, True],
            [False, False, True, True],
        ]])  # (1, 3, 4)

        _, _, _, dec_cross_attns = transformer_model(
            src_tokens, tgt_tokens, memory_mask=memory_mask, return_attentions=True
        )

        for dca in dec_cross_attns:
            # Source positions 2 and 3 should have 0 cross-attention weight
            assert torch.all(dca[0, :, :, 2:] < 1e-4)


# ============================================================================ #
# 4. Gradient Flow Tests
# ============================================================================ #

class TestGradientFlow:
    """Tests end-to-end backpropagation through all Transformer layers."""

    def test_end_to_end_gradients(self, sample_config):
        """Gradients flow back to embeddings, encoder, decoder, and output projection."""
        model = Transformer(**sample_config)
        model.train()

        src_tokens = torch.randint(0, 100, (2, 8))
        tgt_tokens = torch.randint(0, 120, (2, 6))

        logits = model(src_tokens, tgt_tokens)
        target_labels = torch.randint(0, 120, (2, 6))

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, 120), target_labels.view(-1))
        loss.backward()

        # Check gradients in source and target embedding lookup tables
        assert model.src_embedding.embedding.weight.grad is not None
        assert torch.any(model.src_embedding.embedding.weight.grad != 0)

        assert model.tgt_embedding.embedding.weight.grad is not None
        assert torch.any(model.tgt_embedding.embedding.weight.grad != 0)

        # Check encoder layer 0 self-attention projections
        enc0 = model.encoder_layers[0]
        assert enc0.self_attn.w_q.weight.grad is not None
        assert torch.any(enc0.self_attn.w_q.weight.grad != 0)

        # Check decoder layer 0 cross-attention projections
        dec0 = model.decoder_layers[0]
        assert dec0.cross_attn.w_k.weight.grad is not None
        assert torch.any(dec0.cross_attn.w_k.weight.grad != 0)

        # Check output projection
        assert model.projection.weight.grad is not None
        assert torch.any(model.projection.weight.grad != 0)


# ============================================================================ #
# 5. Determinism & Output Quality Tests
# ============================================================================ #

class TestOutputProperties:
    """Tests for numerical determinism, finiteness, and evaluation state."""

    def test_eval_mode_determinism(self, sample_config):
        """In eval() mode, repeated passes with identical inputs produce identical outputs."""
        sample_config["dropout"] = 0.3
        model = Transformer(**sample_config)
        model.eval()

        src_tokens = torch.randint(0, 100, (2, 10))
        tgt_tokens = torch.randint(0, 120, (2, 8))

        with torch.no_grad():
            out1 = model(src_tokens, tgt_tokens)
            out2 = model(src_tokens, tgt_tokens)

        assert torch.equal(out1, out2)

    def test_outputs_are_finite(self, transformer_model):
        """Output logits do not contain NaN or Inf values."""
        transformer_model.eval()
        src_tokens = torch.randint(0, 100, (4, 15))
        tgt_tokens = torch.randint(0, 120, (4, 12))

        logits = transformer_model(src_tokens, tgt_tokens)

        assert torch.isfinite(logits).all()


# ============================================================================ #
# 6. Constructor Validation Tests
# ============================================================================ #

class TestConstructorValidation:
    """Tests that invalid parameter combinations raise descriptive ValueError."""

    def test_invalid_src_vocab_size(self):
        with pytest.raises(ValueError, match="src_vocab_size"):
            Transformer(src_vocab_size=0, tgt_vocab_size=10, d_model=64, num_heads=4, num_encoder_layers=2, num_decoder_layers=2, d_ff=128)

    def test_invalid_tgt_vocab_size(self):
        with pytest.raises(ValueError, match="tgt_vocab_size"):
            Transformer(src_vocab_size=10, tgt_vocab_size=-5, d_model=64, num_heads=4, num_encoder_layers=2, num_decoder_layers=2, d_ff=128)

    def test_invalid_d_model_not_divisible(self):
        with pytest.raises(ValueError, match="divisible by num_heads"):
            Transformer(src_vocab_size=10, tgt_vocab_size=10, d_model=65, num_heads=4, num_encoder_layers=2, num_decoder_layers=2, d_ff=128)

    def test_invalid_encoder_layers(self):
        with pytest.raises(ValueError, match="num_encoder_layers"):
            Transformer(src_vocab_size=10, tgt_vocab_size=10, d_model=64, num_heads=4, num_encoder_layers=0, num_decoder_layers=2, d_ff=128)

    def test_invalid_decoder_layers(self):
        with pytest.raises(ValueError, match="num_decoder_layers"):
            Transformer(src_vocab_size=10, tgt_vocab_size=10, d_model=64, num_heads=4, num_encoder_layers=2, num_decoder_layers=-1, d_ff=128)

    def test_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout"):
            Transformer(src_vocab_size=10, tgt_vocab_size=10, d_model=64, num_heads=4, num_encoder_layers=2, num_decoder_layers=2, d_ff=128, dropout=1.5)
