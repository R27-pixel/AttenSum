"""
tests/test_embeddings.py
=========================
Unit tests for ``model.transformer.embeddings``.

All tests run on CPU (no CUDA required).

Test coverage
-------------
TokenEmbedding
    - Output shape is (batch_size, seq_len, d_model).
    - Scaling by sqrt(d_model) is applied correctly.
    - padding_idx produces a zero embedding vector.
    - Invalid constructor arguments raise ValueError.

PositionalEncoding
    - Output shape matches input shape (batch_size, seq_len, d_model).
    - Works correctly for different sequence lengths (short, medium,
      and a length equal to max_seq_len).
    - Position 0 and position 1 produce distinct encoding vectors.
    - Even dimensions use sin, odd dimensions use cos (no dropout).
    - Requesting seq_len > max_seq_len raises ValueError.
    - Invalid constructor arguments raise ValueError.

Forward-pass correctness
    - TokenEmbedding + PositionalEncoding combined forward pass produces
      a tensor of the right shape and dtype.
    - The combined output is non-trivially modified (not all zeros or
      all ones), confirming both modules actually modify the tensor.
"""

import math

import pytest
import torch

from model.transformer.embeddings import PositionalEncoding, TokenEmbedding

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# TokenEmbedding tests
# ---------------------------------------------------------------------------


class TestTokenEmbedding:
    """Tests for TokenEmbedding."""

    def test_output_shape(self) -> None:
        """Embedding output must be (batch_size, seq_len, d_model)."""
        vocab_size, d_model = 500, 64
        batch_size, seq_len = 4, 30

        embed = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)
        out = embed(ids)

        assert out.shape == (batch_size, seq_len, d_model), (
            f"Expected shape {(batch_size, seq_len, d_model)}, got {out.shape}"
        )

    def test_output_dtype(self) -> None:
        """Output must be a float tensor (not long)."""
        embed = TokenEmbedding(vocab_size=100, d_model=16)
        ids = torch.randint(0, 100, (2, 10))
        out = embed(ids)
        assert out.dtype == torch.float32

    def test_scaling_by_sqrt_d_model(self) -> None:
        """Verify the sqrt(d_model) scaling factor is applied.

        Strategy: manually access the raw embedding weight for a known
        token id and compare it (scaled) to the module output.
        """
        vocab_size, d_model = 50, 32
        embed = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)

        token_id = 7
        ids = torch.tensor([[token_id]])  # (1, 1)
        out = embed(ids)  # (1, 1, d_model)

        # Raw weight vector for token_id
        raw_weight = embed.embedding.weight[token_id]  # (d_model,)
        expected = raw_weight * math.sqrt(d_model)

        assert torch.allclose(out[0, 0], expected), (
            "TokenEmbedding output does not match raw_weight * sqrt(d_model)."
        )

    def test_padding_idx_is_zero(self) -> None:
        """With padding_idx=0 the embedding for token 0 must be all zeros."""
        embed = TokenEmbedding(vocab_size=100, d_model=32, padding_idx=0)
        ids = torch.tensor([[0]])  # padding token
        out = embed(ids)  # (1, 1, 32)
        assert torch.all(out == 0.0), (
            "Embedding for padding_idx should be an all-zero vector."
        )

    def test_batch_independence(self) -> None:
        """The same token id in different batch items must produce the same vector."""
        vocab_size, d_model = 200, 48
        embed = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)

        ids = torch.tensor([[3, 7], [3, 9]])  # token 3 appears in both rows
        out = embed(ids)  # (2, 2, d_model)

        assert torch.allclose(out[0, 0], out[1, 0]), (
            "The same token id should produce identical embeddings across batch items."
        )

    def test_invalid_vocab_size_raises(self) -> None:
        """vocab_size <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="vocab_size"):
            TokenEmbedding(vocab_size=0, d_model=16)

    def test_invalid_d_model_raises(self) -> None:
        """d_model <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="d_model"):
            TokenEmbedding(vocab_size=100, d_model=-8)


# ---------------------------------------------------------------------------
# PositionalEncoding tests
# ---------------------------------------------------------------------------


class TestPositionalEncoding:
    """Tests for PositionalEncoding."""

    # ------------------------------------------------------------------
    # Shape tests
    # ------------------------------------------------------------------

    def test_output_shape_short_sequence(self) -> None:
        """Output shape must equal input shape for a short sequence."""
        d_model = 64
        pe = PositionalEncoding(d_model=d_model, max_seq_len=512, dropout=0.0)

        batch_size, seq_len = 3, 10
        x = torch.zeros(batch_size, seq_len, d_model, device=DEVICE)
        out = pe(x)

        assert out.shape == (batch_size, seq_len, d_model), (
            f"Expected {(batch_size, seq_len, d_model)}, got {out.shape}"
        )

    def test_output_shape_medium_sequence(self) -> None:
        """Output shape must equal input shape for a medium-length sequence."""
        d_model = 128
        pe = PositionalEncoding(d_model=d_model, max_seq_len=512, dropout=0.0)

        batch_size, seq_len = 8, 256
        x = torch.zeros(batch_size, seq_len, d_model)
        out = pe(x)

        assert out.shape == (batch_size, seq_len, d_model)

    def test_output_shape_equals_max_seq_len(self) -> None:
        """Output shape must equal input shape when seq_len == max_seq_len."""
        d_model, max_seq_len = 32, 100
        pe = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len, dropout=0.0)

        x = torch.zeros(2, max_seq_len, d_model)
        out = pe(x)

        assert out.shape == (2, max_seq_len, d_model)

    def test_output_shape_single_token(self) -> None:
        """Output shape must be correct for a sequence of length 1."""
        d_model = 16
        pe = PositionalEncoding(d_model=d_model, max_seq_len=50, dropout=0.0)

        x = torch.zeros(1, 1, d_model)
        out = pe(x)

        assert out.shape == (1, 1, d_model)

    # ------------------------------------------------------------------
    # Correctness tests — encoding values
    # ------------------------------------------------------------------

    def test_sin_on_even_dimensions(self) -> None:
        """Even-indexed dimensions of the encoding must follow sin."""
        d_model = 8
        pe_module = PositionalEncoding(d_model=d_model, max_seq_len=50, dropout=0.0)

        # Pass all-zeros so output == PE table values directly
        x = torch.zeros(1, 10, d_model)
        out = pe_module(x)  # (1, 10, 8)

        # Check position 0: sin(0) == 0 for all even dims
        assert torch.allclose(
            out[0, 0, 0::2],
            torch.zeros(d_model // 2),
            atol=1e-6,
        ), "Even-dimension values at position 0 should all be sin(0) = 0."

    def test_cos_on_odd_dimensions(self) -> None:
        """Odd-indexed dimensions of the encoding must follow cos."""
        d_model = 8
        pe_module = PositionalEncoding(d_model=d_model, max_seq_len=50, dropout=0.0)

        x = torch.zeros(1, 10, d_model)
        out = pe_module(x)  # (1, 10, 8)

        # Check position 0: cos(0) == 1 for all odd dims
        assert torch.allclose(
            out[0, 0, 1::2],
            torch.ones(d_model // 2),
            atol=1e-6,
        ), "Odd-dimension values at position 0 should all be cos(0) = 1."

    def test_different_positions_produce_different_encodings(self) -> None:
        """Distinct positions must yield distinct encoding vectors."""
        d_model = 32
        pe_module = PositionalEncoding(d_model=d_model, max_seq_len=100, dropout=0.0)

        x = torch.zeros(1, 10, d_model)
        out = pe_module(x)  # (1, 10, d_model)

        # Position 0 and position 1 encodings should differ
        assert not torch.allclose(out[0, 0, :], out[0, 1, :]), (
            "Position 0 and position 1 should produce different encoding vectors."
        )

    def test_encoding_values_match_formula(self) -> None:
        """Spot-check several positions and dimensions against the closed-form formula."""
        d_model = 16
        max_seq_len = 50
        pe_module = PositionalEncoding(
            d_model=d_model, max_seq_len=max_seq_len, dropout=0.0
        )

        x = torch.zeros(1, max_seq_len, d_model)
        out = pe_module(x)  # (1, max_seq_len, d_model)

        # Spot-check a few (pos, dim) pairs
        check_cases = [
            (0, 0),   # sin(0 / 10000^0) = sin(0) = 0.0
            (1, 0),   # sin(1 / 10000^0) = sin(1)
            (5, 2),   # sin(5 / 10000^(2/d_model))
            (3, 3),   # cos(3 / 10000^(2/d_model))
        ]
        for pos, dim in check_cases:
            i = dim // 2  # which frequency
            div = math.pow(10000.0, 2 * i / d_model)
            expected = math.sin(pos / div) if dim % 2 == 0 else math.cos(pos / div)
            actual = out[0, pos, dim].item()
            assert abs(actual - expected) < 1e-5, (
                f"PE mismatch at pos={pos}, dim={dim}: "
                f"expected {expected:.6f}, got {actual:.6f}"
            )

    def test_encoding_is_not_modified_by_input_values(self) -> None:
        """The positional encoding added must be identical regardless of input values.

        Because PE is fixed and additive, (output_A - output_B) should equal
        (input_A - input_B) for any two inputs with the same shape.
        """
        d_model, seq_len = 32, 20
        pe_module = PositionalEncoding(d_model=d_model, max_seq_len=100, dropout=0.0)

        x_a = torch.randn(1, seq_len, d_model)
        x_b = torch.randn(1, seq_len, d_model)

        out_a = pe_module(x_a)
        out_b = pe_module(x_b)

        assert torch.allclose(out_a - out_b, x_a - x_b, atol=1e-6), (
            "PE is additive and fixed, so (out_a - out_b) must equal (x_a - x_b)."
        )

    def test_seq_len_exceeds_max_raises(self) -> None:
        """Requesting seq_len > max_seq_len must raise ValueError."""
        pe = PositionalEncoding(d_model=32, max_seq_len=10, dropout=0.0)
        x = torch.zeros(1, 11, 32)  # seq_len 11 > max_seq_len 10
        with pytest.raises(ValueError, match="max_seq_len"):
            pe(x)

    # ------------------------------------------------------------------
    # Constructor validation tests
    # ------------------------------------------------------------------

    def test_invalid_d_model_raises(self) -> None:
        """d_model <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="d_model"):
            PositionalEncoding(d_model=0, max_seq_len=50)

    def test_invalid_max_seq_len_raises(self) -> None:
        """max_seq_len <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="max_seq_len"):
            PositionalEncoding(d_model=16, max_seq_len=0)

    def test_invalid_dropout_raises(self) -> None:
        """dropout outside [0.0, 1.0) should raise ValueError."""
        with pytest.raises(ValueError, match="dropout"):
            PositionalEncoding(d_model=16, max_seq_len=50, dropout=1.0)

    # ------------------------------------------------------------------
    # Dropout behavior
    # ------------------------------------------------------------------

    def test_dropout_zero_gives_deterministic_output(self) -> None:
        """With dropout=0.0, two forward passes must be identical."""
        d_model, seq_len = 32, 15
        pe = PositionalEncoding(d_model=d_model, max_seq_len=100, dropout=0.0)
        pe.eval()

        x = torch.randn(2, seq_len, d_model)
        out1 = pe(x)
        out2 = pe(x)

        assert torch.allclose(out1, out2), (
            "With dropout=0.0 the output should be deterministic."
        )


# ---------------------------------------------------------------------------
# Combined forward-pass correctness
# ---------------------------------------------------------------------------


class TestCombinedForwardPass:
    """Integration-style tests for TokenEmbedding + PositionalEncoding pipeline."""

    def test_combined_output_shape(self) -> None:
        """The composed pipeline must produce the correct output shape."""
        vocab_size, d_model = 1000, 128
        max_seq_len = 512
        batch_size, seq_len = 4, 60

        token_embed = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        pos_enc = PositionalEncoding(
            d_model=d_model, max_seq_len=max_seq_len, dropout=0.0
        )

        ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        out = pos_enc(token_embed(ids))

        assert out.shape == (batch_size, seq_len, d_model), (
            f"Expected {(batch_size, seq_len, d_model)}, got {out.shape}"
        )

    def test_combined_output_dtype(self) -> None:
        """The composed pipeline output must be float32."""
        token_embed = TokenEmbedding(vocab_size=200, d_model=32)
        pos_enc = PositionalEncoding(d_model=32, max_seq_len=100, dropout=0.0)

        ids = torch.randint(0, 200, (1, 20))
        out = pos_enc(token_embed(ids))

        assert out.dtype == torch.float32

    def test_combined_output_is_non_trivial(self) -> None:
        """Output must not be all-zeros or all-ones (sanity check)."""
        vocab_size, d_model = 500, 64
        token_embed = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        pos_enc = PositionalEncoding(d_model=d_model, max_seq_len=200, dropout=0.0)

        ids = torch.randint(1, vocab_size, (2, 50))  # avoid token 0 (padding)
        out = pos_enc(token_embed(ids))

        assert not torch.all(out == 0.0), "Output must not be all zeros."
        assert not torch.all(out == 1.0), "Output must not be all ones."

    def test_combined_runs_on_cpu(self) -> None:
        """All tensors must remain on CPU throughout the pipeline."""
        token_embed = TokenEmbedding(vocab_size=300, d_model=48)
        pos_enc = PositionalEncoding(d_model=48, max_seq_len=100, dropout=0.0)

        ids = torch.randint(0, 300, (2, 15))  # CPU tensor
        out = pos_enc(token_embed(ids))

        assert out.device.type == "cpu", (
            f"Expected CPU output, got device: {out.device}"
        )

    def test_combined_gradient_flows_through_embedding(self) -> None:
        """Gradients must flow back to the embedding weights."""
        vocab_size, d_model = 100, 16
        token_embed = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        pos_enc = PositionalEncoding(d_model=d_model, max_seq_len=50, dropout=0.0)

        ids = torch.randint(0, vocab_size, (1, 5))
        out = pos_enc(token_embed(ids))
        loss = out.sum()
        loss.backward()

        assert token_embed.embedding.weight.grad is not None, (
            "Gradient must flow back to the embedding weight matrix."
        )
        assert not torch.all(token_embed.embedding.weight.grad == 0), (
            "Embedding weight gradients must not all be zero."
        )
