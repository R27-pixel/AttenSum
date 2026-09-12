"""
tests/test_attention.py
========================
Unit tests for:
  - ``model.transformer.attention.ScaledDotProductAttention``
  - ``model.transformer.attention.MultiHeadAttention``

All tests run on CPU (no CUDA required).

Test coverage
-------------
Output shapes
    - Basic output shape (B, seq_q, d_v)
    - Attention weight shape (B, seq_q, seq_k)
    - Batched inputs with batch size > 1
    - Cross-attention scenario: seq_q ≠ seq_k (and d_k ≠ d_v)

Attention weight properties
    - Weights sum to 1.0 along key dimension (valid probability distribution)
    - Scaling by 1/sqrt(d_k) is applied (scores are smaller with larger d_k)

Mask behaviour
    - Bool mask: masked positions receive ~0 attention weight
    - Bool mask: unmasked positions share the full probability mass
    - Float additive mask: -inf entries produce ~0 attention weight
    - All-masked row produces NaN-free output (softmax of all -inf)

Numerical stability
    - Large Q, K, V values produce finite (non-NaN, non-Inf) output
    - Extreme input values do not cause overflow

Gradient flow
    - Gradients flow back to Q, K, and V after backward()
    - No gradient is expected / needed for the mask

Dropout
    - dropout=0.0 gives deterministic output in eval mode
    - dropout=0.0 in train mode also gives deterministic output (p=0 → no-op)

Constructor validation
    - dropout outside [0.0, 1.0) raises ValueError

Dimension validation
    - d_k mismatch between Q and K raises ValueError
    - seq_k mismatch between K and V raises ValueError
"""

import math

import pytest
import torch

from model.transformer.attention import MultiHeadAttention, ScaledDotProductAttention

# ---------------------------------------------------------------------------
# Constants used across tests — all are constructor arguments, not hardcoded
# project hyperparameters.
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")

# Typical small dimensions used in tests (kept small to keep tests fast).
B = 2        # batch size
SEQ_Q = 8    # query sequence length
SEQ_K = 10   # key / value sequence length
D_K = 32     # query / key depth
D_V = 48     # value depth  (intentionally ≠ D_K to catch shape bugs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_qkv(
    b: int = B,
    seq_q: int = SEQ_Q,
    seq_k: int = SEQ_K,
    d_k: int = D_K,
    d_v: int = D_V,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return random (Q, K, V) tensors on CPU."""
    q = torch.randn(b, seq_q, d_k, device=DEVICE, requires_grad=requires_grad)
    k = torch.randn(b, seq_k, d_k, device=DEVICE, requires_grad=requires_grad)
    v = torch.randn(b, seq_k, d_v, device=DEVICE, requires_grad=requires_grad)
    return q, k, v


# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------


class TestOutputShape:
    """Verify that forward() returns tensors with correct shapes."""

    def test_output_shape(self) -> None:
        """Context output must be (B, seq_q, d_v)."""
        attn = ScaledDotProductAttention(dropout=0.0)
        q, k, v = _make_qkv()
        output, _ = attn(q, k, v)
        assert output.shape == (B, SEQ_Q, D_V), (
            f"Expected output shape {(B, SEQ_Q, D_V)}, got {output.shape}"
        )

    def test_attention_weight_shape(self) -> None:
        """Attention weight matrix must be (B, seq_q, seq_k)."""
        attn = ScaledDotProductAttention(dropout=0.0)
        q, k, v = _make_qkv()
        _, weights = attn(q, k, v)
        assert weights.shape == (B, SEQ_Q, SEQ_K), (
            f"Expected weight shape {(B, SEQ_Q, SEQ_K)}, got {weights.shape}"
        )

    def test_batched_inputs(self) -> None:
        """Module must handle batch sizes larger than 1 without shape errors."""
        batch_sizes = [1, 4, 16]
        attn = ScaledDotProductAttention(dropout=0.0)
        for b in batch_sizes:
            q, k, v = _make_qkv(b=b)
            output, weights = attn(q, k, v)
            assert output.shape == (b, SEQ_Q, D_V)
            assert weights.shape == (b, SEQ_Q, SEQ_K)

    def test_cross_attention_shapes(self) -> None:
        """seq_q ≠ seq_k and d_k ≠ d_v must all be handled correctly.

        This simulates the cross-attention scenario in the Decoder where
        the query comes from the decoder sequence and the key/value come
        from the encoder output (potentially different length and depth).
        """
        attn = ScaledDotProductAttention(dropout=0.0)
        b, seq_q, seq_k, d_k, d_v = 3, 5, 12, 16, 64
        q = torch.randn(b, seq_q, d_k)
        k = torch.randn(b, seq_k, d_k)
        v = torch.randn(b, seq_k, d_v)
        output, weights = attn(q, k, v)
        assert output.shape == (b, seq_q, d_v)
        assert weights.shape == (b, seq_q, seq_k)

    def test_single_token_query(self) -> None:
        """seq_q = 1 (e.g. autoregressive decoding step) must work."""
        attn = ScaledDotProductAttention(dropout=0.0)
        q, k, v = _make_qkv(seq_q=1)
        output, weights = attn(q, k, v)
        assert output.shape == (B, 1, D_V)
        assert weights.shape == (B, 1, SEQ_K)

    def test_output_dtype(self) -> None:
        """Output tensors must be float32."""
        attn = ScaledDotProductAttention(dropout=0.0)
        q, k, v = _make_qkv()
        output, weights = attn(q, k, v)
        assert output.dtype == torch.float32
        assert weights.dtype == torch.float32


# ---------------------------------------------------------------------------
# Attention weight property tests
# ---------------------------------------------------------------------------


class TestAttentionWeightProperties:
    """Tests for correctness of the computed attention weight distributions."""

    def test_weights_sum_to_one(self) -> None:
        """Each query's attention distribution must sum to 1.0 (valid PMF)."""
        attn = ScaledDotProductAttention(dropout=0.0)
        q, k, v = _make_qkv()
        _, weights = attn(q, k, v)

        # Sum over the key dimension for every (batch, query_pos) pair
        row_sums = weights.sum(dim=-1)  # (B, seq_q)
        assert torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-5
        ), f"Attention weights must sum to 1; got min={row_sums.min():.6f}, max={row_sums.max():.6f}"

    def test_weights_non_negative(self) -> None:
        """All attention weights must be ≥ 0 (softmax output is non-negative)."""
        attn = ScaledDotProductAttention(dropout=0.0)
        q, k, v = _make_qkv()
        _, weights = attn(q, k, v)
        assert (weights >= 0).all(), "Attention weights must be non-negative."

    def test_scaling_reduces_score_magnitude(self) -> None:
        """Dividing by sqrt(d_k) must reduce score magnitudes compared to no scaling.

        Strategy: fix Q and K, compute the raw dot-product max, and verify
        the scaled version is strictly smaller when d_k > 1.
        """
        d_k = 64
        q = torch.randn(1, 4, d_k)
        k = torch.randn(1, 4, d_k)

        raw_scores = torch.matmul(q, k.transpose(-2, -1))
        scaled_scores = raw_scores / math.sqrt(d_k)

        # The scale factor is 1/sqrt(d_k) < 1 for d_k > 1
        assert scaled_scores.abs().max() < raw_scores.abs().max(), (
            "Scaled scores must have smaller magnitude than raw dot products."
        )

    def test_uniform_attention_for_identical_keys(self) -> None:
        """When all key vectors are identical, attention must be uniform."""
        d_k, d_v, seq_k = 16, 16, 6
        attn = ScaledDotProductAttention(dropout=0.0)

        q = torch.randn(1, 4, d_k)
        k = torch.ones(1, seq_k, d_k)   # all key vectors the same
        v = torch.randn(1, seq_k, d_v)

        _, weights = attn(q, k, v)
        # Every row should be 1/seq_k across all positions
        expected = torch.full((1, 4, seq_k), 1.0 / seq_k)
        assert torch.allclose(weights, expected, atol=1e-5), (
            "Identical keys must produce uniform attention weights."
        )


# ---------------------------------------------------------------------------
# Mask behaviour tests
# ---------------------------------------------------------------------------


class TestMaskBehaviour:
    """Tests that masking correctly suppresses attention to specific positions."""

    def test_bool_mask_zeroes_masked_weights(self) -> None:
        """Bool mask (True = mask) must produce ~0 weight at masked positions."""
        attn = ScaledDotProductAttention(dropout=0.0)
        q, k, v = _make_qkv(b=1, seq_q=4, seq_k=6, d_k=16, d_v=16)

        # Mask the last 2 key positions for all query positions
        mask = torch.zeros(1, 4, 6, dtype=torch.bool)
        mask[:, :, 4:] = True  # mask positions 4 and 5

        _, weights = attn(q, k, v, mask=mask)

        # Masked positions must be (approximately) zero
        assert torch.allclose(
            weights[:, :, 4:], torch.zeros(1, 4, 2), atol=1e-6
        ), "Masked positions should have ~0 attention weight."

    def test_bool_mask_unmasked_weights_sum_to_one(self) -> None:
        """Unmasked positions must still form a valid distribution summing to 1."""
        attn = ScaledDotProductAttention(dropout=0.0)
        q, k, v = _make_qkv(b=1, seq_q=4, seq_k=6, d_k=16, d_v=16)

        mask = torch.zeros(1, 4, 6, dtype=torch.bool)
        mask[:, :, 4:] = True  # mask last 2 positions

        _, weights = attn(q, k, v, mask=mask)

        # The remaining 4 positions must sum to 1
        unmasked_sum = weights[:, :, :4].sum(dim=-1)  # (1, 4)
        assert torch.allclose(unmasked_sum, torch.ones(1, 4), atol=1e-5), (
            "Unmasked positions must sum to 1 after masking."
        )

    def test_float_additive_mask(self) -> None:
        """Float mask with -inf at masked positions must produce ~0 weight there."""
        attn = ScaledDotProductAttention(dropout=0.0)
        q, k, v = _make_qkv(b=2, seq_q=5, seq_k=8, d_k=16, d_v=16)

        mask = torch.zeros(2, 5, 8)
        mask[:, :, 6:] = float("-inf")  # mask last 2 key positions

        _, weights = attn(q, k, v, mask=mask)

        assert torch.allclose(
            weights[:, :, 6:], torch.zeros(2, 5, 2), atol=1e-6
        ), "Float -inf mask must produce ~0 attention weight at masked positions."

    def test_causal_mask_upper_triangle(self) -> None:
        """Upper-triangular causal mask must prevent each position attending to future.

        In a causal (autoregressive) decoder the i-th query can only attend
        to positions 0 … i.  The upper-triangle (strictly above diagonal)
        must be masked.
        """
        seq = 6
        d_k = 16
        attn = ScaledDotProductAttention(dropout=0.0)

        q = torch.randn(1, seq, d_k)
        k = torch.randn(1, seq, d_k)
        v = torch.randn(1, seq, d_k)

        # Upper triangle (excluding diagonal) — positions i cannot see j > i
        causal_mask = torch.triu(
            torch.ones(1, seq, seq, dtype=torch.bool), diagonal=1
        )

        _, weights = attn(q, k, v, mask=causal_mask)

        # Every weight above the diagonal must be ~0
        upper = weights[0].triu(diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6), (
            "Causal mask must zero out above-diagonal attention weights."
        )

        # Every row must still sum to 1
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_no_mask_returns_finite_weights(self) -> None:
        """Without a mask all attention weights must be finite and positive."""
        attn = ScaledDotProductAttention(dropout=0.0)
        q, k, v = _make_qkv()
        _, weights = attn(q, k, v)
        assert torch.isfinite(weights).all(), "Unmasked weights must all be finite."
        assert (weights > 0).all(), "Without masking all weights must be > 0."


# ---------------------------------------------------------------------------
# Numerical stability tests
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Verify that attention remains numerically well-behaved under edge inputs."""

    def test_large_values_produce_finite_output(self) -> None:
        """Very large Q, K, V values must not produce NaN or Inf output."""
        attn = ScaledDotProductAttention(dropout=0.0)
        scale = 1e4
        q = torch.randn(B, SEQ_Q, D_K) * scale
        k = torch.randn(B, SEQ_K, D_K) * scale
        v = torch.randn(B, SEQ_K, D_V) * scale

        output, weights = attn(q, k, v)
        assert torch.isfinite(output).all(), "Output must be finite for large inputs."
        assert torch.isfinite(weights).all(), "Weights must be finite for large inputs."

    def test_zero_query_produces_finite_output(self) -> None:
        """Zero query tensor must not produce NaN or Inf output."""
        attn = ScaledDotProductAttention(dropout=0.0)
        q = torch.zeros(B, SEQ_Q, D_K)
        k = torch.randn(B, SEQ_K, D_K)
        v = torch.randn(B, SEQ_K, D_V)

        output, weights = attn(q, k, v)
        assert torch.isfinite(output).all()
        assert torch.isfinite(weights).all()

    def test_output_finite_with_large_d_k(self) -> None:
        """Large d_k (where unscaled scores would overflow) must stay finite."""
        attn = ScaledDotProductAttention(dropout=0.0)
        d_k_large = 2048
        q = torch.randn(1, 4, d_k_large)
        k = torch.randn(1, 4, d_k_large)
        v = torch.randn(1, 4, d_k_large)

        output, weights = attn(q, k, v)
        assert torch.isfinite(output).all()
        assert torch.isfinite(weights).all()


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify that gradients propagate back through Q, K, and V."""

    def test_gradient_flows_through_query(self) -> None:
        """After backward, query.grad must be non-None and non-zero."""
        attn = ScaledDotProductAttention(dropout=0.0)
        q, k, v = _make_qkv(requires_grad=True)
        output, _ = attn(q, k, v)
        output.sum().backward()

        assert q.grad is not None, "No gradient reached query."
        assert not torch.all(q.grad == 0), "Query gradient must not be all zeros."

    def test_gradient_flows_through_key(self) -> None:
        """After backward, key.grad must be non-None and non-zero."""
        attn = ScaledDotProductAttention(dropout=0.0)
        q, k, v = _make_qkv(requires_grad=True)
        output, _ = attn(q, k, v)
        output.sum().backward()

        assert k.grad is not None, "No gradient reached key."
        assert not torch.all(k.grad == 0), "Key gradient must not be all zeros."

    def test_gradient_flows_through_value(self) -> None:
        """After backward, value.grad must be non-None and non-zero."""
        attn = ScaledDotProductAttention(dropout=0.0)
        q, k, v = _make_qkv(requires_grad=True)
        output, _ = attn(q, k, v)
        output.sum().backward()

        assert v.grad is not None, "No gradient reached value."
        assert not torch.all(v.grad == 0), "Value gradient must not be all zeros."

    def test_gradient_flows_with_bool_mask(self) -> None:
        """Gradients must still flow through Q, K, V when a bool mask is applied."""
        attn = ScaledDotProductAttention(dropout=0.0)
        q, k, v = _make_qkv(b=1, seq_q=4, seq_k=6, d_k=16, d_v=16, requires_grad=True)

        mask = torch.zeros(1, 4, 6, dtype=torch.bool)
        mask[:, :, 5] = True  # mask last key position

        output, _ = attn(q, k, v, mask=mask)
        output.sum().backward()

        assert q.grad is not None and not torch.all(q.grad == 0)
        assert k.grad is not None and not torch.all(k.grad == 0)
        assert v.grad is not None and not torch.all(v.grad == 0)


# ---------------------------------------------------------------------------
# Dropout behaviour tests
# ---------------------------------------------------------------------------


class TestDropout:
    """Tests for the optional dropout applied to attention weights."""

    def test_no_dropout_deterministic_eval(self) -> None:
        """With dropout=0.0 the output must be identical across two forward passes."""
        attn = ScaledDotProductAttention(dropout=0.0)
        attn.eval()
        q, k, v = _make_qkv()
        out1, w1 = attn(q, k, v)
        out2, w2 = attn(q, k, v)
        assert torch.allclose(out1, out2)
        assert torch.allclose(w1, w2)

    def test_no_dropout_deterministic_train(self) -> None:
        """p=0 dropout is a mathematical no-op even in train mode."""
        attn = ScaledDotProductAttention(dropout=0.0)
        attn.train()
        q, k, v = _make_qkv()
        out1, _ = attn(q, k, v)
        out2, _ = attn(q, k, v)
        assert torch.allclose(out1, out2), (
            "dropout=0.0 must be deterministic even in train mode."
        )


# ---------------------------------------------------------------------------
# Constructor validation tests
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    """Invalid constructor arguments must raise ValueError."""

    def test_invalid_dropout_ge_one_raises(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            ScaledDotProductAttention(dropout=1.0)

    def test_invalid_dropout_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            ScaledDotProductAttention(dropout=-0.1)


# ---------------------------------------------------------------------------
# Dimension mismatch tests
# ---------------------------------------------------------------------------


class TestDimensionMismatch:
    """Mismatched tensor dimensions must raise descriptive ValueError."""

    def test_dk_mismatch_raises(self) -> None:
        """Q and K with different d_k must raise ValueError."""
        attn = ScaledDotProductAttention()
        q = torch.randn(2, 5, 32)
        k = torch.randn(2, 5, 16)  # d_k mismatch
        v = torch.randn(2, 5, 32)
        with pytest.raises(ValueError, match="d_k"):
            attn(q, k, v)

    def test_seqk_mismatch_between_k_and_v_raises(self) -> None:
        """K and V with different seq_k must raise ValueError."""
        attn = ScaledDotProductAttention()
        q = torch.randn(2, 5, 32)
        k = torch.randn(2, 6, 32)
        v = torch.randn(2, 7, 32)  # seq_k differs from K's seq_k
        with pytest.raises(ValueError, match="seq_k"):
            attn(q, k, v)


# ===========================================================================
# MultiHeadAttention tests
# ===========================================================================

# Small fixed dimensions used across MHA tests (all configurable, nothing
# specific to AttenSum's final hyperparameters).
MHA_D_MODEL = 64
MHA_HEADS = 4
MHA_HEAD_DIM = MHA_D_MODEL // MHA_HEADS   # 16
MHA_SEQ_Q = 10
MHA_SEQ_K = 12
MHA_BATCH = 3


def _make_mha_inputs(
    b: int = MHA_BATCH,
    seq_q: int = MHA_SEQ_Q,
    seq_k: int = MHA_SEQ_K,
    d_model: int = MHA_D_MODEL,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return random (query, key, value) tensors with d_model last dim."""
    q = torch.randn(b, seq_q, d_model, requires_grad=requires_grad)
    k = torch.randn(b, seq_k, d_model, requires_grad=requires_grad)
    v = torch.randn(b, seq_k, d_model, requires_grad=requires_grad)
    return q, k, v


class TestMHAOutputShapes:
    """Verify MultiHeadAttention returns the correct tensor shapes."""

    def test_output_shape(self) -> None:
        """Context output must be (B, seq_q, d_model)."""
        mha = MultiHeadAttention(d_model=MHA_D_MODEL, num_heads=MHA_HEADS, dropout=0.0)
        q, k, v = _make_mha_inputs()
        output, _ = mha(q, k, v)
        assert output.shape == (MHA_BATCH, MHA_SEQ_Q, MHA_D_MODEL)

    def test_attention_weight_shape(self) -> None:
        """Attention weights must be (B, num_heads, seq_q, seq_k)."""
        mha = MultiHeadAttention(d_model=MHA_D_MODEL, num_heads=MHA_HEADS, dropout=0.0)
        q, k, v = _make_mha_inputs()
        _, weights = mha(q, k, v)
        assert weights.shape == (MHA_BATCH, MHA_HEADS, MHA_SEQ_Q, MHA_SEQ_K)

    def test_output_dtype_is_float32(self) -> None:
        """Both output tensors must be float32."""
        mha = MultiHeadAttention(d_model=MHA_D_MODEL, num_heads=MHA_HEADS, dropout=0.0)
        q, k, v = _make_mha_inputs()
        output, weights = mha(q, k, v)
        assert output.dtype == torch.float32
        assert weights.dtype == torch.float32

    def test_multiple_batch_sizes(self) -> None:
        """Module must handle batch sizes 1, 4, and 16 without shape errors."""
        mha = MultiHeadAttention(d_model=MHA_D_MODEL, num_heads=MHA_HEADS, dropout=0.0)
        for b in [1, 4, 16]:
            q, k, v = _make_mha_inputs(b=b)
            output, weights = mha(q, k, v)
            assert output.shape == (b, MHA_SEQ_Q, MHA_D_MODEL)
            assert weights.shape == (b, MHA_HEADS, MHA_SEQ_Q, MHA_SEQ_K)

    def test_multiple_head_counts(self) -> None:
        """Module must work for different valid head counts.

        Tests 1, 2, 4, and 8 heads with d_model=64 (all evenly divide 64).
        """
        d_model = 64
        for h in [1, 2, 4, 8]:
            mha = MultiHeadAttention(d_model=d_model, num_heads=h, dropout=0.0)
            q, k, v = _make_mha_inputs(d_model=d_model)
            output, weights = mha(q, k, v)
            assert output.shape == (MHA_BATCH, MHA_SEQ_Q, d_model)
            assert weights.shape == (MHA_BATCH, h, MHA_SEQ_Q, MHA_SEQ_K)

    def test_single_head_matches_expected_shape(self) -> None:
        """With num_heads=1, weight shape must be (B, 1, seq_q, seq_k)."""
        mha = MultiHeadAttention(d_model=32, num_heads=1, dropout=0.0)
        q, k, v = _make_mha_inputs(d_model=32)
        _, weights = mha(q, k, v)
        assert weights.shape == (MHA_BATCH, 1, MHA_SEQ_Q, MHA_SEQ_K)


class TestMHASelfAttention:
    """Self-attention: query, key, and value are all the same tensor."""

    def test_self_attention_output_shape(self) -> None:
        """Self-attention output must be (B, seq, d_model)."""
        d_model, seq = 64, 15
        mha = MultiHeadAttention(d_model=d_model, num_heads=4, dropout=0.0)
        x = torch.randn(MHA_BATCH, seq, d_model)
        output, weights = mha(x, x, x)
        assert output.shape == (MHA_BATCH, seq, d_model)
        assert weights.shape == (MHA_BATCH, 4, seq, seq)

    def test_self_attention_weight_shape_square(self) -> None:
        """For self-attention, weight matrix must be square (seq, seq)."""
        seq = 8
        mha = MultiHeadAttention(d_model=MHA_D_MODEL, num_heads=MHA_HEADS, dropout=0.0)
        x = torch.randn(2, seq, MHA_D_MODEL)
        _, weights = mha(x, x, x)
        assert weights.shape[-2] == weights.shape[-1] == seq


class TestMHACrossAttention:
    """Cross-attention: query from one source, key/value from another.

    Simulates the decoder attending to encoder output where query length
    (decoder) differs from key/value length (encoder).
    """

    def test_cross_attention_output_shape(self) -> None:
        """Output shape must be (B, seq_q, d_model) even when seq_q ≠ seq_k."""
        d_model, seq_q, seq_k = 64, 7, 20
        mha = MultiHeadAttention(d_model=d_model, num_heads=4, dropout=0.0)
        q = torch.randn(MHA_BATCH, seq_q, d_model)
        k = torch.randn(MHA_BATCH, seq_k, d_model)
        v = torch.randn(MHA_BATCH, seq_k, d_model)
        output, weights = mha(q, k, v)
        assert output.shape == (MHA_BATCH, seq_q, d_model)
        assert weights.shape == (MHA_BATCH, 4, seq_q, seq_k)

    def test_cross_attention_weight_rows_sum_to_one(self) -> None:
        """Each query position's attention weights over all key positions must sum to 1."""
        d_model, seq_q, seq_k = 64, 5, 15
        mha = MultiHeadAttention(d_model=d_model, num_heads=4, dropout=0.0)
        q = torch.randn(2, seq_q, d_model)
        k = torch.randn(2, seq_k, d_model)
        v = torch.randn(2, seq_k, d_model)
        _, weights = mha(q, k, v)
        # weights: (B, H, seq_q, seq_k) — last dim must sum to 1
        row_sums = weights.sum(dim=-1)  # (B, H, seq_q)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestMHAWeightProperties:
    """Attention weight distribution properties for MultiHeadAttention."""

    def test_weights_sum_to_one_all_heads(self) -> None:
        """Every head's weights must form a valid probability distribution."""
        mha = MultiHeadAttention(d_model=MHA_D_MODEL, num_heads=MHA_HEADS, dropout=0.0)
        q, k, v = _make_mha_inputs()
        _, weights = mha(q, k, v)
        # weights: (B, H, seq_q, seq_k)
        row_sums = weights.sum(dim=-1)  # (B, H, seq_q)
        assert torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-5
        ), f"Weights must sum to 1; min={row_sums.min():.6f}, max={row_sums.max():.6f}"

    def test_weights_non_negative(self) -> None:
        """All per-head attention weights must be ≥ 0."""
        mha = MultiHeadAttention(d_model=MHA_D_MODEL, num_heads=MHA_HEADS, dropout=0.0)
        q, k, v = _make_mha_inputs()
        _, weights = mha(q, k, v)
        assert (weights >= 0).all()

    def test_weights_are_finite(self) -> None:
        """All attention weights must be finite (no NaN or Inf)."""
        mha = MultiHeadAttention(d_model=MHA_D_MODEL, num_heads=MHA_HEADS, dropout=0.0)
        q, k, v = _make_mha_inputs()
        _, weights = mha(q, k, v)
        assert torch.isfinite(weights).all()

    def test_output_is_finite(self) -> None:
        """Context output must be finite for typical random inputs."""
        mha = MultiHeadAttention(d_model=MHA_D_MODEL, num_heads=MHA_HEADS, dropout=0.0)
        q, k, v = _make_mha_inputs()
        output, _ = mha(q, k, v)
        assert torch.isfinite(output).all()


class TestMHAMaskBehaviour:
    """Verify that masks are correctly propagated through all heads."""

    def test_3d_bool_mask_zeroes_masked_positions(self) -> None:
        """A (B, seq_q, seq_k) bool mask must zero out masked positions in every head."""
        d_model, heads, seq_q, seq_k = 32, 2, 6, 8
        mha = MultiHeadAttention(d_model=d_model, num_heads=heads, dropout=0.0)
        q, k, v = _make_mha_inputs(b=1, seq_q=seq_q, seq_k=seq_k, d_model=d_model)

        # Mask the last 3 key positions for all query positions
        mask = torch.zeros(1, seq_q, seq_k, dtype=torch.bool)
        mask[:, :, 5:] = True

        _, weights = mha(q, k, v, mask=mask)
        # weights: (1, heads, seq_q, seq_k)
        assert torch.allclose(
            weights[:, :, :, 5:], torch.zeros(1, heads, seq_q, 3), atol=1e-6
        ), "Masked key positions must have ~0 weight in every head."

    def test_3d_mask_unmasked_sums_to_one(self) -> None:
        """Unmasked key positions must still form a valid distribution per head."""
        d_model, heads, seq_q, seq_k = 32, 2, 4, 8
        mha = MultiHeadAttention(d_model=d_model, num_heads=heads, dropout=0.0)
        q, k, v = _make_mha_inputs(b=2, seq_q=seq_q, seq_k=seq_k, d_model=d_model)

        mask = torch.zeros(2, seq_q, seq_k, dtype=torch.bool)
        mask[:, :, 6:] = True  # mask last 2 of 8 key positions

        _, weights = mha(q, k, v, mask=mask)
        unmasked_sum = weights[:, :, :, :6].sum(dim=-1)  # (B, H, seq_q)
        assert torch.allclose(unmasked_sum, torch.ones_like(unmasked_sum), atol=1e-5)

    def test_causal_mask_upper_triangle_all_heads(self) -> None:
        """Causal (upper-triangular) mask must zero future positions in every head."""
        seq = 8
        d_model, heads = 32, 4
        mha = MultiHeadAttention(d_model=d_model, num_heads=heads, dropout=0.0)
        x = torch.randn(1, seq, d_model)

        causal_mask = torch.triu(
            torch.ones(1, seq, seq, dtype=torch.bool), diagonal=1
        )  # (1, seq, seq)

        _, weights = mha(x, x, x, mask=causal_mask)
        # weights: (1, H, seq, seq) — upper triangle (excluding diagonal) must be ~0
        for h in range(heads):
            upper = weights[0, h].triu(diagonal=1)
            assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6), (
                f"Head {h}: causal mask must zero out above-diagonal weights."
            )

    def test_2d_mask_broadcasts_over_batch_and_heads(self) -> None:
        """A (seq_q, seq_k) mask must be broadcast over all batch items and heads."""
        seq_q, seq_k, d_model, heads = 5, 7, 32, 4
        mha = MultiHeadAttention(d_model=d_model, num_heads=heads, dropout=0.0)
        q, k, v = _make_mha_inputs(b=2, seq_q=seq_q, seq_k=seq_k, d_model=d_model)

        # 2-D mask: mask the last key position for every query
        mask = torch.zeros(seq_q, seq_k, dtype=torch.bool)
        mask[:, -1] = True

        _, weights = mha(q, k, v, mask=mask)
        # Last key column must be ~0 for all batches and heads
        assert torch.allclose(
            weights[:, :, :, -1], torch.zeros(2, heads, seq_q), atol=1e-6
        )

    def test_4d_per_head_mask(self) -> None:
        """A (B, H, seq_q, seq_k) mask allows different masks per head."""
        b, seq_q, seq_k, d_model, heads = 1, 4, 6, 32, 2
        mha = MultiHeadAttention(d_model=d_model, num_heads=heads, dropout=0.0)
        q, k, v = _make_mha_inputs(b=b, seq_q=seq_q, seq_k=seq_k, d_model=d_model)

        # Head 0: no masking.  Head 1: mask the last key position.
        mask = torch.zeros(b, heads, seq_q, seq_k, dtype=torch.bool)
        mask[:, 1, :, -1] = True  # only head 1, last key position

        _, weights = mha(q, k, v, mask=mask)
        # Head 1's last column must be ~0
        assert torch.allclose(
            weights[:, 1, :, -1], torch.zeros(b, seq_q), atol=1e-6
        ), "Per-head mask: head 1 last position must be ~0."
        # Head 0's last column must be > 0 (no masking)
        assert (weights[:, 0, :, -1] > 0).all(), (
            "Per-head mask: head 0 last position must be > 0 (unmasked)."
        )


class TestMHAGradientFlow:
    """Gradients must propagate through projections back to the raw inputs."""

    def test_grad_flows_to_query(self) -> None:
        mha = MultiHeadAttention(d_model=MHA_D_MODEL, num_heads=MHA_HEADS, dropout=0.0)
        q, k, v = _make_mha_inputs(requires_grad=True)
        output, _ = mha(q, k, v)
        output.sum().backward()
        assert q.grad is not None and not torch.all(q.grad == 0)

    def test_grad_flows_to_key(self) -> None:
        mha = MultiHeadAttention(d_model=MHA_D_MODEL, num_heads=MHA_HEADS, dropout=0.0)
        q, k, v = _make_mha_inputs(requires_grad=True)
        output, _ = mha(q, k, v)
        output.sum().backward()
        assert k.grad is not None and not torch.all(k.grad == 0)

    def test_grad_flows_to_value(self) -> None:
        mha = MultiHeadAttention(d_model=MHA_D_MODEL, num_heads=MHA_HEADS, dropout=0.0)
        q, k, v = _make_mha_inputs(requires_grad=True)
        output, _ = mha(q, k, v)
        output.sum().backward()
        assert v.grad is not None and not torch.all(v.grad == 0)

    def test_grad_flows_to_projection_weights(self) -> None:
        """All four projection matrices (W_Q, W_K, W_V, W_O) must receive gradients."""
        mha = MultiHeadAttention(d_model=MHA_D_MODEL, num_heads=MHA_HEADS, dropout=0.0)
        q, k, v = _make_mha_inputs()
        output, _ = mha(q, k, v)
        output.sum().backward()
        for name, param in mha.named_parameters():
            assert param.grad is not None, f"No gradient for parameter '{name}'."
            assert not torch.all(param.grad == 0), (
                f"Parameter '{name}' has all-zero gradients."
            )

    def test_grad_flows_with_mask(self) -> None:
        """Gradients must reach Q, K, V when a bool mask is applied."""
        d_model, heads, seq_q, seq_k = 32, 2, 5, 7
        mha = MultiHeadAttention(d_model=d_model, num_heads=heads, dropout=0.0)
        q, k, v = _make_mha_inputs(
            b=1, seq_q=seq_q, seq_k=seq_k, d_model=d_model, requires_grad=True
        )
        mask = torch.zeros(1, seq_q, seq_k, dtype=torch.bool)
        mask[:, :, -1] = True
        output, _ = mha(q, k, v, mask=mask)
        output.sum().backward()
        assert q.grad is not None and not torch.all(q.grad == 0)
        assert k.grad is not None and not torch.all(k.grad == 0)
        assert v.grad is not None and not torch.all(v.grad == 0)


class TestMHACPU:
    """Confirm all tensors stay on CPU throughout the MHA pipeline."""

    def test_output_on_cpu(self) -> None:
        mha = MultiHeadAttention(d_model=MHA_D_MODEL, num_heads=MHA_HEADS, dropout=0.0)
        q, k, v = _make_mha_inputs()
        output, weights = mha(q, k, v)
        assert output.device.type == "cpu"
        assert weights.device.type == "cpu"


class TestMHAConstructorValidation:
    """Invalid constructor arguments must raise descriptive ValueError."""

    def test_d_model_not_divisible_by_num_heads_raises(self) -> None:
        """d_model=65, num_heads=4 — 65 % 4 ≠ 0 — must raise ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadAttention(d_model=65, num_heads=4)

    def test_num_heads_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_heads"):
            MultiHeadAttention(d_model=64, num_heads=0)

    def test_d_model_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="d_model"):
            MultiHeadAttention(d_model=0, num_heads=4)

    def test_dropout_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            MultiHeadAttention(d_model=64, num_heads=4, dropout=1.0)

    def test_invalid_mask_dims_raises(self) -> None:
        """A 1-D mask (unsupported) must raise ValueError."""
        mha = MultiHeadAttention(d_model=32, num_heads=2)
        q, k, v = _make_mha_inputs(b=1, seq_q=4, seq_k=4, d_model=32)
        bad_mask = torch.zeros(4, dtype=torch.bool)  # 1-D — not supported
        with pytest.raises(ValueError, match="mask"):
            mha(q, k, v, mask=bad_mask)
