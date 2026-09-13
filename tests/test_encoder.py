"""
tests/test_encoder.py
======================
Unit tests for ``model.transformer.encoder.EncoderBlock``.

All tests run on CPU (no CUDA required).

Test coverage
-------------
Output shapes
    - Basic output shape (B, seq_len, d_model)
    - Attention weight shape (B, num_heads, seq_len, seq_len)
    - Different batch sizes
    - Different sequence lengths
    - Different d_model values
    - Different num_heads values
    - Different d_ff values

Component composition
    - self_attn is MultiHeadAttention
    - ffn is PositionwiseFeedForward
    - residual_1 and residual_2 are ResidualConnection instances

Post-norm ordering
    - Stage 1 is LayerNorm(x + SelfAttention(x,x,x))
    - Stage 2 is LayerNorm(x1 + FFN(x1))
    - Verify output != pre-norm variant

Mask support
    - Bool padding mask zeroes out masked attention weights
    - Causal upper-triangular mask prevents future attention

Output properties
    - Output is finite
    - Output dtype is float32
    - Output is on CPU
    - Output mean near zero per position (post-norm property)

Dropout behavior
    - dropout=0.0 gives identical output in eval and train mode

Gradient flow
    - Gradient reaches the input tensor
    - Gradients reach all named parameters (both sub-layers)

Constructor validation
    - d_model not divisible by num_heads raises ValueError
    - d_model = 0 raises ValueError
    - d_ff = 0 raises ValueError
    - dropout out of range raises ValueError
"""

from typing import Optional

import pytest
import torch
import torch.nn as nn

from model.transformer.attention import MultiHeadAttention
from model.transformer.encoder import EncoderBlock
from model.transformer.feed_forward import PositionwiseFeedForward
from model.transformer.normalization import ResidualConnection

# ---------------------------------------------------------------------------
# Shared constants — small test dimensions, not AttenSum's final values.
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")
D_MODEL = 64
NUM_HEADS = 4
D_FF = 256
BATCH = 3
SEQ_LEN = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_enc(
    d_model: int = D_MODEL,
    num_heads: int = NUM_HEADS,
    d_ff: int = D_FF,
    dropout: float = 0.0,
) -> EncoderBlock:
    return EncoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
    )


def _make_x(
    batch: int = BATCH,
    seq_len: int = SEQ_LEN,
    d_model: int = D_MODEL,
    requires_grad: bool = False,
) -> torch.Tensor:
    return torch.randn(
        batch, seq_len, d_model,
        device=DEVICE,
        requires_grad=requires_grad,
    )


# ===========================================================================
# Output shape tests
# ===========================================================================


class TestEncoderOutputShape:
    """The encoder block must preserve (B, seq_len, d_model) shape."""

    def test_basic_output_shape(self) -> None:
        enc = _make_enc()
        out, _ = enc(_make_x())
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_attention_weight_shape(self) -> None:
        """Attention weights must be (B, num_heads, seq_len, seq_len)."""
        enc = _make_enc()
        _, weights = enc(_make_x())
        assert weights.shape == (BATCH, NUM_HEADS, SEQ_LEN, SEQ_LEN)

    def test_batch_size_1(self) -> None:
        enc = _make_enc()
        out, w = enc(_make_x(batch=1))
        assert out.shape == (1, SEQ_LEN, D_MODEL)
        assert w.shape == (1, NUM_HEADS, SEQ_LEN, SEQ_LEN)

    def test_batch_size_8(self) -> None:
        enc = _make_enc()
        out, w = enc(_make_x(batch=8))
        assert out.shape == (8, SEQ_LEN, D_MODEL)
        assert w.shape == (8, NUM_HEADS, SEQ_LEN, SEQ_LEN)

    def test_seq_len_1(self) -> None:
        """Single-token sequence must be handled correctly."""
        enc = _make_enc()
        out, w = enc(_make_x(seq_len=1))
        assert out.shape == (BATCH, 1, D_MODEL)
        assert w.shape == (BATCH, NUM_HEADS, 1, 1)

    def test_seq_len_50(self) -> None:
        enc = _make_enc()
        out, w = enc(_make_x(seq_len=50))
        assert out.shape == (BATCH, 50, D_MODEL)
        assert w.shape == (BATCH, NUM_HEADS, 50, 50)

    def test_different_d_model_32(self) -> None:
        enc = _make_enc(d_model=32, num_heads=4, d_ff=128)
        out, w = enc(_make_x(d_model=32))
        assert out.shape == (BATCH, SEQ_LEN, 32)
        assert w.shape == (BATCH, 4, SEQ_LEN, SEQ_LEN)

    def test_different_d_model_128(self) -> None:
        enc = _make_enc(d_model=128, num_heads=8, d_ff=512)
        out, w = enc(_make_x(d_model=128))
        assert out.shape == (BATCH, SEQ_LEN, 128)
        assert w.shape == (BATCH, 8, SEQ_LEN, SEQ_LEN)

    def test_num_heads_1(self) -> None:
        enc = _make_enc(d_model=32, num_heads=1, d_ff=64)
        out, w = enc(_make_x(d_model=32))
        assert out.shape == (BATCH, SEQ_LEN, 32)
        assert w.shape == (BATCH, 1, SEQ_LEN, SEQ_LEN)

    def test_num_heads_8(self) -> None:
        enc = _make_enc(d_model=64, num_heads=8, d_ff=256)
        out, w = enc(_make_x(d_model=64))
        assert out.shape == (BATCH, SEQ_LEN, 64)
        assert w.shape == (BATCH, 8, SEQ_LEN, SEQ_LEN)

    def test_different_d_ff(self) -> None:
        """d_ff = d_model (no expansion) must work."""
        enc = _make_enc(d_model=64, num_heads=4, d_ff=64)
        out, _ = enc(_make_x())
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_d_ff_large(self) -> None:
        enc = _make_enc(d_model=64, num_heads=4, d_ff=1024)
        out, _ = enc(_make_x())
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


# ===========================================================================
# Component composition tests
# ===========================================================================


class TestComponentComposition:
    """Verify that the encoder wires the correct existing module types."""

    def test_self_attn_is_multihead_attention(self) -> None:
        enc = _make_enc()
        assert isinstance(enc.self_attn, MultiHeadAttention), (
            "enc.self_attn must be a MultiHeadAttention instance."
        )

    def test_ffn_is_positionwise_feed_forward(self) -> None:
        enc = _make_enc()
        assert isinstance(enc.ffn, PositionwiseFeedForward), (
            "enc.ffn must be a PositionwiseFeedForward instance."
        )

    def test_residual_1_is_residual_connection(self) -> None:
        enc = _make_enc()
        assert isinstance(enc.residual_1, ResidualConnection), (
            "enc.residual_1 must be a ResidualConnection instance."
        )

    def test_residual_2_is_residual_connection(self) -> None:
        enc = _make_enc()
        assert isinstance(enc.residual_2, ResidualConnection), (
            "enc.residual_2 must be a ResidualConnection instance."
        )

    def test_residual_1_and_2_are_independent(self) -> None:
        """The two ResidualConnections must be separate instances with
        independent LayerNorm parameters."""
        enc = _make_enc()
        assert enc.residual_1 is not enc.residual_2, (
            "residual_1 and residual_2 must be distinct module instances."
        )
        # Independent parameters — modifying one must not affect the other
        assert enc.residual_1.layer_norm.norm.weight.data_ptr() != \
               enc.residual_2.layer_norm.norm.weight.data_ptr(), (
            "residual_1 and residual_2 must have independent LayerNorm weights."
        )


# ===========================================================================
# Post-norm ordering tests
# ===========================================================================


class TestPostNormOrdering:
    """Verify the exact post-norm equations are applied."""

    def test_stage1_postnorm_formula(self) -> None:
        """Stage 1 must compute LayerNorm(x + SelfAttention(x, x, x)).

        Strategy: run the encoder sub-components manually and compare with
        the full encoder forward pass.
        """
        enc = _make_enc()
        enc.eval()

        x = _make_x()

        # Full encoder forward
        output, _ = enc(x)

        # Reproduce Stage 1 manually using the same module instances
        with torch.no_grad():
            attn_out, _ = enc.self_attn(x, x, x, mask=None)
            x1_expected = enc.residual_1(x, attn_out)

            # Continue to Stage 2 to get the final output
            ffn_out = enc.ffn(x1_expected)
            output_expected = enc.residual_2(x1_expected, ffn_out)

        assert torch.allclose(output, output_expected, atol=1e-5), (
            "EncoderBlock output must match manual post-norm computation."
        )

    def test_stage1_is_postnorm_not_prenorm(self) -> None:
        """Verify Stage 1 is LayerNorm(x + attn_out), NOT x + LayerNorm(attn_out)."""
        enc = _make_enc()
        enc.eval()

        torch.manual_seed(99)
        x = torch.randn(1, 6, D_MODEL)

        with torch.no_grad():
            attn_out, _ = enc.self_attn(x, x, x, mask=None)

            # Post-norm (correct): what residual_1 computes
            postnorm = enc.residual_1.layer_norm(x + attn_out)

            # Pre-norm (incorrect): what residual_1 must NOT compute
            prenorm = x + enc.residual_1.layer_norm(attn_out)

        # The two formulas produce different results (for non-trivial x, attn_out)
        assert not torch.allclose(postnorm, prenorm, atol=1e-3), (
            "Post-norm and pre-norm must differ — confirms the formula is post-norm."
        )

        # The encoder output's Stage 1 intermediate matches post-norm, not pre-norm
        with torch.no_grad():
            attn_out2, _ = enc.self_attn(x, x, x, mask=None)
            x1_actual = enc.residual_1(x, attn_out2)
            postnorm2 = enc.residual_1.layer_norm(x + attn_out2)

        assert torch.allclose(x1_actual, postnorm2, atol=1e-6), (
            "residual_1(x, attn_out) must equal LayerNorm(x + attn_out)."
        )

    def test_stage2_is_postnorm(self) -> None:
        """Verify Stage 2 is LayerNorm(x1 + ffn_out), NOT x1 + LayerNorm(ffn_out)."""
        enc = _make_enc()
        enc.eval()

        torch.manual_seed(42)
        x = torch.randn(1, 6, D_MODEL)

        with torch.no_grad():
            attn_out, _ = enc.self_attn(x, x, x, mask=None)
            x1 = enc.residual_1(x, attn_out)
            ffn_out = enc.ffn(x1)

            output_actual = enc.residual_2(x1, ffn_out)
            postnorm = enc.residual_2.layer_norm(x1 + ffn_out)

        assert torch.allclose(output_actual, postnorm, atol=1e-6), (
            "residual_2(x1, ffn_out) must equal LayerNorm(x1 + ffn_out)."
        )


# ===========================================================================
# Mask support tests
# ===========================================================================


class TestMaskSupport:
    """Masks must be correctly forwarded to the self-attention sub-layer."""

    def test_padding_mask_zeroes_masked_attention(self) -> None:
        """Bool mask must produce ~0 attention weights at masked positions."""
        enc = _make_enc()
        enc.eval()

        x = _make_x(batch=1, seq_len=8)

        # Mask the last 3 key positions for all query positions
        mask = torch.zeros(1, 8, 8, dtype=torch.bool)
        mask[:, :, 5:] = True

        _, weights = enc(x, mask=mask)
        # weights: (1, num_heads, 8, 8)
        assert torch.allclose(
            weights[:, :, :, 5:],
            torch.zeros(1, NUM_HEADS, 8, 3),
            atol=1e-6,
        ), "Masked key positions must have ~0 weight across all heads."

    def test_causal_mask_prevents_future_attention(self) -> None:
        """Upper-triangular causal mask must block attending to future positions."""
        enc = _make_enc()
        enc.eval()

        seq = 8
        x = _make_x(batch=1, seq_len=seq)

        causal_mask = torch.triu(
            torch.ones(1, seq, seq, dtype=torch.bool), diagonal=1
        )

        _, weights = enc(x, mask=causal_mask)
        # weights: (1, num_heads, seq, seq)
        for h in range(NUM_HEADS):
            upper = weights[0, h].triu(diagonal=1)
            assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6), (
                f"Head {h}: causal mask must zero out above-diagonal weights."
            )

    def test_no_mask_attends_everywhere(self) -> None:
        """Without a mask every attention weight must be > 0."""
        enc = _make_enc()
        enc.eval()
        _, weights = enc(_make_x())
        assert (weights > 0).all(), (
            "Without a mask, all attention weights must be positive."
        )

    def test_masked_weights_still_sum_to_one(self) -> None:
        """Unmasked positions must still form a valid probability distribution."""
        enc = _make_enc()
        enc.eval()

        x = _make_x(batch=2, seq_len=6)
        mask = torch.zeros(2, 6, 6, dtype=torch.bool)
        mask[:, :, 4:] = True   # mask last 2 key positions

        _, weights = enc(x, mask=mask)
        # Unmasked positions (first 4) must sum to 1 per (batch, head, query)
        unmasked_sum = weights[:, :, :, :4].sum(dim=-1)
        assert torch.allclose(
            unmasked_sum, torch.ones_like(unmasked_sum), atol=1e-5
        )


# ===========================================================================
# Output property tests
# ===========================================================================


class TestOutputProperties:

    def test_output_is_finite(self) -> None:
        enc = _make_enc()
        out, weights = enc(_make_x())
        assert torch.isfinite(out).all()
        assert torch.isfinite(weights).all()

    def test_output_dtype_float32(self) -> None:
        enc = _make_enc()
        out, weights = enc(_make_x())
        assert out.dtype == torch.float32
        assert weights.dtype == torch.float32

    def test_output_on_cpu(self) -> None:
        enc = _make_enc()
        out, weights = enc(_make_x())
        assert out.device.type == "cpu"
        assert weights.device.type == "cpu"

    def test_output_mean_near_zero(self) -> None:
        """Final output is produced by LayerNorm, so each position's mean ≈ 0."""
        enc = _make_enc()
        x = _make_x() * 5.0
        out, _ = enc(x)
        per_pos_mean = out.mean(dim=-1)   # (B, seq_len)
        assert torch.allclose(
            per_pos_mean, torch.zeros_like(per_pos_mean), atol=1e-5
        ), "Encoder output mean per position must be ≈ 0 (post-norm property)."

    def test_output_not_all_zeros(self) -> None:
        enc = _make_enc()
        out, _ = enc(_make_x())
        assert not torch.all(out == 0.0)


# ===========================================================================
# Dropout behavior tests
# ===========================================================================


class TestDropoutBehavior:

    def test_no_dropout_deterministic_eval(self) -> None:
        """dropout=0.0 in eval mode must give identical outputs."""
        enc = _make_enc(dropout=0.0)
        enc.eval()
        x = _make_x()
        out1, w1 = enc(x)
        out2, w2 = enc(x)
        assert torch.allclose(out1, out2)
        assert torch.allclose(w1, w2)

    def test_no_dropout_deterministic_train(self) -> None:
        """p=0 is a no-op even in train mode."""
        enc = _make_enc(dropout=0.0)
        enc.train()
        x = _make_x()
        out1, _ = enc(x)
        out2, _ = enc(x)
        assert torch.allclose(out1, out2), (
            "dropout=0.0 must be deterministic in train mode too."
        )

    def test_with_dropout_output_shape_unchanged(self) -> None:
        """Enabling dropout must not change the output shape."""
        enc = _make_enc(dropout=0.1)
        out, _ = enc(_make_x())
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


# ===========================================================================
# Gradient flow tests
# ===========================================================================


class TestGradientFlow:
    """Gradients must propagate through all sub-layers back to the input."""

    def test_grad_to_input(self) -> None:
        enc = _make_enc()
        x = _make_x(requires_grad=True)
        out, _ = enc(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_grad_to_all_parameters(self) -> None:
        """Every parameter in the encoder block must receive a gradient."""
        enc = _make_enc()
        x = _make_x()
        out, _ = enc(x)
        out.sum().backward()
        for name, param in enc.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' received no gradient."
            )
            assert not torch.all(param.grad == 0), (
                f"Parameter '{name}' has all-zero gradients."
            )

    def test_grad_flows_with_mask(self) -> None:
        """Gradients must reach the input even when a mask is applied."""
        enc = _make_enc()
        x = _make_x(batch=1, seq_len=6, requires_grad=True)
        mask = torch.zeros(1, 6, 6, dtype=torch.bool)
        mask[:, :, -1] = True
        out, _ = enc(x, mask=mask)
        out.sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)


# ===========================================================================
# Constructor validation tests
# ===========================================================================


class TestConstructorValidation:
    """Invalid constructor arguments must raise descriptive ValueError."""

    def test_d_model_not_divisible_by_num_heads(self) -> None:
        """d_model=65 is not divisible by num_heads=4 → ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            EncoderBlock(d_model=65, num_heads=4, d_ff=256)

    def test_d_model_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            EncoderBlock(d_model=0, num_heads=4, d_ff=256)

    def test_d_ff_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="d_ff"):
            EncoderBlock(d_model=64, num_heads=4, d_ff=0)

    def test_num_heads_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_heads"):
            EncoderBlock(d_model=64, num_heads=0, d_ff=256)

    def test_dropout_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            EncoderBlock(d_model=64, num_heads=4, d_ff=256, dropout=1.0)
