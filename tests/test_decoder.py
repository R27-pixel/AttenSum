"""
tests/test_decoder.py
======================
Unit tests for ``model.transformer.decoder.DecoderBlock``.

All tests run on CPU (no CUDA required).

Test coverage
-------------
Output shapes
    - Basic output shape (B, tgt_len, d_model)
    - Self-attention weight shape (B, num_heads, tgt_len, tgt_len)  [square]
    - Cross-attention weight shape (B, num_heads, tgt_len, src_len) [rectangular]
    - Different batch sizes
    - Different source lengths (src_len != tgt_len)
    - Different target lengths
    - Different d_model / num_heads / d_ff values

Component composition
    - self_attn is MultiHeadAttention
    - cross_attn is MultiHeadAttention (separate instance from self_attn)
    - ffn is PositionwiseFeedForward
    - residual_1, residual_2, residual_3 are independent ResidualConnections

Attention routing verification
    - Self-attention receives Q=K=V=x (verified via hook)
    - Cross-attention receives Q=x1, K=memory, V=memory (verified via hook)

Cross-attention rectangle test
    - With tgt_len=7 and src_len=15, cross_attn_weights is (B, H, 7, 15)

Causal target mask
    - tgt_mask with upper-triangle True zeroes future self-attention weights
    - Each row's unmasked weights sum to 1

Memory mask
    - memory_mask zeroes specified source positions in cross-attention weights

Post-norm ordering
    - Full manual reproduction of all 3 residual steps matches decoder output

Output properties
    - Output is finite
    - Output dtype is float32
    - Output is on CPU
    - Output mean near zero per position

Dropout behavior
    - dropout=0.0 deterministic in eval and train mode

Gradient flow
    - Gradient reaches x (target input)
    - Gradient reaches memory (encoder output)
    - All named parameters receive gradients

Constructor validation
    - d_model % num_heads != 0 raises ValueError
    - d_model = 0 raises ValueError
    - d_ff = 0 raises ValueError
    - num_heads = 0 raises ValueError
    - dropout >= 1.0 raises ValueError
"""

import pytest
import torch
import torch.nn as nn

from model.transformer.attention import MultiHeadAttention
from model.transformer.decoder import DecoderBlock
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
TGT_LEN = 10
SRC_LEN = 15     # intentionally != TGT_LEN to catch shape bugs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dec(
    d_model: int = D_MODEL,
    num_heads: int = NUM_HEADS,
    d_ff: int = D_FF,
    dropout: float = 0.0,
) -> DecoderBlock:
    return DecoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
    )


def _make_x(
    batch: int = BATCH,
    tgt_len: int = TGT_LEN,
    d_model: int = D_MODEL,
    requires_grad: bool = False,
) -> torch.Tensor:
    return torch.randn(
        batch, tgt_len, d_model,
        device=DEVICE,
        requires_grad=requires_grad,
    )


def _make_mem(
    batch: int = BATCH,
    src_len: int = SRC_LEN,
    d_model: int = D_MODEL,
    requires_grad: bool = False,
) -> torch.Tensor:
    return torch.randn(
        batch, src_len, d_model,
        device=DEVICE,
        requires_grad=requires_grad,
    )


# ===========================================================================
# Output shape tests
# ===========================================================================


class TestDecoderOutputShape:
    """The decoder block must return correct shapes for all three outputs."""

    def test_basic_output_shape(self) -> None:
        """Primary output must be (B, tgt_len, d_model)."""
        dec = _make_dec()
        out, _, _ = dec(_make_x(), _make_mem())
        assert out.shape == (BATCH, TGT_LEN, D_MODEL)

    def test_self_attn_weight_shape(self) -> None:
        """Self-attention weights must be (B, H, tgt_len, tgt_len) — square."""
        dec = _make_dec()
        _, sw, _ = dec(_make_x(), _make_mem())
        assert sw.shape == (BATCH, NUM_HEADS, TGT_LEN, TGT_LEN)

    def test_cross_attn_weight_shape(self) -> None:
        """Cross-attention weights must be (B, H, tgt_len, src_len) — rectangular."""
        dec = _make_dec()
        _, _, cw = dec(_make_x(), _make_mem())
        assert cw.shape == (BATCH, NUM_HEADS, TGT_LEN, SRC_LEN)

    def test_cross_attn_is_rectangular(self) -> None:
        """With tgt_len != src_len the cross-attention matrix must NOT be square."""
        dec = _make_dec()
        _, _, cw = dec(_make_x(tgt_len=7), _make_mem(src_len=20))
        assert cw.shape == (BATCH, NUM_HEADS, 7, 20)
        assert cw.shape[-2] != cw.shape[-1], (
            "Cross-attention weight matrix must be rectangular when tgt_len != src_len."
        )

    def test_batch_size_1(self) -> None:
        dec = _make_dec()
        out, sw, cw = dec(_make_x(batch=1), _make_mem(batch=1))
        assert out.shape == (1, TGT_LEN, D_MODEL)
        assert sw.shape == (1, NUM_HEADS, TGT_LEN, TGT_LEN)
        assert cw.shape == (1, NUM_HEADS, TGT_LEN, SRC_LEN)

    def test_batch_size_8(self) -> None:
        dec = _make_dec()
        out, sw, cw = dec(_make_x(batch=8), _make_mem(batch=8))
        assert out.shape == (8, TGT_LEN, D_MODEL)
        assert sw.shape == (8, NUM_HEADS, TGT_LEN, TGT_LEN)
        assert cw.shape == (8, NUM_HEADS, TGT_LEN, SRC_LEN)

    def test_different_tgt_len(self) -> None:
        dec = _make_dec()
        out, sw, cw = dec(_make_x(tgt_len=5), _make_mem())
        assert out.shape == (BATCH, 5, D_MODEL)
        assert sw.shape == (BATCH, NUM_HEADS, 5, 5)
        assert cw.shape == (BATCH, NUM_HEADS, 5, SRC_LEN)

    def test_different_src_len(self) -> None:
        dec = _make_dec()
        out, sw, cw = dec(_make_x(), _make_mem(src_len=30))
        assert out.shape == (BATCH, TGT_LEN, D_MODEL)
        assert cw.shape == (BATCH, NUM_HEADS, TGT_LEN, 30)

    def test_tgt_len_equals_src_len(self) -> None:
        """tgt == src (common in training) must also work."""
        dec = _make_dec()
        out, sw, cw = dec(_make_x(tgt_len=12), _make_mem(src_len=12))
        assert out.shape == (BATCH, 12, D_MODEL)
        assert sw.shape == (BATCH, NUM_HEADS, 12, 12)
        assert cw.shape == (BATCH, NUM_HEADS, 12, 12)

    def test_different_d_model(self) -> None:
        dec = _make_dec(d_model=32, num_heads=4, d_ff=128)
        out, sw, cw = dec(_make_x(d_model=32), _make_mem(d_model=32))
        assert out.shape == (BATCH, TGT_LEN, 32)
        assert sw.shape == (BATCH, 4, TGT_LEN, TGT_LEN)
        assert cw.shape == (BATCH, 4, TGT_LEN, SRC_LEN)

    def test_num_heads_1(self) -> None:
        dec = _make_dec(d_model=32, num_heads=1, d_ff=64)
        out, sw, cw = dec(_make_x(d_model=32), _make_mem(d_model=32))
        assert sw.shape == (BATCH, 1, TGT_LEN, TGT_LEN)
        assert cw.shape == (BATCH, 1, TGT_LEN, SRC_LEN)

    def test_num_heads_8(self) -> None:
        dec = _make_dec(d_model=64, num_heads=8, d_ff=256)
        out, sw, cw = dec(_make_x(d_model=64), _make_mem(d_model=64))
        assert sw.shape == (BATCH, 8, TGT_LEN, TGT_LEN)
        assert cw.shape == (BATCH, 8, TGT_LEN, SRC_LEN)

    def test_different_d_ff(self) -> None:
        dec = _make_dec(d_model=64, num_heads=4, d_ff=512)
        out, _, _ = dec(_make_x(), _make_mem())
        assert out.shape == (BATCH, TGT_LEN, D_MODEL)

    def test_tgt_len_1(self) -> None:
        """Single-step decoding (tgt_len=1) must work."""
        dec = _make_dec()
        out, sw, cw = dec(_make_x(tgt_len=1), _make_mem())
        assert out.shape == (BATCH, 1, D_MODEL)
        assert sw.shape == (BATCH, NUM_HEADS, 1, 1)
        assert cw.shape == (BATCH, NUM_HEADS, 1, SRC_LEN)


# ===========================================================================
# Component composition tests
# ===========================================================================


class TestComponentComposition:
    """Verify that the decoder wires the correct existing module types."""

    def test_self_attn_is_multihead_attention(self) -> None:
        dec = _make_dec()
        assert isinstance(dec.self_attn, MultiHeadAttention)

    def test_cross_attn_is_multihead_attention(self) -> None:
        dec = _make_dec()
        assert isinstance(dec.cross_attn, MultiHeadAttention)

    def test_self_attn_and_cross_attn_are_different_instances(self) -> None:
        """The two attention modules must have independent parameters."""
        dec = _make_dec()
        assert dec.self_attn is not dec.cross_attn
        assert dec.self_attn.w_q.weight.data_ptr() != dec.cross_attn.w_q.weight.data_ptr()

    def test_ffn_is_positionwise_feed_forward(self) -> None:
        dec = _make_dec()
        assert isinstance(dec.ffn, PositionwiseFeedForward)

    def test_residual_1_is_residual_connection(self) -> None:
        dec = _make_dec()
        assert isinstance(dec.residual_1, ResidualConnection)

    def test_residual_2_is_residual_connection(self) -> None:
        dec = _make_dec()
        assert isinstance(dec.residual_2, ResidualConnection)

    def test_residual_3_is_residual_connection(self) -> None:
        dec = _make_dec()
        assert isinstance(dec.residual_3, ResidualConnection)

    def test_three_residuals_are_independent(self) -> None:
        """All three ResidualConnections must be distinct with independent LayerNorms."""
        dec = _make_dec()
        assert dec.residual_1 is not dec.residual_2
        assert dec.residual_2 is not dec.residual_3
        assert dec.residual_1 is not dec.residual_3

        ptrs = {
            dec.residual_1.layer_norm.norm.weight.data_ptr(),
            dec.residual_2.layer_norm.norm.weight.data_ptr(),
            dec.residual_3.layer_norm.norm.weight.data_ptr(),
        }
        assert len(ptrs) == 3, "All three ResidualConnections must have independent weights."


# ===========================================================================
# Attention routing verification tests
# ===========================================================================


class TestAttentionRouting:
    """Verify Q/K/V routing using forward hooks."""

    def test_self_attn_receives_x_as_q_k_v(self) -> None:
        """Self-attention must receive x as Q, K, and V (same tensor content).

        Strategy: register a hook on self_attn's forward to capture the
        three positional args and compare them to x.
        """
        dec = _make_dec()
        dec.eval()

        x = _make_x(batch=1, tgt_len=5)
        mem = _make_mem(batch=1, src_len=8)

        captured_qkv: list[tuple[torch.Tensor, ...]] = []

        def _hook(
            module: nn.Module,
            args: tuple,
            kwargs: dict,
        ) -> None:
            # MHA forward receives (query, key, value, mask=...)
            captured_qkv.append(args[:3])

        handle = dec.self_attn.register_forward_pre_hook(
            _hook, with_kwargs=True
        )
        try:
            with torch.no_grad():
                dec(x, mem)
        finally:
            handle.remove()

        assert len(captured_qkv) == 1
        q_in, k_in, v_in = captured_qkv[0]

        # All three must equal x
        assert torch.allclose(q_in, x, atol=0.0), "Self-attn Q must be x."
        assert torch.allclose(k_in, x, atol=0.0), "Self-attn K must be x."
        assert torch.allclose(v_in, x, atol=0.0), "Self-attn V must be x."

    def test_cross_attn_receives_memory_as_k_v(self) -> None:
        """Cross-attention must receive memory as both K and V.

        Strategy: register a hook on cross_attn's forward and check that
        the second and third positional args equal memory.
        """
        dec = _make_dec()
        dec.eval()

        x = _make_x(batch=1, tgt_len=5)
        mem = _make_mem(batch=1, src_len=8)

        captured_kv: list[tuple[torch.Tensor, torch.Tensor]] = []

        def _hook(
            module: nn.Module,
            args: tuple,
            kwargs: dict,
        ) -> None:
            captured_kv.append((args[1], args[2]))   # key, value

        handle = dec.cross_attn.register_forward_pre_hook(
            _hook, with_kwargs=True
        )
        try:
            with torch.no_grad():
                dec(x, mem)
        finally:
            handle.remove()

        assert len(captured_kv) == 1
        k_in, v_in = captured_kv[0]

        assert torch.allclose(k_in, mem, atol=0.0), "Cross-attn K must be memory."
        assert torch.allclose(v_in, mem, atol=0.0), "Cross-attn V must be memory."

    def test_cross_attn_q_is_not_x(self) -> None:
        """Cross-attention Q must come from the decoder (x1 after residual_1),
        not from the raw input x.

        With non-zero self-attention, x1 = LayerNorm(x + SelfAttn(x)) ≠ x
        for typical random inputs.
        """
        dec = _make_dec()
        dec.eval()

        x = _make_x(batch=1, tgt_len=5)
        mem = _make_mem(batch=1, src_len=8)

        captured_q: list[torch.Tensor] = []

        def _hook(module: nn.Module, args: tuple, kwargs: dict) -> None:
            captured_q.append(args[0])  # query

        handle = dec.cross_attn.register_forward_pre_hook(
            _hook, with_kwargs=True
        )
        try:
            with torch.no_grad():
                dec(x, mem)
        finally:
            handle.remove()

        q_in = captured_q[0]
        # Q must differ from x because residual_1 transforms x
        assert not torch.allclose(q_in, x, atol=1e-5), (
            "Cross-attention Q must be x1 (post-self-attn), not the raw x."
        )


# ===========================================================================
# Causal target mask tests
# ===========================================================================


class TestCausalTargetMask:
    """Causal masking must prevent each target position from attending to future tokens."""

    def test_causal_mask_zeroes_future_self_attn_weights(self) -> None:
        """Upper-triangular mask must produce ~0 weights above the diagonal."""
        tgt = 8
        dec = _make_dec()
        dec.eval()

        x = _make_x(batch=1, tgt_len=tgt)
        mem = _make_mem(batch=1)

        causal_mask = torch.triu(
            torch.ones(1, tgt, tgt, dtype=torch.bool), diagonal=1
        )

        _, sw, _ = dec(x, mem, tgt_mask=causal_mask)
        # sw: (1, H, tgt, tgt)
        for h in range(NUM_HEADS):
            upper = sw[0, h].triu(diagonal=1)
            assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6), (
                f"Head {h}: causal mask must zero out above-diagonal self-attn weights."
            )

    def test_causal_mask_row_sums_to_one(self) -> None:
        """After applying causal mask, each unmasked row must sum to 1."""
        tgt = 6
        dec = _make_dec()
        dec.eval()

        x = _make_x(batch=2, tgt_len=tgt)
        mem = _make_mem(batch=2)

        causal_mask = torch.triu(
            torch.ones(1, tgt, tgt, dtype=torch.bool), diagonal=1
        )

        _, sw, _ = dec(x, mem, tgt_mask=causal_mask)
        row_sums = sw.sum(dim=-1)   # (B, H, tgt)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            "Each self-attention row must sum to 1 after causal masking."
        )

    def test_no_tgt_mask_all_self_attn_weights_positive(self) -> None:
        """Without a mask every self-attention weight must be positive."""
        dec = _make_dec()
        dec.eval()
        _, sw, _ = dec(_make_x(), _make_mem())
        assert (sw > 0).all()


# ===========================================================================
# Memory mask tests
# ===========================================================================


class TestMemoryMask:
    """memory_mask must be forwarded to the cross-attention sub-layer."""

    def test_memory_mask_zeroes_masked_src_positions(self) -> None:
        """Bool memory_mask must produce ~0 cross-attn weights at masked source positions."""
        dec = _make_dec()
        dec.eval()

        x = _make_x(batch=1, tgt_len=6)
        mem = _make_mem(batch=1, src_len=10)

        # Mask the last 3 source positions
        mask = torch.zeros(1, 6, 10, dtype=torch.bool)
        mask[:, :, 7:] = True

        _, _, cw = dec(x, mem, memory_mask=mask)
        # cw: (1, H, 6, 10)
        assert torch.allclose(
            cw[:, :, :, 7:], torch.zeros(1, NUM_HEADS, 6, 3), atol=1e-6
        ), "Masked source positions must have ~0 cross-attention weight."

    def test_memory_mask_unmasked_sums_to_one(self) -> None:
        """Unmasked source positions must form a valid probability distribution."""
        dec = _make_dec()
        dec.eval()

        x = _make_x(batch=2, tgt_len=5)
        mem = _make_mem(batch=2, src_len=8)

        mask = torch.zeros(2, 5, 8, dtype=torch.bool)
        mask[:, :, 6:] = True   # mask last 2 of 8

        _, _, cw = dec(x, mem, memory_mask=mask)
        unmasked_sum = cw[:, :, :, :6].sum(dim=-1)   # (B, H, tgt)
        assert torch.allclose(unmasked_sum, torch.ones_like(unmasked_sum), atol=1e-5)


# ===========================================================================
# Post-norm ordering tests
# ===========================================================================


class TestPostNormOrdering:
    """Verify the exact post-norm equations for all three sub-layers."""

    def test_full_decoder_matches_manual_computation(self) -> None:
        """Manually reproduce all six stages and compare with DecoderBlock output."""
        dec = _make_dec()
        dec.eval()

        x = _make_x(batch=1, tgt_len=6)
        mem = _make_mem(batch=1, src_len=9)

        output, _, _ = dec(x, mem)

        with torch.no_grad():
            # Stage 1: masked self-attention (no mask here)
            sa_out, _ = dec.self_attn(x, x, x, mask=None)
            # Stage 2: first residual
            x1 = dec.residual_1(x, sa_out)
            # Stage 3: cross-attention
            ca_out, _ = dec.cross_attn(x1, mem, mem, mask=None)
            # Stage 4: second residual
            x2 = dec.residual_2(x1, ca_out)
            # Stage 5: FFN
            ff_out = dec.ffn(x2)
            # Stage 6: third residual
            expected = dec.residual_3(x2, ff_out)

        assert torch.allclose(output, expected, atol=1e-5), (
            "DecoderBlock output must match manual stage-by-stage computation."
        )

    def test_stage1_is_postnorm(self) -> None:
        """residual_1 must compute LayerNorm(x + self_attn_out)."""
        dec = _make_dec()
        dec.eval()

        torch.manual_seed(7)
        x = torch.randn(1, 5, D_MODEL)
        mem = torch.randn(1, 8, D_MODEL)

        with torch.no_grad():
            sa_out, _ = dec.self_attn(x, x, x)
            x1_actual = dec.residual_1(x, sa_out)
            x1_expected = dec.residual_1.layer_norm(x + sa_out)

        assert torch.allclose(x1_actual, x1_expected, atol=1e-6)

    def test_stage3_is_postnorm(self) -> None:
        """residual_2 must compute LayerNorm(x1 + cross_attn_out)."""
        dec = _make_dec()
        dec.eval()

        torch.manual_seed(42)
        x = torch.randn(1, 5, D_MODEL)
        mem = torch.randn(1, 8, D_MODEL)

        with torch.no_grad():
            sa_out, _ = dec.self_attn(x, x, x)
            x1 = dec.residual_1(x, sa_out)
            ca_out, _ = dec.cross_attn(x1, mem, mem)
            x2_actual = dec.residual_2(x1, ca_out)
            x2_expected = dec.residual_2.layer_norm(x1 + ca_out)

        assert torch.allclose(x2_actual, x2_expected, atol=1e-6)

    def test_stage5_is_postnorm(self) -> None:
        """residual_3 must compute LayerNorm(x2 + ffn_out)."""
        dec = _make_dec()
        dec.eval()

        torch.manual_seed(13)
        x = torch.randn(1, 5, D_MODEL)
        mem = torch.randn(1, 8, D_MODEL)

        with torch.no_grad():
            sa_out, _ = dec.self_attn(x, x, x)
            x1 = dec.residual_1(x, sa_out)
            ca_out, _ = dec.cross_attn(x1, mem, mem)
            x2 = dec.residual_2(x1, ca_out)
            ff_out = dec.ffn(x2)
            out_actual = dec.residual_3(x2, ff_out)
            out_expected = dec.residual_3.layer_norm(x2 + ff_out)

        assert torch.allclose(out_actual, out_expected, atol=1e-6)


# ===========================================================================
# Output property tests
# ===========================================================================


class TestOutputProperties:

    def test_output_is_finite(self) -> None:
        dec = _make_dec()
        out, sw, cw = dec(_make_x(), _make_mem())
        assert torch.isfinite(out).all()
        assert torch.isfinite(sw).all()
        assert torch.isfinite(cw).all()

    def test_output_dtype_float32(self) -> None:
        dec = _make_dec()
        out, sw, cw = dec(_make_x(), _make_mem())
        assert out.dtype == torch.float32
        assert sw.dtype == torch.float32
        assert cw.dtype == torch.float32

    def test_output_on_cpu(self) -> None:
        dec = _make_dec()
        out, sw, cw = dec(_make_x(), _make_mem())
        assert out.device.type == "cpu"
        assert sw.device.type == "cpu"
        assert cw.device.type == "cpu"

    def test_output_mean_near_zero(self) -> None:
        """Last LayerNorm makes each position's output mean ≈ 0."""
        dec = _make_dec()
        out, _, _ = dec(_make_x() * 5.0, _make_mem() * 5.0)
        per_pos_mean = out.mean(dim=-1)
        assert torch.allclose(
            per_pos_mean, torch.zeros_like(per_pos_mean), atol=1e-5
        )

    def test_self_attn_weights_sum_to_one(self) -> None:
        """Each row of self-attention weights must sum to 1."""
        dec = _make_dec()
        _, sw, _ = dec(_make_x(), _make_mem())
        row_sums = sw.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_cross_attn_weights_sum_to_one(self) -> None:
        """Each row of cross-attention weights must sum to 1."""
        dec = _make_dec()
        _, _, cw = dec(_make_x(), _make_mem())
        row_sums = cw.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


# ===========================================================================
# Dropout tests
# ===========================================================================


class TestDropoutBehavior:

    def test_no_dropout_deterministic_eval(self) -> None:
        dec = _make_dec(dropout=0.0)
        dec.eval()
        x, mem = _make_x(), _make_mem()
        out1, sw1, cw1 = dec(x, mem)
        out2, sw2, cw2 = dec(x, mem)
        assert torch.allclose(out1, out2)
        assert torch.allclose(sw1, sw2)
        assert torch.allclose(cw1, cw2)

    def test_no_dropout_deterministic_train(self) -> None:
        dec = _make_dec(dropout=0.0)
        dec.train()
        x, mem = _make_x(), _make_mem()
        out1, _, _ = dec(x, mem)
        out2, _, _ = dec(x, mem)
        assert torch.allclose(out1, out2)

    def test_with_dropout_output_shape_unchanged(self) -> None:
        dec = _make_dec(dropout=0.1)
        out, _, _ = dec(_make_x(), _make_mem())
        assert out.shape == (BATCH, TGT_LEN, D_MODEL)


# ===========================================================================
# Gradient flow tests
# ===========================================================================


class TestGradientFlow:

    def test_grad_to_x(self) -> None:
        dec = _make_dec()
        x = _make_x(requires_grad=True)
        out, _, _ = dec(x, _make_mem())
        out.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_grad_to_memory(self) -> None:
        dec = _make_dec()
        mem = _make_mem(requires_grad=True)
        out, _, _ = dec(_make_x(), mem)
        out.sum().backward()
        assert mem.grad is not None
        assert not torch.all(mem.grad == 0)

    def test_grad_to_all_parameters(self) -> None:
        """Every parameter in the decoder must receive a gradient."""
        dec = _make_dec()
        out, _, _ = dec(_make_x(), _make_mem())
        out.sum().backward()
        for name, param in dec.named_parameters():
            assert param.grad is not None, f"No gradient for '{name}'."
            assert not torch.all(param.grad == 0), f"All-zero grad for '{name}'."

    def test_grad_flows_with_causal_mask(self) -> None:
        dec = _make_dec()
        x = _make_x(batch=1, tgt_len=6, requires_grad=True)
        mem = _make_mem(batch=1, requires_grad=True)
        causal_mask = torch.triu(
            torch.ones(1, 6, 6, dtype=torch.bool), diagonal=1
        )
        out, _, _ = dec(x, mem, tgt_mask=causal_mask)
        out.sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)
        assert mem.grad is not None and not torch.all(mem.grad == 0)


# ===========================================================================
# Constructor validation tests
# ===========================================================================


class TestConstructorValidation:

    def test_d_model_not_divisible_by_num_heads(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            DecoderBlock(d_model=65, num_heads=4, d_ff=256)

    def test_d_model_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            DecoderBlock(d_model=0, num_heads=4, d_ff=256)

    def test_d_ff_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="d_ff"):
            DecoderBlock(d_model=64, num_heads=4, d_ff=0)

    def test_num_heads_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_heads"):
            DecoderBlock(d_model=64, num_heads=0, d_ff=256)

    def test_dropout_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            DecoderBlock(d_model=64, num_heads=4, d_ff=256, dropout=1.0)
