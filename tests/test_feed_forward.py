"""
tests/test_feed_forward.py
===========================
Unit tests for ``model.transformer.feed_forward.PositionwiseFeedForward``.

All tests run on CPU (no CUDA required).

Test coverage
-------------
Output shape
    - Basic shape (B, seq_len, d_model) is preserved
    - Different batch sizes (1, 4, 16)
    - Different sequence lengths (1, 10, 128)
    - Different d_model / d_ff configurations
    - Single-position sequence (seq_len = 1)

Output properties
    - Output dtype is float32
    - All output values are finite (no NaN / Inf)
    - Output is on CPU

Position-wise independence
    - The network is applied identically at each position (verify by
      running one position in isolation vs. inside a full sequence)

Activation behaviour
    - Default activation (ReLU) produces non-negative inner values
    - Custom activation (GELU) is accepted without error
    - Custom activation (SiLU) is accepted without error

Dropout behaviour
    - dropout=0.0 gives identical outputs across two forward passes
      (in both eval and train mode)

Gradient flow
    - Gradients reach linear_1.weight and linear_2.weight after backward()
    - Gradient reaches the input tensor (enables use inside deeper networks)

Constructor validation
    - d_model <= 0 raises ValueError
    - d_ff <= 0 raises ValueError
    - dropout outside [0.0, 1.0) raises ValueError
"""

import torch
import torch.nn as nn
import pytest

from model.transformer.feed_forward import PositionwiseFeedForward

# ---------------------------------------------------------------------------
# Shared constants — small values, not AttenSum's final hyperparameters.
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")

D_MODEL = 64
D_FF = 256           # typical 4× expansion
BATCH = 3
SEQ_LEN = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ff(
    d_model: int = D_MODEL,
    d_ff: int = D_FF,
    dropout: float = 0.0,
    activation: nn.Module | None = None,
) -> PositionwiseFeedForward:
    """Construct a PositionwiseFeedForward with given params."""
    return PositionwiseFeedForward(
        d_model=d_model,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
    )


def _make_input(
    batch: int = BATCH,
    seq_len: int = SEQ_LEN,
    d_model: int = D_MODEL,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Return a random float32 tensor of shape (batch, seq_len, d_model)."""
    return torch.randn(batch, seq_len, d_model, requires_grad=requires_grad)


# ===========================================================================
# Output shape tests
# ===========================================================================


class TestOutputShape:
    """Forward output must have the same shape as the input."""

    def test_basic_output_shape(self) -> None:
        """Output shape must be (B, seq_len, d_model)."""
        ff = _make_ff()
        x = _make_input()
        out = ff(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL), (
            f"Expected {(BATCH, SEQ_LEN, D_MODEL)}, got {out.shape}"
        )

    def test_shape_batch_size_1(self) -> None:
        ff = _make_ff()
        x = _make_input(batch=1)
        assert ff(x).shape == (1, SEQ_LEN, D_MODEL)

    def test_shape_batch_size_4(self) -> None:
        ff = _make_ff()
        x = _make_input(batch=4)
        assert ff(x).shape == (4, SEQ_LEN, D_MODEL)

    def test_shape_batch_size_16(self) -> None:
        ff = _make_ff()
        x = _make_input(batch=16)
        assert ff(x).shape == (16, SEQ_LEN, D_MODEL)

    def test_shape_seq_len_1(self) -> None:
        """A single-token sequence must be handled correctly."""
        ff = _make_ff()
        x = _make_input(seq_len=1)
        assert ff(x).shape == (BATCH, 1, D_MODEL)

    def test_shape_seq_len_10(self) -> None:
        ff = _make_ff()
        x = _make_input(seq_len=10)
        assert ff(x).shape == (BATCH, 10, D_MODEL)

    def test_shape_seq_len_128(self) -> None:
        ff = _make_ff()
        x = _make_input(seq_len=128)
        assert ff(x).shape == (BATCH, 128, D_MODEL)

    def test_shape_small_d_model(self) -> None:
        """Smallest practical d_model (4) with d_ff=16 must work."""
        ff = _make_ff(d_model=4, d_ff=16)
        x = _make_input(d_model=4)
        assert ff(x).shape == (BATCH, SEQ_LEN, 4)

    def test_shape_large_d_model(self) -> None:
        """Larger d_model (512) with d_ff=2048 (paper's scale) must work."""
        ff = _make_ff(d_model=512, d_ff=2048)
        x = _make_input(d_model=512)
        assert ff(x).shape == (BATCH, SEQ_LEN, 512)

    def test_shape_d_ff_less_than_d_model(self) -> None:
        """d_ff < d_model (bottleneck) is a valid configuration."""
        ff = _make_ff(d_model=64, d_ff=16)
        x = _make_input(d_model=64)
        assert ff(x).shape == (BATCH, SEQ_LEN, 64)

    def test_shape_d_ff_equals_d_model(self) -> None:
        """d_ff == d_model (no expansion) is a valid configuration."""
        ff = _make_ff(d_model=64, d_ff=64)
        x = _make_input(d_model=64)
        assert ff(x).shape == (BATCH, SEQ_LEN, 64)


# ===========================================================================
# Output value property tests
# ===========================================================================


class TestOutputProperties:
    """Forward output must satisfy basic numerical and type properties."""

    def test_output_dtype_float32(self) -> None:
        """Output must be float32."""
        ff = _make_ff()
        out = ff(_make_input())
        assert out.dtype == torch.float32

    def test_output_on_cpu(self) -> None:
        """All computation must remain on CPU."""
        ff = _make_ff()
        out = ff(_make_input())
        assert out.device.type == "cpu"

    def test_output_finite(self) -> None:
        """Output must contain no NaN or Inf values for typical inputs."""
        ff = _make_ff()
        out = ff(_make_input())
        assert torch.isfinite(out).all(), "Output must be finite."

    def test_output_finite_large_inputs(self) -> None:
        """Large-magnitude inputs must not produce NaN / Inf."""
        ff = _make_ff()
        x = _make_input() * 1e3
        assert torch.isfinite(ff(x)).all()

    def test_output_not_all_zeros_for_nonzero_input(self) -> None:
        """A non-zero input must generally produce a non-zero output."""
        ff = _make_ff()
        x = _make_input()
        out = ff(x)
        assert not torch.all(out == 0.0), (
            "Output must not be all-zero for a typical non-zero input."
        )


# ===========================================================================
# Position-wise independence tests
# ===========================================================================


class TestPositionwiseBehaviour:
    """The FFN must apply the same transformation independently per position."""

    def test_single_position_matches_full_sequence(self) -> None:
        """Running the FFN on one position in isolation must match the output
        when that position is embedded inside a longer sequence.

        Because the FFN is position-wise (nn.Linear operates on the last dim),
        the output for position ``p`` should be identical whether or not other
        positions are present in the input tensor.
        """
        ff = _make_ff()
        ff.eval()

        # Full sequence: (1, 5, d_model)
        x_full = _make_input(batch=1, seq_len=5)
        out_full = ff(x_full)   # (1, 5, d_model)

        # Extract position 2 and run it alone: (1, 1, d_model)
        x_single = x_full[:, 2:3, :]    # (1, 1, d_model)
        out_single = ff(x_single)        # (1, 1, d_model)

        assert torch.allclose(out_full[:, 2:3, :], out_single, atol=1e-6), (
            "FFN output at a position must be identical regardless of surrounding positions."
        )

    def test_different_positions_can_produce_different_outputs(self) -> None:
        """Different input values at different positions must produce different outputs.

        This confirms the FFN is *not* collapsing all positions to a constant.
        """
        ff = _make_ff()
        ff.eval()
        x = _make_input(batch=1, seq_len=4)
        out = ff(x)  # (1, 4, d_model)
        # Very unlikely that two random inputs give identical outputs
        assert not torch.allclose(out[0, 0], out[0, 1]), (
            "Different input positions should generally produce different outputs."
        )


# ===========================================================================
# Activation tests
# ===========================================================================


class TestActivation:
    """Tests related to the configurable activation function."""

    def test_default_activation_is_relu(self) -> None:
        """Default activation must be ReLU (per original Transformer paper)."""
        ff = _make_ff()
        assert isinstance(ff.activation, nn.ReLU), (
            "Default activation must be nn.ReLU."
        )

    def test_relu_inner_values_non_negative(self) -> None:
        """With ReLU, the inner representation (after activation) must be ≥ 0.

        Strategy: register a forward hook to capture the post-activation tensor.
        """
        ff = _make_ff(activation=nn.ReLU())
        captured: list[torch.Tensor] = []

        def _hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
            captured.append(out.detach())

        handle = ff.activation.register_forward_hook(_hook)
        try:
            ff(_make_input())
        finally:
            handle.remove()

        assert len(captured) == 1
        assert (captured[0] >= 0).all(), (
            "ReLU activation must produce non-negative values."
        )

    def test_custom_gelu_activation_accepted(self) -> None:
        """GELU activation must be accepted and must produce finite output."""
        ff = _make_ff(activation=nn.GELU())
        out = ff(_make_input())
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
        assert torch.isfinite(out).all()

    def test_custom_silu_activation_accepted(self) -> None:
        """SiLU (Swish) activation must be accepted and produce finite output."""
        ff = _make_ff(activation=nn.SiLU())
        out = ff(_make_input())
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
        assert torch.isfinite(out).all()

    def test_custom_tanh_activation_accepted(self) -> None:
        """Tanh activation must be accepted and produce finite output."""
        ff = _make_ff(activation=nn.Tanh())
        out = ff(_make_input())
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
        assert torch.isfinite(out).all()


# ===========================================================================
# Dropout tests
# ===========================================================================


class TestDropout:
    """Dropout must be deterministic when p=0 and must not alter output shape."""

    def test_no_dropout_deterministic_eval_mode(self) -> None:
        """With dropout=0.0 in eval mode, two passes must be identical."""
        ff = _make_ff(dropout=0.0)
        ff.eval()
        x = _make_input()
        out1 = ff(x)
        out2 = ff(x)
        assert torch.allclose(out1, out2), (
            "dropout=0.0 must give deterministic output in eval mode."
        )

    def test_no_dropout_deterministic_train_mode(self) -> None:
        """p=0 is a mathematical no-op — output must be identical in train mode too."""
        ff = _make_ff(dropout=0.0)
        ff.train()
        x = _make_input()
        out1 = ff(x)
        out2 = ff(x)
        assert torch.allclose(out1, out2), (
            "dropout=0.0 must give deterministic output even in train mode."
        )

    def test_dropout_preserves_output_shape(self) -> None:
        """Enabling dropout must not change the output shape."""
        ff = _make_ff(dropout=0.1)
        x = _make_input()
        assert ff(x).shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_eval_mode_gives_deterministic_output_with_nonzero_dropout(self) -> None:
        """In eval mode, nn.Dropout is a no-op regardless of p."""
        ff = _make_ff(dropout=0.3)
        ff.eval()
        x = _make_input()
        out1 = ff(x)
        out2 = ff(x)
        assert torch.allclose(out1, out2), (
            "In eval mode, dropout must not introduce randomness."
        )


# ===========================================================================
# Gradient flow tests
# ===========================================================================


class TestGradientFlow:
    """Gradients must propagate back through the entire network."""

    def test_grad_flows_to_input(self) -> None:
        """After backward(), the input tensor must have a non-None, non-zero grad."""
        ff = _make_ff()
        x = _make_input(requires_grad=True)
        out = ff(x)
        out.sum().backward()
        assert x.grad is not None, "Gradient did not reach the input tensor."
        assert not torch.all(x.grad == 0), "Input gradient must not be all zeros."

    def test_grad_flows_to_linear_1_weight(self) -> None:
        """Gradient must reach linear_1.weight after backward()."""
        ff = _make_ff()
        x = _make_input()
        ff(x).sum().backward()
        assert ff.linear_1.weight.grad is not None
        assert not torch.all(ff.linear_1.weight.grad == 0)

    def test_grad_flows_to_linear_2_weight(self) -> None:
        """Gradient must reach linear_2.weight after backward()."""
        ff = _make_ff()
        x = _make_input()
        ff(x).sum().backward()
        assert ff.linear_2.weight.grad is not None
        assert not torch.all(ff.linear_2.weight.grad == 0)

    def test_grad_flows_to_linear_1_bias(self) -> None:
        """Gradient must reach linear_1.bias after backward()."""
        ff = _make_ff()
        x = _make_input()
        ff(x).sum().backward()
        assert ff.linear_1.bias is not None
        assert ff.linear_1.bias.grad is not None
        assert not torch.all(ff.linear_1.bias.grad == 0)

    def test_grad_flows_to_linear_2_bias(self) -> None:
        """Gradient must reach linear_2.bias after backward()."""
        ff = _make_ff()
        x = _make_input()
        ff(x).sum().backward()
        assert ff.linear_2.bias is not None
        assert ff.linear_2.bias.grad is not None
        assert not torch.all(ff.linear_2.bias.grad == 0)

    def test_grad_flows_with_custom_activation(self) -> None:
        """Gradients must flow through a GELU activation as well."""
        ff = _make_ff(activation=nn.GELU())
        x = _make_input(requires_grad=True)
        ff(x).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)

    def test_no_bias_grad_when_bias_false(self) -> None:
        """When bias=False, bias attributes must be None (no gradient expected)."""
        ff = PositionwiseFeedForward(d_model=D_MODEL, d_ff=D_FF, bias=False)
        x = _make_input()
        ff(x).sum().backward()
        assert ff.linear_1.bias is None
        assert ff.linear_2.bias is None


# ===========================================================================
# Internal structure tests
# ===========================================================================


class TestInternalStructure:
    """Verify that weight matrices have the expected shapes."""

    def test_linear_1_weight_shape(self) -> None:
        """linear_1.weight must be (d_ff, d_model) — PyTorch Linear convention."""
        ff = _make_ff(d_model=32, d_ff=128)
        # nn.Linear stores weights transposed: shape is (out_features, in_features)
        assert ff.linear_1.weight.shape == (128, 32)

    def test_linear_2_weight_shape(self) -> None:
        """linear_2.weight must be (d_model, d_ff)."""
        ff = _make_ff(d_model=32, d_ff=128)
        assert ff.linear_2.weight.shape == (32, 128)

    def test_stored_d_model(self) -> None:
        """d_model must be stored as an attribute."""
        ff = _make_ff(d_model=48, d_ff=192)
        assert ff.d_model == 48

    def test_stored_d_ff(self) -> None:
        """d_ff must be stored as an attribute."""
        ff = _make_ff(d_model=48, d_ff=192)
        assert ff.d_ff == 192


# ===========================================================================
# Constructor validation tests
# ===========================================================================


class TestConstructorValidation:
    """Invalid constructor arguments must raise descriptive ValueError."""

    def test_d_model_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="d_model"):
            PositionwiseFeedForward(d_model=0, d_ff=256)

    def test_d_model_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="d_model"):
            PositionwiseFeedForward(d_model=-1, d_ff=256)

    def test_d_ff_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="d_ff"):
            PositionwiseFeedForward(d_model=64, d_ff=0)

    def test_d_ff_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="d_ff"):
            PositionwiseFeedForward(d_model=64, d_ff=-8)

    def test_dropout_one_raises(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            PositionwiseFeedForward(d_model=64, d_ff=256, dropout=1.0)

    def test_dropout_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            PositionwiseFeedForward(d_model=64, d_ff=256, dropout=-0.5)
