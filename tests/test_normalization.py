"""
tests/test_normalization.py
============================
Unit tests for:
  - ``model.transformer.normalization.LayerNorm``
  - ``model.transformer.normalization.ResidualConnection``

All tests run on CPU (no CUDA required).

Test coverage
-------------
LayerNorm
    - Output shape for standard and edge-case inputs
    - Different batch sizes
    - Different sequence lengths
    - Different d_model values
    - Normalized output properties (mean ≈ 0, std ≈ 1 with default init)
    - Affine parameters exist with correct shapes
    - Gradient flow to input, weight, and bias
    - dtype / device preservation
    - Constructor validation (d_model ≤ 0)

ResidualConnection
    - Output shape
    - Post-norm formula verification: LayerNorm(x + sub) vs x + LayerNorm(sub)
    - Identity-like behavior when sublayer_output is zero
    - Correct residual addition (result changes when x or sublayer_output changes)
    - Gradient flow to both x and sublayer_output
    - Different tensor sizes
    - Constructor validation (propagated from LayerNorm)
"""

import torch
import torch.nn as nn
import pytest

from model.transformer.normalization import LayerNorm, ResidualConnection

# ---------------------------------------------------------------------------
# Shared constants — small, not AttenSum's final hyperparameters.
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")
D_MODEL = 64
BATCH = 3
SEQ_LEN = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand(
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
# LayerNorm tests
# ===========================================================================


class TestLayerNormShape:
    """Output shape must always equal input shape."""

    def test_basic_output_shape(self) -> None:
        """Standard (B, seq_len, d_model) input must give same-shape output."""
        ln = LayerNorm(d_model=D_MODEL)
        out = ln(_rand())
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_batch_size_1(self) -> None:
        ln = LayerNorm(d_model=D_MODEL)
        assert ln(_rand(batch=1)).shape == (1, SEQ_LEN, D_MODEL)

    def test_batch_size_8(self) -> None:
        ln = LayerNorm(d_model=D_MODEL)
        assert ln(_rand(batch=8)).shape == (8, SEQ_LEN, D_MODEL)

    def test_batch_size_16(self) -> None:
        ln = LayerNorm(d_model=D_MODEL)
        assert ln(_rand(batch=16)).shape == (16, SEQ_LEN, D_MODEL)

    def test_seq_len_1(self) -> None:
        """Single-token sequence must work."""
        ln = LayerNorm(d_model=D_MODEL)
        assert ln(_rand(seq_len=1)).shape == (BATCH, 1, D_MODEL)

    def test_seq_len_128(self) -> None:
        ln = LayerNorm(d_model=D_MODEL)
        assert ln(_rand(seq_len=128)).shape == (BATCH, 128, D_MODEL)

    def test_small_d_model(self) -> None:
        ln = LayerNorm(d_model=4)
        assert ln(_rand(d_model=4)).shape == (BATCH, SEQ_LEN, 4)

    def test_large_d_model(self) -> None:
        ln = LayerNorm(d_model=512)
        assert ln(_rand(d_model=512)).shape == (BATCH, SEQ_LEN, 512)


class TestLayerNormNormalisedProperties:
    """With default weight=1, bias=0 init, the output must be near-normalised."""

    def test_mean_near_zero(self) -> None:
        """Each position's d_model-dimensional output must have mean ≈ 0.

        ``nn.LayerNorm`` initialises γ=1 and β=0, so a fresh instance
        produces outputs with mean ≈ 0 (exactly 0 in theory; tiny float
        errors explain the tolerance).
        """
        ln = LayerNorm(d_model=D_MODEL)
        # Scale the input to make the test non-trivial
        x = _rand() * 5.0 + 3.0   # shifted and scaled, mean ≠ 0
        out = ln(x)                 # (B, seq_len, d_model)

        per_position_mean = out.mean(dim=-1)   # (B, seq_len)
        assert torch.allclose(
            per_position_mean,
            torch.zeros_like(per_position_mean),
            atol=1e-5,
        ), (
            f"LayerNorm output mean must be ≈ 0; "
            f"max |mean| = {per_position_mean.abs().max():.2e}"
        )

    def test_std_near_one(self) -> None:
        """Each position's d_model-dimensional output must have std ≈ 1.

        With default γ=1 and β=0, LayerNorm(x) ≈ (x − μ) / √(σ²+ε),
        whose standard deviation is √(σ²) / √(σ²+ε) ≈ 1 for small ε.
        """
        ln = LayerNorm(d_model=D_MODEL)
        x = _rand() * 10.0          # arbitrary scale — LN normalises it
        out = ln(x)                  # (B, seq_len, d_model)

        # Biased std (unbiased=False) to match what LayerNorm uses internally.
        per_position_std = out.std(dim=-1, unbiased=False)   # (B, seq_len)
        assert torch.allclose(
            per_position_std,
            torch.ones_like(per_position_std),
            atol=1e-2,   # eps introduces a tiny deviation
        ), (
            f"LayerNorm output std must be ≈ 1; "
            f"max |std−1| = {(per_position_std - 1).abs().max():.2e}"
        )

    def test_output_finite(self) -> None:
        """LayerNorm must produce finite output for typical inputs."""
        ln = LayerNorm(d_model=D_MODEL)
        assert torch.isfinite(ln(_rand())).all()

    def test_large_input_produces_finite_output(self) -> None:
        """LayerNorm must remain finite for large-magnitude inputs."""
        ln = LayerNorm(d_model=D_MODEL)
        x = _rand() * 1e4
        assert torch.isfinite(ln(x)).all()

    def test_uniform_input_does_not_produce_nan(self) -> None:
        """Constant input (zero variance) must not produce NaN.

        ``nn.LayerNorm`` adds eps to the denominator precisely to handle
        this degenerate case.
        """
        ln = LayerNorm(d_model=D_MODEL)
        x = torch.ones(BATCH, SEQ_LEN, D_MODEL)    # all same → variance = 0
        out = ln(x)
        assert torch.isfinite(out).all(), "Uniform input must not produce NaN/Inf."


class TestLayerNormAffineParameters:
    """Learnable γ and β must exist with the correct shapes."""

    def test_weight_exists_and_shape(self) -> None:
        """norm.weight (γ) must be a tensor of shape (d_model,)."""
        ln = LayerNorm(d_model=D_MODEL)
        assert ln.norm.weight is not None
        assert ln.norm.weight.shape == (D_MODEL,)

    def test_bias_exists_and_shape(self) -> None:
        """norm.bias (β) must be a tensor of shape (d_model,)."""
        ln = LayerNorm(d_model=D_MODEL)
        assert ln.norm.bias is not None
        assert ln.norm.bias.shape == (D_MODEL,)

    def test_weight_initialised_to_ones(self) -> None:
        """nn.LayerNorm initialises γ = 1; verify this is preserved."""
        ln = LayerNorm(d_model=D_MODEL)
        assert torch.all(ln.norm.weight == 1.0)

    def test_bias_initialised_to_zeros(self) -> None:
        """nn.LayerNorm initialises β = 0; verify this is preserved."""
        ln = LayerNorm(d_model=D_MODEL)
        assert torch.all(ln.norm.bias == 0.0)


class TestLayerNormGradientFlow:
    """Gradients must propagate back through LayerNorm."""

    def test_grad_to_input(self) -> None:
        ln = LayerNorm(d_model=D_MODEL)
        x = _rand(requires_grad=True)
        ln(x).sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_grad_to_weight(self) -> None:
        ln = LayerNorm(d_model=D_MODEL)
        x = _rand()
        ln(x).sum().backward()
        assert ln.norm.weight.grad is not None
        assert not torch.all(ln.norm.weight.grad == 0)

    def test_grad_to_bias(self) -> None:
        ln = LayerNorm(d_model=D_MODEL)
        x = _rand()
        ln(x).sum().backward()
        assert ln.norm.bias.grad is not None
        # Bias gradient is the sum of all output elements → definitely non-zero
        assert not torch.all(ln.norm.bias.grad == 0)


class TestLayerNormDtypeDevice:
    """Output dtype and device must match the input."""

    def test_output_dtype_float32(self) -> None:
        ln = LayerNorm(d_model=D_MODEL)
        out = ln(_rand())
        assert out.dtype == torch.float32

    def test_output_on_cpu(self) -> None:
        ln = LayerNorm(d_model=D_MODEL)
        out = ln(_rand())
        assert out.device.type == "cpu"

    def test_stored_d_model_attribute(self) -> None:
        ln = LayerNorm(d_model=48)
        assert ln.d_model == 48


class TestLayerNormConstructorValidation:
    """Invalid d_model must raise ValueError."""

    def test_d_model_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="d_model"):
            LayerNorm(d_model=0)

    def test_d_model_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="d_model"):
            LayerNorm(d_model=-32)


# ===========================================================================
# ResidualConnection tests
# ===========================================================================


class TestResidualConnectionShape:
    """Output shape must equal input shape."""

    def test_basic_output_shape(self) -> None:
        rc = ResidualConnection(d_model=D_MODEL)
        x = _rand()
        sub = _rand()
        out = rc(x, sub)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_batch_size_1(self) -> None:
        rc = ResidualConnection(d_model=D_MODEL)
        x, sub = _rand(batch=1), _rand(batch=1)
        assert rc(x, sub).shape == (1, SEQ_LEN, D_MODEL)

    def test_batch_size_8(self) -> None:
        rc = ResidualConnection(d_model=D_MODEL)
        x, sub = _rand(batch=8), _rand(batch=8)
        assert rc(x, sub).shape == (8, SEQ_LEN, D_MODEL)

    def test_seq_len_1(self) -> None:
        rc = ResidualConnection(d_model=D_MODEL)
        x, sub = _rand(seq_len=1), _rand(seq_len=1)
        assert rc(x, sub).shape == (BATCH, 1, D_MODEL)

    def test_seq_len_50(self) -> None:
        rc = ResidualConnection(d_model=D_MODEL)
        x, sub = _rand(seq_len=50), _rand(seq_len=50)
        assert rc(x, sub).shape == (BATCH, 50, D_MODEL)

    def test_small_d_model(self) -> None:
        rc = ResidualConnection(d_model=8)
        x, sub = _rand(d_model=8), _rand(d_model=8)
        assert rc(x, sub).shape == (BATCH, SEQ_LEN, 8)

    def test_large_d_model(self) -> None:
        rc = ResidualConnection(d_model=512)
        x, sub = _rand(d_model=512), _rand(d_model=512)
        assert rc(x, sub).shape == (BATCH, SEQ_LEN, 512)


class TestResidualConnectionPostNorm:
    """Verify the post-norm formula: LayerNorm(x + sublayer_output).

    Requirement 15: confirm the module computes post-norm, NOT pre-norm.
    """

    def test_matches_postnorm_formula(self) -> None:
        """output must equal LayerNorm(x + sub), not x + LayerNorm(sub)."""
        rc = ResidualConnection(d_model=D_MODEL)
        rc.eval()

        x = _rand()
        sub = _rand()

        result = rc(x, sub)

        # ---- What it SHOULD be: LayerNorm(x + sub) ---
        expected_postnorm = rc.layer_norm(x + sub)
        assert torch.allclose(result, expected_postnorm, atol=1e-6), (
            "ResidualConnection must compute LayerNorm(x + sublayer_output)."
        )

    def test_does_not_match_prenorm_formula(self) -> None:
        """output must NOT equal x + LayerNorm(sub) (the pre-norm variant)."""
        rc = ResidualConnection(d_model=D_MODEL)
        rc.eval()

        # Use a large, varied input so the two formulas produce clearly
        # different results with overwhelming probability.
        torch.manual_seed(42)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL) * 3.0
        sub = torch.randn(BATCH, SEQ_LEN, D_MODEL) * 3.0

        result = rc(x, sub)                          # LayerNorm(x + sub)
        prenorm = x + rc.layer_norm(sub)             # x + LayerNorm(sub)

        assert not torch.allclose(result, prenorm, atol=1e-3), (
            "ResidualConnection must NOT produce the pre-norm result "
            "(x + LayerNorm(sublayer_output))."
        )

    def test_postnorm_differs_from_prenorm_numerically(self) -> None:
        """Quantify the difference: post-norm and pre-norm are not the same.

        For non-trivial x and sub, LayerNorm(x + sub) ≠ x + LayerNorm(sub)
        because LayerNorm normalises the *sum* in post-norm vs. only the
        sublayer output in pre-norm.
        """
        rc = ResidualConnection(d_model=D_MODEL)
        rc.eval()

        torch.manual_seed(7)
        x = torch.randn(1, 4, D_MODEL)
        sub = torch.randn(1, 4, D_MODEL)

        postnorm_out = rc(x, sub)                    # LayerNorm(x + sub)
        prenorm_out  = x + rc.layer_norm(sub)        # x + LayerNorm(sub)

        max_diff = (postnorm_out - prenorm_out).abs().max().item()
        assert max_diff > 1e-3, (
            f"Post-norm and pre-norm should differ substantially; max diff = {max_diff:.6f}"
        )


class TestResidualConnectionBehavior:
    """Correctness tests for various input conditions."""

    def test_zero_sublayer_output_equals_layer_norm_of_x(self) -> None:
        """When sublayer_output = 0, output must equal LayerNorm(x + 0) = LayerNorm(x).

        This is the identity-like case: a sublayer that produces no change
        (zeroed output) should result in the same LayerNorm applied to x alone.
        """
        rc = ResidualConnection(d_model=D_MODEL)
        rc.eval()

        x = _rand()
        zero_sub = torch.zeros_like(x)

        result = rc(x, zero_sub)
        expected = rc.layer_norm(x)

        assert torch.allclose(result, expected, atol=1e-6), (
            "With zero sublayer output, ResidualConnection must equal LayerNorm(x)."
        )

    def test_changing_x_changes_output(self) -> None:
        """Changing x (with same sub) must change the output."""
        rc = ResidualConnection(d_model=D_MODEL)
        rc.eval()
        sub = _rand()
        out1 = rc(_rand(), sub)
        out2 = rc(_rand(), sub)
        # Different random x values should (almost certainly) give different outputs
        assert not torch.allclose(out1, out2, atol=1e-4), (
            "Changing x must change the output."
        )

    def test_changing_sublayer_output_changes_output(self) -> None:
        """Changing sublayer_output (with same x) must change the output."""
        rc = ResidualConnection(d_model=D_MODEL)
        rc.eval()
        x = _rand()
        out1 = rc(x, _rand())
        out2 = rc(x, _rand())
        assert not torch.allclose(out1, out2, atol=1e-4), (
            "Changing sublayer_output must change the output."
        )

    def test_output_is_normalised_per_position(self) -> None:
        """Each position's output must have mean ≈ 0 (default γ=1, β=0).

        Since the final operation is LayerNorm, the output should inherit
        the normalisation property — each d_model slice has mean ≈ 0.
        """
        rc = ResidualConnection(d_model=D_MODEL)
        x = _rand() * 5.0
        sub = _rand() * 5.0
        out = rc(x, sub)   # (B, seq_len, d_model)

        per_pos_mean = out.mean(dim=-1)   # (B, seq_len)
        assert torch.allclose(
            per_pos_mean,
            torch.zeros_like(per_pos_mean),
            atol=1e-5,
        ), "ResidualConnection output mean per position must be ≈ 0."

    def test_output_finite(self) -> None:
        rc = ResidualConnection(d_model=D_MODEL)
        out = rc(_rand(), _rand())
        assert torch.isfinite(out).all()

    def test_output_dtype_float32(self) -> None:
        rc = ResidualConnection(d_model=D_MODEL)
        assert rc(_rand(), _rand()).dtype == torch.float32

    def test_output_on_cpu(self) -> None:
        rc = ResidualConnection(d_model=D_MODEL)
        assert rc(_rand(), _rand()).device.type == "cpu"


class TestResidualConnectionGradientFlow:
    """Gradients must flow back to both x and sublayer_output."""

    def test_grad_to_x(self) -> None:
        """Gradient must reach x after backward()."""
        rc = ResidualConnection(d_model=D_MODEL)
        x = _rand(requires_grad=True)
        sub = _rand()
        rc(x, sub).sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_grad_to_sublayer_output(self) -> None:
        """Gradient must reach sublayer_output after backward()."""
        rc = ResidualConnection(d_model=D_MODEL)
        x = _rand()
        sub = _rand(requires_grad=True)
        rc(x, sub).sum().backward()
        assert sub.grad is not None
        assert not torch.all(sub.grad == 0)

    def test_grad_to_layer_norm_weight(self) -> None:
        """Gradient must reach the LayerNorm's γ parameter."""
        rc = ResidualConnection(d_model=D_MODEL)
        rc(_rand(), _rand()).sum().backward()
        assert rc.layer_norm.norm.weight.grad is not None

    def test_grad_to_layer_norm_bias(self) -> None:
        """Gradient must reach the LayerNorm's β parameter."""
        rc = ResidualConnection(d_model=D_MODEL)
        rc(_rand(), _rand()).sum().backward()
        assert rc.layer_norm.norm.bias.grad is not None

    def test_grad_flows_with_large_tensors(self) -> None:
        """Gradient flow must work for larger tensors."""
        rc = ResidualConnection(d_model=128)
        x = _rand(batch=4, seq_len=50, d_model=128, requires_grad=True)
        sub = _rand(batch=4, seq_len=50, d_model=128, requires_grad=True)
        rc(x, sub).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)
        assert sub.grad is not None and not torch.all(sub.grad == 0)


class TestResidualConnectionConstructorValidation:
    """Invalid d_model must raise ValueError (propagated from LayerNorm)."""

    def test_d_model_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="d_model"):
            ResidualConnection(d_model=0)

    def test_d_model_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="d_model"):
            ResidualConnection(d_model=-16)
