import pytest
import torch
from data_generators.boundary_operator import PeriodicBoundary, Laplacian

# ---------------------------------------------------------------------------
# Laplacian kernel
# ---------------------------------------------------------------------------

def test_laplacian_kernel_values():
    """Laplacian kernel matches the standard 5-point stencil exactly.

    :returns: None
    """
    lp = Laplacian()
    expected = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float64)
    assert torch.equal(lp.get_kernel(device=torch.device("cpu"), dtype=torch.float64), expected)


def test_laplacian_kernel_sums_to_zero():
    """The full Laplacian kernel sums to zero (flux conservation).

    :returns: None
    """
    lp = Laplacian()
    kernel = lp.get_kernel(device=torch.device("cpu"), dtype=torch.float64)
    assert kernel.sum() == 0.0


# ---------------------------------------------------------------------------
# PeriodicBoundary.apply_boundary_condition (2D)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m", [10, 32])
def test_periodic_boundary_2d_linearity(m: int):
    """apply_boundary_condition is linear: L(a*u + b*v) == a*L(u) + b*L(v).

    :param m: Grid side length.
    :type m: int
    :returns: None
    """
    dtype = torch.float64
    gen = torch.Generator()
    gen.manual_seed(0)
    u = torch.randn(m, m, dtype=dtype, generator=gen)
    v = torch.randn(m, m, dtype=dtype, generator=gen)
    a, b = 3.0, -2.0

    lp = Laplacian()
    bc = PeriodicBoundary()
    kernel = lp.get_kernel(device=torch.device("cpu"), dtype=dtype)

    lhs = bc.apply_boundary_condition(a * u + b * v, kernel)
    rhs = a * bc.apply_boundary_condition(u, kernel) + b * bc.apply_boundary_condition(v, kernel)
    assert torch.allclose(lhs, rhs, atol=1e-10)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_periodic_boundary_2d_batched_matches_unbatched(batch_size: int):
    """Batched (B, 1, H, W) input gives the same result per sample as (H, W).

    :param batch_size: Number of samples in the batch.
    :type batch_size: int
    :returns: None
    """
    m = 16
    dtype = torch.float64
    gen = torch.Generator()
    gen.manual_seed(1)
    samples = [torch.randn(m, m, dtype=dtype, generator=gen) for _ in range(batch_size)]

    lp = Laplacian()
    bc = PeriodicBoundary()
    kernel = lp.get_kernel(device=torch.device("cpu"), dtype=dtype)

    batched_input = torch.stack(samples).unsqueeze(1)  # (B, 1, H, W)
    batched_out = bc.apply_boundary_condition(batched_input, kernel)

    for i, sample in enumerate(samples):
        unbatched_out = bc.apply_boundary_condition(sample, kernel)
        assert torch.allclose(batched_out[i, 0], unbatched_out, atol=1e-10)


# ---------------------------------------------------------------------------
# PeriodicBoundary.apply_boundary_condition_one_direction (1D)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m", [10, 32])
def test_periodic_boundary_1d_constant_gives_zero(m: int):
    """1D second-difference of a constant field is zero under periodic BCs.

    :param m: Grid side length.
    :type m: int
    :returns: None
    """
    dtype = torch.float64
    bc = PeriodicBoundary()
    kernel = torch.tensor([1.0, -2.0, 1.0], dtype=dtype)
    u = torch.ones(m, m, dtype=dtype)
    out = bc.apply_boundary_condition_one_direction(u, kernel)
    assert torch.allclose(out, torch.zeros_like(u), atol=1e-12)


@pytest.mark.parametrize("m", [10, 32])
def test_periodic_boundary_1d_linearity(m: int):
    """apply_boundary_condition_one_direction is linear.

    :param m: Grid side length.
    :type m: int
    :returns: None
    """
    dtype = torch.float64
    gen = torch.Generator()
    gen.manual_seed(2)
    u = torch.randn(m, m, dtype=dtype, generator=gen)
    v = torch.randn(m, m, dtype=dtype, generator=gen)
    a, b = 2.0, -5.0

    bc = PeriodicBoundary()
    kernel = torch.tensor([1.0, -2.0, 1.0], dtype=dtype)

    lhs = bc.apply_boundary_condition_one_direction(a * u + b * v, kernel)
    rhs = (a * bc.apply_boundary_condition_one_direction(u, kernel)
           + b * bc.apply_boundary_condition_one_direction(v, kernel))
    assert torch.allclose(lhs, rhs, atol=1e-10)
