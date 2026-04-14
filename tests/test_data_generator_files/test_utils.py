import pytest
from utils.conjugate_gradient import ConjugateGradient
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


# ---------------------------------------------------------------------------
# ConjugateGradient
# ---------------------------------------------------------------------------

def test_CG():

    A = torch.tensor([[4, 1, 0],
                    [1, 3, 1],
                    [0, 1, 2]], dtype = torch.float32)

    b = torch.tensor([1, 2, 3], dtype = torch.float32)

    CG = ConjugateGradient(10e-9, 1000)

    x = CG.solve(A, b, torch.zeros_like(b, dtype = torch.float32))
    x_true = torch.linalg.solve(A, b) # type: ignore
    l2_error = torch.norm(x_true - x) # type: ignore
    print(f'CG L2 ERROR: {l2_error}')
    assert l2_error < 10e-6


def _make_spd(n: int, seed: int) -> torch.Tensor:
    """Generate a random symmetric positive-definite matrix of size n×n.

    Constructs ``M @ M.T + n*I`` from a random matrix M, guaranteeing
    positive definiteness with condition number roughly proportional to n.

    :param n: Matrix dimension.
    :type n: int
    :param seed: RNG seed for reproducibility.
    :type seed: int
    :returns: Random SPD matrix of shape (n, n).
    :rtype: torch.Tensor
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    M = torch.randn(n, n, dtype=torch.float32, generator=gen)
    return M @ M.T + n * torch.eye(n, dtype=torch.float32)


def _make_poisson_1d(n: int) -> torch.Tensor:
    """Build the n×n 1D Poisson finite-difference matrix with Dirichlet BCs.

    This is the standard tridiagonal SPD matrix arising from discretising
    ``-u'' = f`` on [0,1] with n interior points.

    :param n: Number of interior grid points.
    :type n: int
    :returns: Tridiagonal SPD matrix of shape (n, n).
    :rtype: torch.Tensor
    """
    A = 2.0 * torch.eye(n, dtype=torch.float64)
    A -= torch.diag(torch.ones(n - 1, dtype=torch.float64),  1) # type: ignore
    A -= torch.diag(torch.ones(n - 1, dtype=torch.float64), -1) # type: ignore
    return A


@pytest.mark.parametrize("n, seed", [
    (10,  0),
    (50,  7),
    (100, 13),
])
def test_CG_random_spd(n: int, seed: int):
    """CG converges on a random n×n SPD system.

    :param n: Matrix dimension.
    :type n: int
    :param seed: RNG seed used to generate A and b.
    :type seed: int
    """
    A = _make_spd(n, seed)
    gen = torch.Generator()
    gen.manual_seed(seed + 1)
    b = torch.randn(n, dtype=torch.float32, generator=gen)

    CG = ConjugateGradient(1e-6, 2 * n)
    x = CG.solve(A, b, torch.zeros(n, dtype=torch.float32))
    x_true = torch.linalg.solve(A, b) # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    l2_error = torch.norm(x_true - x) # type: ignore
    print(f'CG L2 ERROR (random SPD, n={n}): {l2_error}')
    assert l2_error < 1e-3


@pytest.mark.parametrize("n", [10, 50, 100])
def test_CG_poisson_1d(n: int):
    """CG converges on the 1D Poisson finite-difference system.

    Uses a uniform right-hand side ``b = ones(n)``.

    :param n: Number of interior grid points.
    :type n: int
    """
    A = _make_poisson_1d(n)
    b = torch.ones(n, dtype=torch.float64)

    CG = ConjugateGradient(1e-6, 2 * n)
    x = CG.solve(A, b, torch.zeros(n, dtype=torch.float64))
    x_true = torch.linalg.solve(A, b) # type: ignore

    l2_error = torch.norm(x_true - x) # type: ignore
    print(f'CG L2 ERROR (Poisson 1D, n={n}): {l2_error}')
    assert l2_error < 1e-3

@pytest.mark.parametrize("m", [10, 50, 100])
def test_cyclical_boundary_class(m:int):
    dtype = torch.float64
    one_tensor = torch.ones((m,m), dtype=dtype)
    lp = Laplacian()

    bc = PeriodicBoundary()

    output = bc.apply_boundary_condition(one_tensor, lp.get_kernel(device=torch.device("cpu"), dtype=dtype))

    assert torch.equal(output, torch.zeros_like(one_tensor, dtype=dtype))


