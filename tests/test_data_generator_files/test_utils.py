import pytest
from data_generators.utils import construct_cyclical_laplacian
from utils.conjugate_gradient import ConjugateGradient
import torch

L3 = torch.tensor([
    [-2.,  1.,  1.],
    [ 1., -2.,  1.],
    [ 1.,  1., -2.]
])

L4 = torch.tensor([
    [-2.,  1.,  0.,  1.],
    [ 1., -2.,  1.,  0.],
    [ 0.,  1., -2.,  1.],
    [ 1.,  0.,  1., -2.]
])

L5 = torch.tensor([
    [-2.,  1.,  0.,  0.,  1.],
    [ 1., -2.,  1.,  0.,  0.],
    [ 0.,  1., -2.,  1.,  0.],
    [ 0.,  0.,  1., -2.,  1.],
    [ 1.,  0.,  0.,  1., -2.]
])

@pytest.mark.parametrize("m", [4, 8, 16, 32])
def test_1D_laplacian_cyclical_properties(m : int):
    L = construct_cyclical_laplacian(m)
    assert L.shape == (m, m)
    assert torch.allclose(L.sum(dim=1), torch.zeros(m))

@pytest.mark.parametrize("m, expected", [
    (3, L3),
    (4, L4),
    (5, L5),
])
def test_1D_laplacian_cyclical_construction(m : int, expected: torch.Tensor):
    LTest = construct_cyclical_laplacian(m)

    assert torch.equal(LTest, expected)

def test_CG():

    A = torch.tensor([[4, 1, 0],
                    [1, 3, 1],
                    [0, 1, 2]], dtype = torch.float32)

    b = torch.tensor([1, 2, 3], dtype = torch.float32)

    CG = ConjugateGradient(10e-9, 1000)

    x = CG.solve(A, b, torch.zeros_like(b, dtype = torch.float32))
    x_true = torch.linalg.solve(A, b)
    l2_error = torch.norm(x_true - x)
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
    A -= torch.diag(torch.ones(n - 1, dtype=torch.float64),  1)
    A -= torch.diag(torch.ones(n - 1, dtype=torch.float64), -1)
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
    x_true = torch.linalg.solve(A, b)

    l2_error = torch.norm(x_true - x)
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
    x_true = torch.linalg.solve(A, b)

    l2_error = torch.norm(x_true - x)
    print(f'CG L2 ERROR (Poisson 1D, n={n}): {l2_error}')
    assert l2_error < 1e-3

