import pytest
from data_generators.utils import construct_cyclical_laplacian
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
