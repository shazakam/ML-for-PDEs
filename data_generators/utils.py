import torch
from abc import ABC, abstractmethod
from boundary_conditions import BoundaryCondition
def construct_cyclical_laplacian(m:int) -> torch.Tensor:
    """
    Constructs Cyclical laplacian of a given size
    
    :param m: Laplacian size
    :type m: int
    """

    D = torch.eye(m)*(-2)

    for i in range(m):
        D[i][(i+1)%m] = 1
        D[i][(i-1)%m] = 1

    D[0][-1] = 1
    D[-1][0] = 1

    return D

def apply_laplacian(u : torch.Tensor, bc) -> torch.Tensor:

    return