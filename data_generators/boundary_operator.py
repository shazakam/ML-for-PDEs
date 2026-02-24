import torch
from abc import ABC, abstractmethod

class Operator(ABC):
    def __init__(self):
        super().__init__()

    def get_kernel(self, dtype = torch.float64) -> torch.Tensor:
        pass

class Laplacian(Operator):
    def __init__(self):
        super().__init__()

    def get_kernel(self, dtype = torch.float64):
        return torch.tensor([[0,  1, 0],
                              [1, -4, 1],
                              [0,  1, 0]], dtype=dtype)

class BoundaryCondition(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply_boundary_condition(self, u : torch.Tensor) -> torch.Tensor:
        return
    
class PeriodicBoundary(BoundaryCondition):
    def __init__(self):
        super().__init__()
    
    def apply_boundary_condition(self, u : torch.Tensor) -> torch.Tensor:

        return 