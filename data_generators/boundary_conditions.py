import torch
from abc import ABC, abstractmethod

class BoundaryCondition():
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