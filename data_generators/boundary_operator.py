import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Any

class Operator(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def get_kernel(self, dtype : torch.dtype = torch.float64) -> torch.Tensor:
        pass

class Laplacian(Operator):
    def __init__(self):
        super().__init__()

    def get_kernel(self, dtype : torch.dtype  = torch.float64):
        return torch.tensor([[0,  1, 0],
                              [1, -4, 1],
                              [0,  1, 0]], dtype=dtype)

class BoundaryCondition(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply_boundary_condition(self, *args:Any, **kwargs:Any) -> torch.Tensor:
        pass
    
class PeriodicBoundary(BoundaryCondition):
    def __init__(self):
        super().__init__()
    
    def apply_boundary_condition(self, u: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
        """
        Applies an operator kernel to u with circular (periodic) padding.

        :param u: Input field of shape (H, W) or (B, H, W).
        :type u: torch.Tensor
        :param operator: 2-D convolution kernel of shape (kH, kW).
        :type operator: torch.Tensor
        :returns: Result of applying the operator, same leading shape as u.
        :rtype: torch.Tensor
        """
        squeeze = u.dim() == 2
        if squeeze:
            u = u.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

        kernel = operator.unsqueeze(0).unsqueeze(0)  # (1, 1, kH, kW)
        pad = operator.shape[0] // 2
        u_padded = F.pad(u, (pad, pad, pad, pad), mode='circular')
        result = F.conv2d(u_padded, kernel, padding=0)

        return result.squeeze(0).squeeze(0) if squeeze else result