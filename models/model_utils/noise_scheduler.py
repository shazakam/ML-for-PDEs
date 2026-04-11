import torch
from abc import ABC, abstractmethod

class NoiseScheduler(ABC):
    def __init__(self, T, s):
        super().__init__()
        self.T = T
        self.s = s

    @abstractmethod
    def schedule(self):
        return

class CosineScheduler(NoiseScheduler):
    def __init__(self, T: int, s: float) -> None:
        super().__init__(T, s)
        
    def schedule(self):
        t       = torch.linspace(0, self.T, self.T+1)
        f_t     = torch.cos((t/self.T + self.s) / (1 + self.s) * torch.pi / 2) ** 2
        alpha_bar_t = f_t / f_t[0]
        return alpha_bar_t
