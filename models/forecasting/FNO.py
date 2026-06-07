from typing import Any
from typing import Any, Mapping, Sequence
import torch 
import torch.nn as nn
import lightning as L
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from model_utils.nn_helpers.ffn import FFN
class FNO(L.LightningModule):
    def __init__(self, d_a : int, d_v : int, num_fourier_layers : int, num_fourier_modes : int, d_out : int, P : FFN, Q : FFN) -> None:
        super().__init__()

        self.P = P
        self.Q = Q

    def forward(self, x):
        x = self.P(x)
        
        return
    
    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor | Mapping[str, Any] | None:
        return super().training_step(*args, **kwargs)
    
    def validation_step(self, *args: Any, **kwargs: Any) -> torch.Tensor | Mapping[str, Any] | None:
        return super().validation_step(*args, **kwargs)
    
    def configure_optimizers(self) -> Optimizer | Sequence[Optimizer] | tuple[Sequence[Optimizer], Sequence[LRScheduler | ReduceLROnPlateau]] | None:
        raise ValueError(f"Unsupported optimiser: '{self.optimiser}'. Choose 'adam' or 'adamw'.") 
