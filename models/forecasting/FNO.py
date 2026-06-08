from typing import Any
from typing import Any, Mapping, Sequence
import torch 
import torch.nn as nn
import lightning as L
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from model_utils.nn_helpers.ffn import FFN
from model_utils.nn_helpers.FNO_layer import FNOLayer
import torch.nn.functional as F

class FNO(L.LightningModule):
    def __init__(self, d_a : int, d_v : int, num_fourier_layers : int, num_fourier_modes : int, P : FFN, Q : FFN, optimiser: str, learning_rate: float) -> None:
        super().__init__()
        self.d_a = d_a
        self.d_v = d_v
        self.num_fourier_layers = num_fourier_layers
        self.num_fourier_modes = num_fourier_modes

        self.optimiser = optimiser
        self.learning_rate = learning_rate

        self.P = P
        self.Q = Q
        self.fourier_layers = nn.ModuleList([FNOLayer(self.num_fourier_modes, self.d_v) for i in range(self.num_fourier_layers)])

    def forward(self, x):
        x = self.P(x) # Projection layer : Projects every sampled point to a higher dimension

        # Splits into a spectral branch and a spatial branch. Spectral branch truncates / selects the lower fourier modes to perform operations on. 
        # Learns / applies spectral weights per mode (in this case for every (k_i, k_j) pair)
        # Applies inverse transform to truncated modes after weights are applied to convert from spectral to spatial space
        # Spatial branch applies a linear transform to every point across channels / dimension which is then summed to the output of the spectral branch
        x = self.fourier_layers(x)

        x = self.Q(x) # Projects output of the fourier layer back to our desired output / to perform backprop with

        return x
    
    def training_step(self, batch:torch.Tensor, batch_idx : int) -> torch.Tensor | Mapping[str, Any] | None:
        # u_0    : (B, C, H, W) — conditioning frame + PDE param channels
        # target : (B, H, W)   — next PDE state to forecast
        u_0, target  = batch

        target = target.unsqueeze(1)                                    # (B, 1, H, W)
        u_next = self.forward(u_0)

        output_loss = F.mse_loss(u_next,target)
        self.log("train_loss", output_loss, on_step=True, on_epoch=True, prog_bar=True)
        return output_loss
    
    def validation_step(self, batch : torch.Tensor, batch_idx : int) -> torch.Tensor | Mapping[str, Any] | None:
        u_0, target  = batch

        target = target.unsqueeze(1)                                    # (B, 1, H, W)
        u_next = self.forward(u_0)

        output_loss = F.mse_loss(u_next,target)
        self.log("val_loss", output_loss, on_epoch=True, prog_bar=True)

        return output_loss
    
    def configure_optimizers(self):
        if self.optimiser.lower() == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.optimiser.lower() == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        raise ValueError(f"Unsupported optimiser: '{self.optimiser}'. Choose 'adam' or 'adamw'.")   
