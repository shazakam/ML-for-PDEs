from typing import Any, Mapping, Sequence

import torch
import lightning as L
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
import torch.nn as nn
from ..model_utils.nn_helpers.conv_block import Conv2DBlock
from ..model_utils.nn_helpers.ffn import FFN
import torch.nn.functional as F
class DeepONet(L.LightningModule):

    def __init__(self, conv_branch_layers : list,
                  conv_branch_activations : list, 
                  stride_branch : int, 
                  ffn_branch_layers : list[int], 
                  ffn_branch_activations: list[str],
                  ffn_trunk_layers : list[int],
                  ffn_trunk_activations : list[str], 
                  dropout: float,
                  optimiser:str,
                  learning_rate : float) -> None:
        
        super().__init__()
        self.optimiser = optimiser
        self.learning_rate = learning_rate

        # Branch Net
        self.branch = ONetBranch(conv_branch_layers, conv_branch_activations, stride_branch, ffn_branch_layers, ffn_branch_activations, dropout)

        # Trunk Net with FFN
        self.trunk = FFN(layer_sizes=ffn_trunk_layers, activation=ffn_trunk_activations, dropout_rate=dropout)


    def forward(self, x_branch, x_trunk) -> Any:
        x_branch = self.branch(x_branch)
        x_trunk = self.trunk(x_trunk)

        return torch.einsum('bp,bp->b', x_branch, x_trunk).unsqueeze(-1)
    
    def training_step(self, batch) -> torch.Tensor | Mapping[str, Any] | None:
        # x_branch: (B, measurements + pde_coefficients, H, W) -> Current pde solution at a given time
        # x_trunk: (B, x, y, t) -> Where and when we want to predict for
        # y_target: (B, measurments) output measurement at (x,y,t) where t is the number of time steps from snpshot at x_branch
        x_branch, x_trunk, y_target = batch 

        y_hat = self.forward(x_branch, x_trunk)

        output_loss = F.mse_loss(y_hat, y_target)
        self.log("train_loss", output_loss, on_step=True, on_epoch=True, prog_bar=True)
        return output_loss
    
    def validation_step(self, batch) -> torch.Tensor | Mapping[str, Any] | None:
        # x_branch: (B, measurements + pde_coefficients, H, W) -> Current pde solution at a given time
        # x_trunk: (B, x, y, t) -> Where and when we want to predict for
        # y_target: (B, measurments) output measurement at (x,y,t) where t is the number of time steps from snpshot at x_branch
        x_branch, x_trunk, y_target = batch 

        y_hat = self.forward(x_branch, x_trunk)

        output_loss = F.mse_loss(y_hat, y_target)
        self.log("val_loss", output_loss, on_epoch=True, prog_bar=True)
        return output_loss
    
    def configure_optimizers(self) -> Optimizer | Sequence[Optimizer] | tuple[Sequence[Optimizer], Sequence[LRScheduler | ReduceLROnPlateau]] | None:
        if self.optimiser.lower() == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.optimiser.lower() == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        raise ValueError(f"Unsupported optimiser: '{self.optimiser}'. Choose 'adam' or 'adamw'.") 

# NOTE: ffn_layer input needs to be equal to flattened output of convolutional section 
class ONetBranch(nn.Module):
    def __init__(self, conv_layers : list, conv_activations : list, stride : int, ffn_layers : list[int] , ffn_activations: list[str], dropout : float) -> None:
        super().__init__()

        if len(conv_layers) - 1 != len(conv_activations):
            raise ValueError(
                    "Total number of activation functions do not match with sum of convolution layers!"
                )

        self.convolutions = nn.Sequential(*[
            Conv2DBlock(
                c_in=conv_layers[idx],
                c_out=conv_layers[idx+1],
                stride=stride,
                activation=conv_activations[idx],
            )
            for idx in range(len(conv_layers) - 1)
        ])

        self.ffn_encoder = FFN(layer_sizes=ffn_layers, activation=ffn_activations, dropout_rate=dropout)

    def forward(self,x):
        x = self.convolutions(x).flatten(start_dim=1)
        x = self.ffn_encoder(x)
        return x