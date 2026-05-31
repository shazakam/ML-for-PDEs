from typing import Any, Mapping, Sequence

import torch
import lightning as L
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
import torch.nn as nn
from ..model_utils.nn_helpers.conv_block import Conv2DBlock
from ..model_utils.nn_helpers.ffn import FFN

class DeepONet(L.LightningModule):

    def __init__(self, conv_branch_layers : list,
                  conv_branch_activations : list, 
                  stride_branch : int, 
                  ffn_branch_layers : list[int], 
                  ffn_branch_activations: list[str],
                  ffn_trunk_layers : list[int],
                  ffn_trunk_activations : list[str], 
                  dropout: float) -> None:
        
        super().__init__()

        # Branch Net
        self.branch = ONetBranch(conv_branch_layers, conv_branch_activations, stride_branch, ffn_branch_layers, ffn_branch_activations, dropout)

        # Trunk Net with FFN
        self.trunk = FFN(layer_sizes=ffn_trunk_layers, activation=ffn_trunk_activations, dropout_rate=dropout)


    def forward(self, x_branch, x_trunk) -> Any:
        x_branch = self.branch(x_branch)
        x_trunk = self.trunk(x_trunk)

        return x_branch @ x_trunk.T
    
    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor | Mapping[str, Any] | None:
        return super().training_step(*args, **kwargs)
    
    def validation_step(self, *args: Any, **kwargs: Any) -> torch.Tensor | Mapping[str, Any] | None:
        return super().validation_step(*args, **kwargs)
    
    def configure_optimizers(self) -> Optimizer | Sequence[Optimizer] | tuple[Sequence[Optimizer], Sequence[LRScheduler | ReduceLROnPlateau]] | None:
        return 
    
    def forward_total(self, x):
        return

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