from typing import Any, Mapping, Sequence

import torch
import lightning as L
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from ..unet.model_blocks import UNetEncoder


class DeepONet(L.LightningModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Branch Net with UNet Encoder

        # Trunk Net with FFN

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return super().forward(*args, **kwargs)
    
    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor | Mapping[str, Any] | None:
        return super().training_step(*args, **kwargs)
    
    def validation_step(self, *args: Any, **kwargs: Any) -> torch.Tensor | Mapping[str, Any] | None:
        return super().validation_step(*args, **kwargs)
    
    def configure_optimizers(self) -> Optimizer | Sequence[Optimizer] | tuple[Sequence[Optimizer], Sequence[LRScheduler | ReduceLROnPlateau]] | None:
        return 
    
    def forward_total(self, x):
        return