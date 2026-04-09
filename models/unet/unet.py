import torch
import torch.nn as nn
import lightning as L
from torch import Tensor

from models.unet.model_blocks import (
    SinusoidalTimestepEmbedding,
    UNetEncoder,
    UNetDecoder,
    UNetMidBlock,
)


class UNet(L.LightningModule):
    """UNet backbone for DDPM-style diffusion models.

    The architecture is a standard 4-level encoder-decoder UNet with skip connections.
    Diffusion timestep conditioning is applied at every block via a sinusoidal embedding
    projected and added to intermediate feature maps.

    in_channels controls the total number of input channels. For a PDE diffusion
    setup this is typically 2 + N_coeff: two channels for u_0 (conditioning frame)
    and u_noisy (noisy current state), plus one channel per PDE coefficient field.

    :param in_channels: Total number of input/output channels.
    :type in_channels: int
    :param out_channels: Base channel count for the encoder/decoder.
    :type out_channels: int
    :param kernel_size: Convolution kernel size used throughout.
    :type kernel_size: int
    :param final_filters: Number of output channels of the final 1×1 convolution.
    :type final_filters: int
    :param encoder_dropout: Spatial dropout probability applied in the deeper encoder blocks.
    :type encoder_dropout: float
    :param input_size: Spatial side length of the input (assumes square). Used to restore
                       odd spatial dimensions after upsampling (e.g. 65 for a 65×65 grid).
                       Defaults to 0 (even dimensions).
    :type input_size: int
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        final_filters: int,
        encoder_dropout: float,
        input_size: int = 0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.final_filters = final_filters
        self.encoder_dropout = encoder_dropout

        emb_dim: int = out_channels * 4
        self.time_emb = SinusoidalTimestepEmbedding(dim=out_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(out_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.unet_encoder = UNetEncoder(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            emb_dim=emb_dim,
            dropout=self.encoder_dropout,
        )
        self.unet_midblock = UNetMidBlock(
            in_channels=self.out_channels * 8,
            out_channels=self.out_channels * 16,
            kernel_size=self.kernel_size,
            emb_dim=emb_dim,
        )
        self.unet_decoder = UNetDecoder(
            in_channels=self.out_channels * 16,
            out_channels=self.out_channels * 8,
            kernel_size=self.kernel_size,
            emb_dim=emb_dim,
            input_size=input_size,
        )
        self.final_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.final_filters,
            kernel_size=1,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict the noise component of a noisy sample at diffusion timestep t.

        :param x: Concatenated input of shape (B, in_channels, H, W), where channels
                  are [u_0, u_noisy, coeff_1, ..., coeff_N].
        :type x: torch.Tensor
        :param t: Integer diffusion timestep indices of shape (B,).
        :type t: torch.Tensor
        :returns: Predicted noise of shape (B, final_filters, H', W').
        :rtype: torch.Tensor
        """
        emb = self.time_mlp(self.time_emb(t))
        x, skip_connections = self.unet_encoder(x, emb)
        x = self.unet_midblock(x, emb)
        x = self.unet_decoder(x, skip_connections, emb)
        x = self.final_conv(x)
        return x


class ImageSegmentationModel(L.LightningModule):
    def __init__(self, model, loss_function, lr, scheduler_step, scheduler_gamma):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.lr = lr
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_step = scheduler_step

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_gamma,
            patience=self.scheduler_step,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
