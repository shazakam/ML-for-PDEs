import math
import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimestepEmbedding(nn.Module):
    """Non-parametric sinusoidal timestep embedding.

    Maps integer diffusion timesteps to continuous positional embeddings
    following the transformer sinusoidal encoding scheme (Vaswani et al., 2017).

    :param dim: Embedding dimension. Must be even.
    :type dim: int
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, f"dim must be even, got {dim}"
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """
        :param t: Integer timestep indices of shape (B,).
        :type t: torch.Tensor
        :returns: Sinusoidal embeddings of shape (B, dim).
        :rtype: torch.Tensor
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class UNetEncoderBlock(nn.Module):
    """Single encoder stage: two Conv2d+ReLU layers, optional dropout, and MaxPool2d.

    A sinusoidal timestep embedding is projected and added to the feature map
    after the first convolution to condition the block on the diffusion timestep.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param out_channels: Number of output channels.
    :type out_channels: int
    :param kernel_size: Convolution kernel size.
    :type kernel_size: int
    :param emb_dim: Dimension of the timestep conditioning embedding.
    :type emb_dim: int
    :param dropout: Spatial dropout probability. Defaults to 0 (disabled).
    :type dropout: float
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        emb_dim: int,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.relu = nn.ReLU()
        self.dropout = dropout
        if dropout != 0:
            self.drop = nn.Dropout2d(dropout)

        self.conv_layer_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=1,
        )
        self.conv_layer_2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=1,
        )
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.emb_proj = nn.Linear(emb_dim, out_channels)

    def forward(self, x: Tensor, emb: Tensor) -> tuple[Tensor, Tensor]:
        """
        :param x: Input feature map of shape (B, in_channels, H, W).
        :type x: torch.Tensor
        :param emb: Timestep embedding of shape (B, emb_dim).
        :type emb: torch.Tensor
        :returns: Tuple of (downsampled, skip) with shapes
                  (B, out_channels, H//2, W//2) and (B, out_channels, H, W).
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        x = self.relu(self.conv_layer_1(x))
        x = x + self.emb_proj(emb)[:, :, None, None]

        if self.dropout != 0:
            x = self.drop(self.relu(self.conv_layer_2(x)))
        else:
            x = self.relu(self.conv_layer_2(x))

        x_skip = x.clone()
        x = self.max_pool(x)
        return x, x_skip


class UNetDecoderBlock(nn.Module):
    """Single decoder stage: ConvTranspose2d upsample, skip concatenation, two Conv2d+ReLU.

    A sinusoidal timestep embedding is projected and added to the feature map
    after the first post-concatenation convolution.

    :param in_channels: Number of channels entering this block (from the level below).
                        Also used as the concatenated channel count after skip fusion.
    :type in_channels: int
    :param out_channels: Number of output channels.
    :type out_channels: int
    :param kernel_size: Convolution kernel size.
    :type kernel_size: int
    :param emb_dim: Dimension of the timestep conditioning embedding.
    :type emb_dim: int
    :param output_padding: Extra output padding for ConvTranspose2d. Pass 1 on the
                           topmost decoder block to restore odd spatial dimensions
                           (e.g. 32 → 65 instead of 32 → 64).
    :type output_padding: int
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        emb_dim: int,
        output_padding: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.up_conv_layer_1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            output_padding=output_padding,
        )
        self.conv_layer_2 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=1,
        )
        self.conv_layer_3 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.emb_proj = nn.Linear(emb_dim, out_channels)

    def forward(self, x: Tensor, skip_input: Tensor, emb: Tensor) -> Tensor:
        """
        :param x: Feature map from the level below of shape (B, in_channels, H, W).
        :type x: torch.Tensor
        :param skip_input: Skip connection from the corresponding encoder block.
        :type skip_input: torch.Tensor
        :param emb: Timestep embedding of shape (B, emb_dim).
        :type emb: torch.Tensor
        :returns: Decoded feature map of shape (B, out_channels, 2H, 2W).
        :rtype: torch.Tensor
        """
        x = self.up_conv_layer_1(x)
        x = torch.concat([skip_input, x], dim=1)
        x = self.relu(self.conv_layer_2(x))
        x = x + self.emb_proj(emb)[:, :, None, None]
        x = self.relu(self.conv_layer_3(x))
        return x


class UNetMidBlock(nn.Module):
    """Bottleneck block: two Conv2d+ReLU layers with timestep conditioning between them.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param out_channels: Number of output channels. The intermediate channel count
                         is 2 * out_channels.
    :type out_channels: int
    :param kernel_size: Convolution kernel size.
    :type kernel_size: int
    :param emb_dim: Dimension of the timestep conditioning embedding.
    :type emb_dim: int
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        emb_dim: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.relu = nn.ReLU()
        self.conv_layer_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * self.out_channels,
            kernel_size=self.kernel_size,
            padding=1,
        )
        self.conv_layer_2 = nn.Conv2d(
            in_channels=2 * self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=1,
        )
        self.emb_proj = nn.Linear(emb_dim, 2 * out_channels)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        :param x: Bottleneck feature map of shape (B, in_channels, H, W).
        :type x: torch.Tensor
        :param emb: Timestep embedding of shape (B, emb_dim).
        :type emb: torch.Tensor
        :returns: Feature map of shape (B, out_channels, H, W).
        :rtype: torch.Tensor
        """
        x = self.relu(self.conv_layer_1(x))
        x = x + self.emb_proj(emb)[:, :, None, None]
        x = self.relu(self.conv_layer_2(x))
        return x


class UNetEncoder(nn.Module):
    """Four-level UNet encoder with progressive channel doubling.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param out_channels: Base channel count (first encoder block output channels).
    :type out_channels: int
    :param kernel_size: Convolution kernel size passed to each encoder block.
    :type kernel_size: int
    :param emb_dim: Dimension of the timestep conditioning embedding.
    :type emb_dim: int
    :param dropout: Spatial dropout probability applied in encoder blocks 3 and 4.
    :type dropout: float
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        emb_dim: int,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.encoder_block_1 = UNetEncoderBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            emb_dim=emb_dim,
        )
        self.encoder_block_2 = UNetEncoderBlock(
            in_channels=self.out_channels,
            out_channels=self.out_channels * 2,
            kernel_size=self.kernel_size,
            emb_dim=emb_dim,
        )
        self.encoder_block_3 = UNetEncoderBlock(
            in_channels=self.out_channels * 2,
            out_channels=self.out_channels * 4,
            kernel_size=self.kernel_size,
            emb_dim=emb_dim,
            dropout=self.dropout,
        )
        self.encoder_block_4 = UNetEncoderBlock(
            in_channels=self.out_channels * 4,
            out_channels=self.out_channels * 8,
            kernel_size=self.kernel_size,
            emb_dim=emb_dim,
            dropout=self.dropout,
        )

    def forward(self, x: Tensor, emb: Tensor) -> tuple[Tensor, list[Tensor]]:
        """
        :param x: Input tensor of shape (B, in_channels, H, W).
        :type x: torch.Tensor
        :param emb: Timestep embedding of shape (B, emb_dim).
        :type emb: torch.Tensor
        :returns: Tuple of (bottleneck_input, [skip_1, skip_2, skip_3, skip_4])
                  ordered from highest to lowest resolution.
        :rtype: tuple[torch.Tensor, list[torch.Tensor]]
        """
        x, x_skip_1 = self.encoder_block_1(x, emb)
        x, x_skip_2 = self.encoder_block_2(x, emb)
        x, x_skip_3 = self.encoder_block_3(x, emb)
        x, x_skip_4 = self.encoder_block_4(x, emb)
        return x, [x_skip_1, x_skip_2, x_skip_3, x_skip_4]


class UNetDecoder(nn.Module):
    """Four-level UNet decoder with progressive channel halving.

    :param in_channels: Number of channels entering the first decoder block.
    :type in_channels: int
    :param out_channels: Output channels of the first decoder block. Each subsequent
                         block halves this count.
    :type out_channels: int
    :param kernel_size: Convolution kernel size passed to each decoder block.
    :type kernel_size: int
    :param emb_dim: Dimension of the timestep conditioning embedding.
    :type emb_dim: int
    :param input_size: Spatial side length of the original input. Used to compute
                       output_padding for the topmost decoder block to restore
                       odd spatial dimensions (e.g. 65×65). Defaults to 0 (even).
    :type input_size: int
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        emb_dim: int,
        input_size: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        output_padding_last: int = input_size % 2

        self.decoder_block_1 = UNetDecoderBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            emb_dim=emb_dim,
        )
        self.decoder_block_2 = UNetDecoderBlock(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 2,
            kernel_size=self.kernel_size,
            emb_dim=emb_dim,
        )
        self.decoder_block_3 = UNetDecoderBlock(
            in_channels=self.out_channels // 2,
            out_channels=self.out_channels // 4,
            kernel_size=self.kernel_size,
            emb_dim=emb_dim,
        )
        self.decoder_block_4 = UNetDecoderBlock(
            in_channels=self.out_channels // 4,
            out_channels=self.out_channels // 8,
            kernel_size=self.kernel_size,
            emb_dim=emb_dim,
            output_padding=output_padding_last,
        )

    def forward(self, x: Tensor, skip_connections: list[Tensor], emb: Tensor) -> Tensor:
        """
        :param x: Bottleneck feature map of shape (B, in_channels, H, W).
        :type x: torch.Tensor
        :param skip_connections: Skip tensors [skip_1, skip_2, skip_3, skip_4]
                                 from the encoder, consumed deepest-first.
        :type skip_connections: list[torch.Tensor]
        :param emb: Timestep embedding of shape (B, emb_dim).
        :type emb: torch.Tensor
        :returns: Decoded feature map.
        :rtype: torch.Tensor
        """
        x = self.decoder_block_1(x, skip_connections[3], emb)
        x = self.decoder_block_2(x, skip_connections[2], emb)
        x = self.decoder_block_3(x, skip_connections[1], emb)
        x = self.decoder_block_4(x, skip_connections[0], emb)
        return x
