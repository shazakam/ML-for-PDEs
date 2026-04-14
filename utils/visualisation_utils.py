import torch
import imageio.v2 as imageio
import numpy as np
import matplotlib.cm as cm
from typing import Sequence

def tensor_to_gif(
    tensor: torch.Tensor,
    filename: str = "simulation.gif",
    fps: int = 10,
    cmap: str = "inferno",
) -> None:
    """
    Convert a (T, H, W) tensor into a colourful GIF using a matplotlib colormap.

    :param tensor: Input tensor of shape (T, H, W) representing simulation frames.
    :type tensor: torch.Tensor
    :param filename: Output GIF file path.
    :type filename: str
    :param fps: Frames per second for the output GIF.
    :type fps: int
    :param cmap: Matplotlib colormap name to apply (e.g. ``"inferno"``, ``"viridis"``).
    :type cmap: str
    :returns: None
    :rtype: None
    """
    tensor_array: np.ndarray = tensor.detach().cpu().numpy()

    # Normalise to [0, 1] for colormap mapping
    vmin, vmax = tensor_array.min(), tensor_array.max()
    if vmax - vmin > 0:
        tensor_array = (tensor_array - vmin) / (vmax - vmin)
    else:
        tensor_array = np.zeros_like(tensor_array)

    colormap = cm.get_cmap(cmap)  # pyright: ignore[reportUnknownMemberType]
    # colormap returns (T, H, W, 4) RGBA float in [0, 1]; take RGB and convert to uint8
    coloured: np.ndarray = (colormap(tensor_array)[:, :, :, :3] * 255).astype(np.uint8)

    frames: Sequence[np.ndarray] = [frame for frame in coloured]

    imageio.mimsave(filename, frames, fps=fps, loop=0)  # pyright: ignore[reportUnknownMemberType]