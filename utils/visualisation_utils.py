import torch
import imageio.v2 as imageio
import numpy as np
import matplotlib.cm as cm
from typing import Sequence
import matplotlib.pyplot as plt

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
    :param cmap: Matplotlib colormap name to apply (e.g. "inferno", "viridis").
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


def output_comparison(target, diffusion_pred, deep_onet_pred, fno_pred):
    """Build a 1x4 comparison figure (Target | Diffusion | Deep ONet | FNO).

    Each input is a frame sequence shaped (T, H, W) or (T, 1, H, W) — a
    singleton channel dim is squeezed and all four are trimmed to the shortest
    frame count so the columns animate in lockstep on a shared colour scale.

    :returns: (ims, frames_grid) where ims are the four AxesImage
              handles and frames_grid is the stacked (4, T, H, W) tensor
              consumed by :func:`update`.
    :rtype: tuple[list, torch.Tensor]
    """
    models = ['Target', 'Diffusion', 'Deep ONet', 'FNO']

    tensors = [target, diffusion_pred, deep_onet_pred, fno_pred]
    # Predictions arrive as (T, 1, H, W); drop the singleton channel dim.
    tensors = [t.squeeze(1) if t.dim() == 4 and t.shape[1] == 1 else t for t in tensors]
    # Trim to the shortest frame count so every column has the same number of frames.
    num_frames = min(t.shape[0] for t in tensors)
    tensors = [t[:num_frames] for t in tensors]

    frames_grid = torch.stack(tensors)                       # (4, T, H, W)

    cmap = 'inferno'
    vmin = min(t.min().item() for t in tensors)
    vmax = max(t.max().item() for t in tensors)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    ims = []
    for i in range(4):
        axes[i].set_title(f'{models[i]}', fontsize=13)
        im = axes[i].imshow(
            frames_grid[i, 0].numpy(), animated=True,
            cmap=cmap, vmin=vmin, vmax=vmax,
        )
        axes[i].set_xticks([])
        axes[i].set_yticks([])

        for spine in axes[i].spines.values():
            spine.set_visible(False)

        ims.append(im)

    fig.tight_layout()
    return ims, frames_grid

def update(frame, ims, frames_grid):
    """Advance every column to frame. frames_grid is (num_models, T, H, W)."""
    for idx in range(len(ims)):
        ims[idx].set_data(frames_grid[idx, frame].numpy())
    return ims