import sys
import argparse
import yaml
from pathlib import Path
import torch
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt

from utils.visualisation_utils import output_comparison, update


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 3-column GIF comparing model forecasts to the target.")

    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a YAML config file. CLI args override config values.")

    parser.add_argument("--diffusion-pred-path", type=str, default=None)
    parser.add_argument("--fno-pred-path", type=str, default=None)
    parser.add_argument("--target-path", type=str, default=None,
                        help="Path to the aggregated .pt file holding the ground-truth 'X'")
    parser.add_argument("--target-idx", type=int, default=None,
                        help="Simulation index within the target file to compare against")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="Frame the forecast was seeded from (matches inference --input-idx). "
                             "Target is aligned so frame (start+1) lines up with prediction frame 0.")
    parser.add_argument("--save-path", type=str, default=None)

    args = parser.parse_args()

    defaults = {
        "diffusion_pred_path":None,
        "fno_pred_path":None,
        "target_path":None,
        "target_idx":None,
        "start_frame":0,
        "save_path":None
    }

    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        defaults.update(config)

    # CLI args override config/defaults (only when explicitly passed)
    cli = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    defaults.update(cli)

    required = [
        "diffusion_pred_path",
        "fno_pred_path", "target_path", "target_idx", "save_path"
    ]
    for key in required:
        if defaults[key] is None:
            sys.exit(f"Missing required config value: '{key}'")

    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load forecasts — each is (T, 1, H, W) from the inference scripts.
    diffusion_pred = torch.load(args.diffusion_pred_path, mmap=True)
    fno_pred       = torch.load(args.fno_pred_path, mmap=True)

    # Full ground-truth simulation for the chosen run: (T_full, H, W).
    target_full = torch.load(args.target_path, mmap=True)['X'][args.target_idx]

    # The models forecast the states *after* the seed frame, so align the target to
    # the same window: target frame (start + 1) corresponds to prediction frame 0.
    horizon = min(diffusion_pred.shape[0], fno_pred.shape[0])
    start = args.start_frame
    target = target_full[start + 1 : start + 1 + horizon]

    # Column order must match the labels in output_comparison.
    ims, frames_grid = output_comparison(target, diffusion_pred, fno_pred)
    fig = ims[0].axes.figure

    anim = FuncAnimation(
        fig, update, frames=frames_grid.shape[1],
        fargs=(ims, frames_grid), interval=200, blit=False,
    )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(save_path), writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"Saved comparison gif to {save_path}")


if __name__ == "__main__":
    main()
