"""Script to generate 2D Gray-Scott reaction-diffusion training data.

Instantiates :class:data_generators.gray_scott_2d.GrayScottEquation with
the given parameters and saves one .pt file per simulation run to the output
directory.

Usage::

    uv run scripts/data_generation_scripts/generate_gray_scott_data.py \\
        --n-samples 1000 --m 64 --num-steps 2000

Each saved file contains a dict with keys:
- u: u-field trajectory, shape (num_steps, m, m)
- v: v-field trajectory, shape (num_steps, m, m)
- F: feed rate (float, sampled per run)
- k: kill rate (float, sampled per run)

Du and Dv are fixed across the dataset; F and k are sampled uniformly from
[F_min, F_max] and [k_min, k_max] respectively for each simulation run.

Typical parameter ranges that produce interesting patterns:
- Coral/labyrinthine: Du=0.16, Dv=0.08, F≈0.035, k≈0.065
- Spots:             Du=0.16, Dv=0.08, F≈0.035, k≈0.060
- Stripes:           Du=0.16, Dv=0.08, F≈0.060, k≈0.062
"""

import argparse
from pathlib import Path

import torch
import yaml

from data_generators.boundary_operator import PeriodicBoundary
from data_generators.gray_scott_2d import GrayScottEquation

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw_data" / "gray_scott_2d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 2D Gray-Scott reaction-diffusion training data."
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a YAML config file. CLI args override config values.")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of simulation runs (default: 1000)")
    parser.add_argument("--m", type=int, default=None,
                        help="Grid side length (default: 64)")
    parser.add_argument("--h", type=float, default=None,
                        help="Spatial grid spacing (default: 1.0)")
    parser.add_argument("--time", type=float, default=None,
                        help="Total simulation time (default: 2000.0)")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Number of timesteps (default: 2000, giving dt=1.0)")
    parser.add_argument("--Du", type=float, default=None,
                        help="Diffusion coefficient for u (default: 0.16)")
    parser.add_argument("--Dv", type=float, default=None,
                        help="Diffusion coefficient for v (default: 0.08)")
    parser.add_argument("--F-min", type=float, default=None,
                        help="Minimum feed rate (default: 0.03)")
    parser.add_argument("--F-max", type=float, default=None,
                        help="Maximum feed rate (default: 0.07)")
    parser.add_argument("--k-min", type=float, default=None,
                        help="Minimum kill rate (default: 0.055)")
    parser.add_argument("--k-max", type=float, default=None,
                        help="Maximum kill rate (default: 0.070)")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device, e.g. cpu / cuda / mps (default: cpu)")
    parser.add_argument("--dtype", type=str, default=None,
                        choices=list(DTYPE_MAP.keys()),
                        help="Floating-point dtype (default: float32)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")

    args = parser.parse_args()

    # Defaults (used when neither config nor CLI provides a value)
    defaults = {
        "n_samples": 1000,
        "m": 64,
        "h": 1.0,
        "time": 2000.0,
        "num_steps": 2000,
        "Du": 0.16,
        "Dv": 0.08,
        "F_min": 0.03,
        "F_max": 0.06,
        "k_min": 0.050,
        "k_max": 0.062,
        "device": "cpu",
        "dtype": "float32",
        "out_dir": DEFAULT_OUT_DIR,
    }

    # Load config file if provided
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            defaults[key] = value

    # CLI args override config/defaults (only when explicitly passed)
    cli = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    defaults.update(cli)

    return argparse.Namespace(**defaults)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bc = PeriodicBoundary()
    generator = GrayScottEquation(
        n_samples=args.n_samples,
        m=args.m,
        bc=bc,
        h=args.h,
        time=args.time,
        num_steps=args.num_steps,
        device=device,
        dtype=dtype,
    )

    print(f"Generating {args.n_samples} samples → {args.out_dir}")
    generator.generate_dataset(
        Du=args.Du,
        Dv=args.Dv,
        F_min=args.F_min,
        F_max=args.F_max,
        k_min=args.k_min,
        k_max=args.k_max,
        folder_path=str(args.out_dir),
    )
    print("Done.")


if __name__ == "__main__":
    main()
