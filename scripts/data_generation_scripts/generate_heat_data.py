"""Script to generate 2D heat equation training data.

Instantiates :class: data_generators.heat_2d.HeatEquation with the given
parameters and saves one .pt file per simulation run to the output directory.

Usage::

    uv run scripts/data_generation_scripts/generate_heat_data.py \\
        --n-samples 1000 --m 64 --num-steps 100

Each saved file contains a dict {'X': tensor, 'alpha': float} where
X has shape (2, num_steps, m, m) (solution + diffusivity channel)
and alpha is the thermal diffusivity used for that run.
"""

import argparse
from pathlib import Path

import torch
import yaml

from data_generators.boundary_operator import PeriodicBoundary
from data_generators.heat_2d import HeatEquation
from utils.data_processing_utils import aggregate_files

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw_data" / "heat_equation_2d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 2D heat equation training data.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a YAML config file. CLI args override config values.")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of simulation runs (default: 1000)")
    parser.add_argument("--m", type=int, default=None,
                        help="Grid side length (default: 64)")
    parser.add_argument("--h", type=float, default=None,
                        help="Spatial grid spacing (default: 1/m)")
    parser.add_argument("--time", type=float, default=None,
                        help="Total simulation time (default: 1.0)")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Number of ADI timesteps (default: 100)")
    parser.add_argument("--a-min", type=float, default=None,
                        help="Minimum thermal diffusivity (default: 0.01)")
    parser.add_argument("--a-max", type=float, default=None,
                        help="Maximum thermal diffusivity (default: 0.5)")
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
        "h": None,
        "time": 1.0,
        "num_steps": 100,
        "a_min": 0.01,
        "a_max": 0.5,
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

    h = args.h if args.h is not None else 1.0 / args.m
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    aggregated_dir = args.out_dir
    args.out_dir = Path(args.out_dir+'/seperated')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bc = PeriodicBoundary()
    generator = HeatEquation(
        n_samples=args.n_samples,
        m=args.m,
        bc=bc,
        h=h,
        time=args.time,
        num_steps=args.num_steps,
        device=device,
        dtype=dtype,
    )

    print(f"Generating {args.n_samples} samples → {args.out_dir}")
    generator.generate_dataset(
        a_min=args.a_min,
        a_max=args.a_max,
        folder_path=str(args.out_dir)
    )

    print('Aggregating files')
    aggregate_files(str(args.out_dir), aggregated_dir+"/aggregated_2D_heat_sim.pt")
    print("Done.")


if __name__ == "__main__":
    main()
