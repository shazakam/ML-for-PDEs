"""Script to generate 2D wave equation training data.

Instantiates :class:`data_generators.wave_2d.WaveEquation` with the given
parameters and saves one .pt file per simulation run to the output directory.

Usage::

    uv run scripts/data_generation_scripts/generate_wave_data.py \\
        --n-samples 1000 --m 64 --num-steps 400

Each saved file contains a dict ``{'X': tensor}`` where X has shape
``(2, num_steps, m, m)`` — index 0 is the solution field and index 1 is
the wave speed (constant channel).

The CFL stability constraint ``c * dt / h <= 1/sqrt(2)`` is enforced at
runtime. With the default settings (h=0.1, time=2.0, num_steps=400,
dt=0.005) the maximum safe wave speed is ~14.1, so the default c_max=5.0
is well within the stable regime.
"""

import argparse
from pathlib import Path

import torch
import yaml

from data_generators.boundary_operator import PeriodicBoundary
from data_generators.wave_2d import WaveEquation
from utils.data_processing_utils import aggregate_files

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw_data" / "wave_equation_2d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 2D wave equation training data.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a YAML config file. CLI args override config values.")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of simulation runs (default: 1000)")
    parser.add_argument("--m", type=int, default=None,
                        help="Grid side length (default: 64)")
    parser.add_argument("--h", type=float, default=None,
                        help="Spatial grid spacing (default: 1/m)")
    parser.add_argument("--time", type=float, default=None,
                        help="Total simulation time (default: 2.0)")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Number of leapfrog timesteps (default: 400)")
    parser.add_argument("--c-min", type=float, default=None,
                        help="Minimum wave speed (default: 0.5)")
    parser.add_argument("--c-max", type=float, default=None,
                        help="Maximum wave speed (default: 5.0)")
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
        "time": 2.0,
        "num_steps": 400,
        "c_min": 0.5,
        "c_max": 5.0,
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

    args.out_dir = Path(args.out_dir)
    separated = args.out_dir / "separated"
    separated.mkdir(parents=True, exist_ok=True)

    bc = PeriodicBoundary()
    generator = WaveEquation(
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
        c_min=args.c_min,
        c_max=args.c_max,
        folder_path=str(separated),
    )

    print("Aggregating files")
    aggregate_files(str(separated), str(args.out_dir / f"aggregated_2D_wave_sim_m{args.m}_h{h}_t{args.time}_s{args.num_steps}.pt"))
    print("Done.")


if __name__ == "__main__":
    main()
