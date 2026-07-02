from pathlib import Path
import argparse
import yaml
import sys
from utils.data_processing_utils import min_max_aggregated, z_normal_aggregated
REPO_ROOT = Path(__file__).resolve().parents[2]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-process specified dataset")

    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a YAML config file. CLI args override config values.")

    parser.add_argument("--data-folder-path", type=Path, default=None,
                        help="Path to folder containing the aggregated .pt file")
    parser.add_argument("--normalisation_type", type=str, default=None,
                        help="Either minmax or z-norm")
    parser.add_argument("--tensor_keys", type=list[str], default=None,
                        help="Keys containing initial conditions with tensor inputs / generated solutions")
    parser.add_argument("--pde-keys", type=list[str], default=None,
                        help="pde specific parameter values")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Base filename for the output .pt files (without extension)")
    parser.add_argument("--train-output-dir", type=Path, default=None,
                        help="Output directory for normalised train split")
    parser.add_argument("--test-output-dir",  type=Path, default=None,
                        help="Output directory for normalised test split")

    args = parser.parse_args()

    # Defaults (used when neither config nor CLI provides a value)
    defaults = {
        "data_folder_path": None,
        "normalisation_type": None,
        "tensor_keys": None,
        "pde_keys": None,
        "dataset_name": None,
        "train_output_dir": None,
        "test_output_dir": None,
    }

    # Load config file if provided
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            defaults[key] = value

    # CLI args override config/defaults (only when explicitly passed)
    cli = {k: v for k, v in vars(args).items() if k != "config" and v is not None}

    for k in defaults.keys():
        if defaults[k] is None:
            sys.exit(f"Missing config requirement: {k}")

    defaults.update(cli)

    return argparse.Namespace(**defaults)

def main() -> None:
    print('Beginning preprocessing...')
    args = parse_args()

    tensor_keys = [args.tensor_keys] if isinstance(args.tensor_keys, str) else list(args.tensor_keys)
    pde_keys    = [args.pde_keys]    if isinstance(args.pde_keys, str)    else list(args.pde_keys)

    if args.normalisation_type == 'minmax':
        min_max_aggregated(args.data_folder_path, tensor_keys, pde_keys,
                           args.train_output_dir, args.test_output_dir, args.dataset_name)
    elif args.normalisation_type == 'z-normal':
        z_normal_aggregated(args.data_folder_path, tensor_keys, pde_keys,
                            args.train_output_dir, args.test_output_dir, args.dataset_name)
    else:
        sys.exit('Invalid normalisation type')


if __name__ == "__main__":
    main()
