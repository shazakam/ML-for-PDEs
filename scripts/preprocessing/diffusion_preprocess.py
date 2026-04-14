from pathlib import Path
import argparse
import yaml
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-process specified dataset")

    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a YAML config file. CLI args override config values.")
    
    parser.add_argument("--data-folder-path", type=Path, default=None,
                        help="Path to raw data")
    parser.add_argument("--normalisation_type", type=str, default=None,
                        help="Either minmax or z-norm")
    parser.add_argument("--raw_input_tensor_keys", type=list[str], default=None,
                        help="Keys containing initial conditions with tensor inputs / generated solutions")
    parser.add_argument("--pde-param", type=list[str], default=None,
                        help="pde specific parameter values")
    parser.add_argument("--ouptu-dir",  type=Path, default=None,
                        help="Output directory")
    
    args = parser.parse_args()

    # Defaults (used when neither config nor CLI provides a value)
    defaults = {
        "data_folder_path": None,
        "normalisation_type": None,
        "raw_input_tensor_keys":None,
        "pde_params":None,
        "output_dir":None
    }

    # Load config file if provided
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            defaults[key] = value

    # CLI args override config/defaults (only when explicitly passed)
    cli = {k: v for k, v in vars(args).items() if k != "config" and v is not None}

    non_present = False
    missing_key = ""
    for k in defaults.keys():
        if defaults[k] == None:
            non_present = True
            missing_key = k
            break
            
    if non_present:
        sys.exit(f"Missing config requirement: {missing_key}")
    defaults.update(cli)

    return argparse.Namespace(**defaults)

def main() -> None:

    pass