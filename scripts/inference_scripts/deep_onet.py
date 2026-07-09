from pathlib import Path
import argparse
import yaml
import sys
from utils.inference.deep_onet_inference_utils import deep_onet_forecast
from utils.data_processing_utils import load_single_sample
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained model")

    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a YAML config file. CLI args override config values.")

    parser.add_argument("--model-folder-path", type=Path, default=None,
                        help="Path to model folder. CLI args override config values.")

    parser.add_argument("--data-file-path", type=Path, default=None,
                        help="Path to file containing the aggregated .pt file")

    parser.add_argument("--input-idx", type=int, default=None, help="Test input index to use")

    parser.add_argument("--forecast-steps", type=int, default=None, help="Number of forecast steps")

    args = parser.parse_args()

    # Defaults (used when neither config nor CLI provides a value)
    defaults = {
        "model_folder_path": None,
        "data_file_path": None,
        "input_idx": 0,
        "forecast_steps": None
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

    # Validate only after config + CLI values have been merged in
    for k in defaults.keys():
        if defaults[k] is None:
            sys.exit(f"Missing config requirement: {k}")

    return argparse.Namespace(**defaults)


def main() -> None:
    print('Beginning inference...')
    args = parse_args()

    # Get model path
    ckpt_files = list(Path(args.model_folder_path).glob("*.ckpt"))
    model_path = str(max(ckpt_files, key=lambda p: p.stat().st_mtime))

    # Get cfg path
    yaml_files = list(Path(args.model_folder_path).glob("*.yaml"))
    model_cfg_path = max(yaml_files, key=lambda p: p.stat().st_mtime)
    with open(model_cfg_path, "r") as file:
        model_cfg = yaml.safe_load(file)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    field_keys = model_cfg["field_keys"]
    X = load_single_sample(field_keys=field_keys, data_file_path=str(args.data_file_path), input_idx=args.input_idx)

    # The model is a fixed one-step operator (u_n -> u_n + Δ); multi-step forecasting
    # is autoregressive, so forecast_steps can exceed the training horizon.
    forecast = deep_onet_forecast(model_path, model_cfg, args.forecast_steps, device, X)
    torch.save(forecast, f"data/inference_output/{str(model_cfg_path).split('/')[1]}.pt") 

if __name__ == "__main__":
    main()