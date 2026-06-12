import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from datetime import datetime
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from datasets.heat_dataset import HeatONetDataset
from models.forecasting.deep_onet import DeepONet


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DeepONet model on PDE data.")

    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a YAML config file. CLI args override config values.")

    # Data
    parser.add_argument("--training-data-path", type=str, default=None)
    parser.add_argument("--field-keys", type=str, nargs="+", default=None,
                        help="PDE parameter keys in each .pt file, e.g. --field-keys c")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--problem-type", type=str, default=None)
    parser.add_argument("--num-query-points", type=int, default=None,
                        help="Number of spatial query points sampled per function each step")

    # DeepONet Branch (convolutional section)
    parser.add_argument("--conv-branch-layers", type=int, nargs="+", default=None,
                        help="Channel sizes for conv branch, e.g. --conv-branch-layers 2 16 32 64")
    parser.add_argument("--conv-branch-activations", type=str, nargs="+", default=None,
                        help="Activations per conv block, e.g. --conv-branch-activations relu relu relu")
    parser.add_argument("--stride-branch", type=int, default=None,
                        help="Stride used in all conv branch blocks")

    # DeepONet Branch (FFN section)
    parser.add_argument("--ffn-branch-layers", type=int, nargs="+", default=None,
                        help="Layer sizes for branch FFN (first entry = flattened conv output)")
    parser.add_argument("--ffn-branch-activations", type=str, nargs="+", default=None)

    # DeepONet Trunk
    parser.add_argument("--ffn-trunk-layers", type=int, nargs="+", default=None,
                        help="Layer sizes for trunk FFN (first entry must be 3 for x,y,t)")
    parser.add_argument("--ffn-trunk-activations", type=str, nargs="+", default=None)

    # Regularisation
    parser.add_argument("--dropout", type=float, default=None)

    # Optimiser
    parser.add_argument("--optimiser", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)

    # Lightning Trainer
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--log-every-n-steps", type=int, default=None)

    # Checkpointing
    parser.add_argument("--model-save-path", type=str, default=None)
    parser.add_argument("--save-every-n-epochs", type=int, default=None)

    args = parser.parse_args()

    defaults = {
        "training_data_path": None,
        "field_keys": None,
        "batch_size": None,
        "problem_type": None,
        "num_query_points": 128,
        "conv_branch_layers": None,
        "conv_branch_activations": None,
        "stride_branch": None,
        "ffn_branch_layers": None,
        "ffn_branch_activations": None,
        "ffn_trunk_layers": None,
        "ffn_trunk_activations": None,
        "dropout": 0.1,
        "optimiser": None,
        "learning_rate": None,
        "max_epochs": None,
        "accelerator": None,
        "devices": 1,
        "precision": 32,
        "log_every_n_steps": 50,
        "model_save_path": None,
        "save_every_n_epochs": 10,
        "val_split": 0.1,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.0,
        "wandb_project": None,
        "wandb_entity": None,
        "wandb_run_name": None,
    }

    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        defaults.update(config)

    # CLI args override config/defaults (only when explicitly passed)
    cli = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    defaults.update(cli)

    required = [
        "training_data_path", "field_keys", "batch_size", "problem_type",
        "conv_branch_layers", "conv_branch_activations", "stride_branch",
        "ffn_branch_layers", "ffn_branch_activations",
        "ffn_trunk_layers", "ffn_trunk_activations",
        "optimiser", "learning_rate",
        "max_epochs", "accelerator",
        "model_save_path",
    ]
    for key in required:
        if defaults[key] is None:
            sys.exit(f"Missing required config value: '{key}'")

    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = parse_args()

    run_dir = Path(cfg.model_save_path) / datetime.now().strftime("%Y%m%d_%H%M")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_save_path = str(run_dir)

    with open(run_dir / "model_configs.yaml", "w") as file:
        yaml.dump(vars(cfg), file)

    # --- Dataset & DataLoader ---
    if cfg.problem_type == "heat":
        dataset = HeatONetDataset(cfg.training_data_path, field_keys=cfg.field_keys,
                                  num_query_points=cfg.num_query_points)
    else:
        sys.exit(f"Unsupported problem_type '{cfg.problem_type}'. Currently supported: 'heat'")

    val_size = int(len(dataset) * cfg.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=8, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                                num_workers=8, persistent_workers=True)

    # --- Model ---
    model = DeepONet(
        conv_branch_layers=cfg.conv_branch_layers,
        conv_branch_activations=cfg.conv_branch_activations,
        stride_branch=cfg.stride_branch,
        ffn_branch_layers=cfg.ffn_branch_layers,
        ffn_branch_activations=cfg.ffn_branch_activations,
        ffn_trunk_layers=cfg.ffn_trunk_layers,
        ffn_trunk_activations=cfg.ffn_trunk_activations,
        dropout=cfg.dropout,
        optimiser=cfg.optimiser,
        learning_rate=cfg.learning_rate,
    )

    # --- WandB logger ---
    now = datetime.now()
    run_name = cfg.wandb_run_name + '_' + now.strftime("%Y-%m-%d %H:%M")
    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        log_model=False,
    )
    wandb_logger.log_hyperparams(vars(cfg))

    # --- Early stopping ---
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
        mode="min",
        verbose=True,
    )

    # --- Checkpoint callback ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.model_save_path,
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        filename="deep-onet-{epoch:04d}-{val_loss:.4f}",
        every_n_epochs=cfg.save_every_n_epochs,
    )

    # --- Trainer ---
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        log_every_n_steps=cfg.log_every_n_steps,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()
