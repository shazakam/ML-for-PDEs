import sys
import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from datasets.diffusion_dataset import DiffusionDataset
from models.unet.unet import UNet
from models.forecasting.diffusion import DDPM
from models.model_utils.noise_scheduler import CosineScheduler


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DDPM diffusion model on PDE data.")

    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a YAML config file. CLI args override config values.")

    # Data
    parser.add_argument("--training-data-path", type=str, default=None)
    parser.add_argument("--field-keys", type=str, nargs="+", default=None,
                        help="PDE parameter keys in each .pt file, e.g. --field-keys c")
    parser.add_argument("--batch-size", type=int, default=None)

    # UNet architecture
    parser.add_argument("--in-channels", type=int, default=None)
    parser.add_argument("--out-channels", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--final-filters", type=int, default=None)
    parser.add_argument("--encoder-dropout", type=float, default=None)
    parser.add_argument("--input-size", type=int, default=None)

    # Diffusion / noise schedule
    parser.add_argument("--num-timesteps", type=int, default=None)
    parser.add_argument("--cosine-shift", type=float, default=None)

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
        "in_channels": None,
        "out_channels": None,
        "kernel_size": None,
        "final_filters": None,
        "encoder_dropout": None,
        "input_size": 0,
        "num_timesteps": None,
        "cosine_shift": None,
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
        "training_data_path", "field_keys", "batch_size",
        "in_channels", "out_channels", "kernel_size", "final_filters", "encoder_dropout",
        "num_timesteps", "cosine_shift",
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

    with open(f"{cfg.model_save_path}/model_configs.yaml", "w") as file:
        yaml.dump(cfg, file)

    # --- Dataset & DataLoader ---
    dataset = DiffusionDataset(cfg.training_data_path, field_keys=cfg.field_keys,
                               num_timesteps=cfg.num_timesteps)
    val_size = int(len(dataset) * cfg.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers= 8, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers= 8, persistent_workers=True)

    # --- Model ---
    unet = UNet(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        kernel_size=cfg.kernel_size,
        final_filters=cfg.final_filters,
        encoder_dropout=cfg.encoder_dropout,
        input_size=cfg.input_size,
    )

    noise_schedule = CosineScheduler(T=cfg.num_timesteps, s=cfg.cosine_shift).schedule()

    model = DDPM(
        denoising_model=unet,
        noise_schedule=noise_schedule,
        optimiser=cfg.optimiser,
        learning_rate=cfg.learning_rate,
    )

    # --- WandB logger ---
    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
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

    # --- Checkpoint callback — saves best val_loss and periodic snapshots ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.model_save_path,
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        filename="ddpm-{epoch:04d}-{val_loss:.4f}",
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
