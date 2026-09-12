import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from torch_geometric.loader import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.forecasting.GNO import GNO
from datasets.graph_datasets.graph_heat_dataset import HeatGraphDataset
from datasets.split_utils import split_by_simulation


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Graph Neural Operator on PDE data.")

    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a YAML config file. CLI args override config values.")

    # Data
    parser.add_argument("--training-data-path", type=str, default=None)
    parser.add_argument("--field-keys", type=str, nargs="+", default=None,
                        help="PDE parameter keys in each .pt file, e.g. --field-keys a")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--problem-type", type=str, default=None)

    # Graph construction
    parser.add_argument("--sub-graph-size", type=int, default=None,
                        help="Number of nodes m sampled uniformly from the grid per training example.")
    parser.add_argument("--radius", type=float, default=None,
                        help="Connection radius in normalised [0, 1) domain coordinates.")
    parser.add_argument("--boundary-condition", type=str, default=None)

    # GNO architecture
    parser.add_argument("--num-node-input-features", type=int, default=None)
    parser.add_argument("--num-edge-features", type=int, default=None)
    parser.add_argument("--num-latent-dim", type=int, default=None)
    parser.add_argument("--output-dim", type=int, default=None)
    parser.add_argument("--num-gno-layers", type=int, default=None)
    parser.add_argument("--kernel-ffn-layers", type=int, nargs="+", default=None)
    parser.add_argument("--kernel-ffn-dropout", type=float, default=None)
    parser.add_argument("--gno-layer-activation", type=str, default=None)

    # Optimiser
    parser.add_argument("--optimiser", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)

    # Lightning Trainer
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--log-every-n-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)

    # Checkpointing
    parser.add_argument("--model-save-path", type=str, default=None)
    parser.add_argument("--save-every-n-epochs", type=int, default=None)

    args = parser.parse_args()

    defaults = {
        "training_data_path": None,
        "field_keys": None,
        "batch_size": None,
        "problem_type": None,

        "sub_graph_size": None,
        "radius": None,
        "boundary_condition": "periodic",

        "num_node_input_features": None,
        "num_edge_features": None,
        "num_latent_dim": None,
        "output_dim": 1,
        "num_gno_layers": None,
        "kernel_ffn_layers": None,
        "kernel_ffn_dropout": 0.0,
        "gno_layer_activation": "relu",

        "optimiser": None,
        "learning_rate": None,
        "max_epochs": None,
        "accelerator": None,
        "devices": 1,
        "precision": 32,
        "log_every_n_steps": 50,
        "num_workers": 8,
        "model_save_path": None,
        "save_every_n_epochs": 10,
        "val_split": 0.1,
        "split_seed": 42,
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
        "sub_graph_size", "radius", "boundary_condition",
        "num_node_input_features", "num_edge_features", "num_latent_dim",
        "output_dim", "num_gno_layers", "kernel_ffn_layers",
        "optimiser", "learning_rate", "max_epochs", "accelerator", "model_save_path",
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
        dataset = HeatGraphDataset(
            aggregated_path=cfg.training_data_path,
            field_keys=cfg.field_keys,
            r=cfg.radius,
            bc=cfg.boundary_condition,
            sub_graph_size=cfg.sub_graph_size,
        )
    elif cfg.problem_type == "wave":
        sys.exit("Not implemented yet -- select heat instead")
    else:
        sys.exit("Problem type not specified and could not load dataset")

    # Split by simulation, not by frame: consecutive frames of one trajectory are
    # nearly identical, so a flat random_split leaks them across train/val and makes
    # val_loss measure recall rather than generalisation to unseen initial conditions.
    train_dataset, val_dataset = split_by_simulation(dataset, cfg.val_split, seed=cfg.split_seed)

    # torch_geometric's DataLoader batches Data objects by disjoint union, so the
    # variable node/edge counts across sampled subgraphs need no padding.
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                                num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0)

    # --- Model ---
    model = GNO(
        optimiser=cfg.optimiser,
        learning_rate=cfg.learning_rate,
        num_node_input_features=cfg.num_node_input_features,
        num_edge_features=cfg.num_edge_features,
        num_latent_dim=cfg.num_latent_dim,
        output_dim=cfg.output_dim,
        num_gno_layers=cfg.num_gno_layers,
        kernel_ffn_layers=list(cfg.kernel_ffn_layers),
        kernel_ffn_dropout=cfg.kernel_ffn_dropout,
        GNO_layer_activation=cfg.gno_layer_activation,
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

    # --- Checkpoint callback — saves best val_loss and periodic snapshots ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.model_save_path,
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        filename="gno-{epoch:04d}-{val_loss:.4f}",
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
