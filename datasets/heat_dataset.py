from typing import Any
import torch
from torch.utils.data import Dataset


class HeatDiffusionDataset(Dataset):
    def __init__(self, aggregated_path: str, field_keys: list[str], num_timesteps: int) -> None:
        super().__init__()
        self.field_keys = field_keys
        self.T = num_timesteps
        # Load once with mmap=True — tensor pages are faulted in on demand, so
        # the full dataset never needs to fit in RAM.
        self.data = torch.load(aggregated_path, weights_only=False, mmap=True)
        self.N, self.num_t_steps_per_sample = self.data['X'].shape[:2]  # (N, T, H, W)

    def __len__(self) -> int:
        return self.N * (self.num_t_steps_per_sample - 1)

    
    def __getitem__(self, index) -> Any:
        sim_idx   = index // (self.num_t_steps_per_sample - 1)
        frame_idx = index %  (self.num_t_steps_per_sample - 1)

        X_t = self.data['X'][sim_idx, frame_idx]                                         # (H, W)
        pde_params = [float(self.data[k][sim_idx]) for k in self.field_keys]
        X = torch.stack([X_t] + [torch.full_like(X_t, p) for p in pde_params], dim=0)   # (1 + num_params, H, W)
        Y = self.data['X'][sim_idx, frame_idx + 1]                                       # (H, W)
        t = torch.randint(1, self.T, (1,)).item()
        return X, Y, t
    
class HeatONetDataset(Dataset):
    def __init__(self, aggregated_path: str, field_keys: list[str]) -> None:
        super().__init__()
        self.field_keys = field_keys
        self.data = torch.load(aggregated_path, weights_only=False, mmap=True)
        self.N, self.num_t_steps_per_sample, self.H, self.W = self.data['X'].shape

    def __len__(self) -> int:
        # One item per (simulation, source frame) pair; source frames are 0 … T-2
        return self.N * (self.num_t_steps_per_sample - 1)

    def __getitem__(self, index) -> Any:
        frames     = self.num_t_steps_per_sample - 1
        sim_idx    = index // frames
        src_frame  = index %  frames          # source frame in [0, T-2]

        # Branch: snapshot at source frame + PDE params — (1 + num_params, H, W)
        X_src      = self.data['X'][sim_idx, src_frame]
        pde_params = [float(self.data[k][sim_idx]) for k in self.field_keys]
        branch     = torch.stack([X_src] + [torch.full_like(X_src, p) for p in pde_params], dim=0)

        # Sample a random target frame strictly after the source frame
        tgt_frame  = torch.randint(src_frame + 1, self.num_t_steps_per_sample, (1,)).item()

        # Sample a random spatial query point
        row = torch.randint(0, self.H, (1,)).item()
        col = torch.randint(0, self.W, (1,)).item()

        # Trunk: (x, y, Δt) — time offset relative to source, normalised to [0, 1]
        delta_t = (tgt_frame - src_frame) / (self.num_t_steps_per_sample - 1)
        trunk   = torch.tensor([col / (self.W - 1), row / (self.H - 1), delta_t], dtype=torch.float32)

        # Target: solution value at the query point on the target frame
        target = self.data['X'][sim_idx, tgt_frame, row, col]

        return branch, trunk, target.unsqueeze(-1)
    

class HeatFNODataset(Dataset):
    def __init__(self, aggregated_path: str, field_keys: list[str]) -> None:
        super().__init__()
        self.field_keys = field_keys

        # Load once with mmap=True — tensor pages are faulted in on demand, so
        # the full dataset never needs to fit in RAM.
        self.data = torch.load(aggregated_path, weights_only=False, mmap=True)
        self.N, self.num_t_steps_per_sample = self.data['X'].shape[:2]  # (N, T, H, W)

    def __len__(self) -> int:
        return self.N * (self.num_t_steps_per_sample - 1)

    
    def __getitem__(self, index) -> Any:
        sim_idx   = index // (self.num_t_steps_per_sample - 1)
        frame_idx = index %  (self.num_t_steps_per_sample - 1)

        X_t = self.data['X'][sim_idx, frame_idx]                                         # (H, W)
        pde_params = [float(self.data[k][sim_idx]) for k in self.field_keys]
        X = torch.stack([X_t] + [torch.full_like(X_t, p) for p in pde_params], dim=0)   # (1 + num_params, H, W)
        Y = self.data['X'][sim_idx, frame_idx + 1]                                       # (H, W)
        
        return X, Y