from typing import Any
import torch
from torch.utils.data import Dataset


class DiffusionDataset(Dataset):
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
