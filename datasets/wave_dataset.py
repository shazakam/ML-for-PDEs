from typing import Any

from torch.utils.data import Dataset
from diffusion_dataset import DiffusionDataset
class WaveDiffusionDataset(DiffusionDataset):
    def __init__(self, aggregated_path: str, field_keys: list[str], num_timesteps: int) -> None:
        super().__init__(aggregated_path, field_keys, num_timesteps)

    # Needs to be rewritten to match two-step wave equation input
    def __len__(self) -> int:
        return super().__len__()
    
    # Needs to be rewritten to match two-step wave equation input
    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)
    