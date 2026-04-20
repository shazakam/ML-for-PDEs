from typing import Any
import torch
from torch.utils.data import Dataset
import os

# NOTE: DON'T THINK WE WILL NEED FIELD KEYS ANYLONGER AS PREPROCESSING DONE IN PREPROCESSING SCRIPTS
class DiffusionDataset(Dataset):
    def __init__(self, pde_folder_path : str, field_keys : list[str], num_timesteps: int) -> None:
        super().__init__()
        self.folder_path = pde_folder_path
        self.field_keys = field_keys
        self.T = num_timesteps
        self.data_file_paths = [f"{self.folder_path}/{file}" for file in os.listdir(pde_folder_path) if file.endswith('.pt')]
        self.num_t_steps_per_sample = torch.load(self.data_file_paths[0], weights_only=False)['X'].shape[0]

    def __len__(self):
        return len(self.data_file_paths)*self.num_t_steps_per_sample - len(self.data_file_paths)

    def __getitem__(self, index) -> Any:
        file_to_load = index // (self.num_t_steps_per_sample - 1)
        file = torch.load(self.data_file_paths[file_to_load], weights_only=False, mmap=True)

        x_from_file = index % (self.num_t_steps_per_sample - 1)
        y_from_file = (index + 1) % self.num_t_steps_per_sample

        X_t = file['X'][x_from_file]
        pde_params = self.__get_pde_param__(file)
        X = torch.stack([X_t] + [torch.full_like(X_t, param) for param in pde_params], dim=0)  # (1 + num_params, H, W)
        Y = file['X'][y_from_file]                                                               # (H, W)
        t = torch.randint(1, self.T, (1,)).item()
        return X, Y, t
    
    def __get_pde_param__(self, file) -> list:
        params = [file[key] for key in self.field_keys]
        return params