from typing import Any
import torch
from torch.utils.data import Dataset
from abc import ABC
import os
class PDEDiffusionDataset(Dataset):
    def __init__(self, pde_folder_path) -> None:
        super().__init__()
        self.folder_path = pde_folder_path
        self.data_file_paths = [f"{self.folder_path}/{file}" for file in os.listdir(pde_folder_path)]
        self.num_t_steps_per_sample = torch.load(self.data_file_paths[0])['X'].shape[0]

    def __len__(self):
        return len(self.data_file_paths)*self.num_t_steps_per_sample - len(self.data_file_paths)

    def __getitem__(self, index) -> Any:
        file_to_load = index // (self.num_t_steps_per_sample - 1)
        x_from_file = index % (self.num_t_steps_per_sample - 1)
        y_from_file = (index + 1)%(self.num_t_steps_per_sample)
        X = torch.load(self.data_file_paths[file_to_load])['X'][0][x_from_file] # NOTE: NEED TO CHECK THE DIMENSION THE GENERATORS CURRENTLY CREATE WRT TO SAVED PDE PARAMETERS AND WHAT THE CURRENT MODEL EXPECTS
        Y = torch.load(self.data_file_paths[file_to_load])['X'][0][y_from_file]
        return X, Y