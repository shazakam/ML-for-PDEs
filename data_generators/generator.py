import torch
from abc import ABC, abstractmethod
from typing import Any

class DataGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def timestep(self, *args: Any, **kwargs:Any) -> torch.Tensor:
        """Override this method in subclasses with your specific parameters."""
        pass

    @abstractmethod
    def generate_simulation_run(self, *args: Any, **kwargs:Any)-> torch.Tensor:
        pass

    @abstractmethod
    def generate_dataset(self, *args:Any, **kwargs:Any) -> torch.Tensor:
        pass
    
    @abstractmethod
    def generate_random_initial_condition(self, *args:Any, **kwards:Any) -> torch.Tensor:
        pass
    
    def save_data(self, file_path : str):
        return 