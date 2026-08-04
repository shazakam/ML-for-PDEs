from typing import Any
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from graph_data_utils import generate_discretised_graph

# This Graph Dataset assumes only a single discretisation is used
class HeatGraphDataset(Dataset):
    def __init__(self, aggregated_path: str, field_keys: list[str], num_sub_graphs : int, r : float, bc : str, max_sub_graph_size : int | None = None) -> None:
        super().__init__()
        self.field_keys = field_keys
        self.num_sub_graphs = num_sub_graphs
        self.data = torch.load(aggregated_path, weights_only=False, mmap=True)
        self.N, self.num_t_steps_per_sample = self.data['X'].shape[:2]  # (N, T, H, W)

        if max_sub_graph_size is None: 
            self.max_sub_graph_size = self.data['X'].shape[-1] * self.data['X'].shape[-2]
        elif (max_sub_graph_size > self.data['X'].shape[-1] * self.data['X'].shape[-2] or max_sub_graph_size <= 0):
            raise ValueError("Max sub graph size too large or too small / negative")
        
        else:
            self.max_sub_graph_size = self.data['X'].shape[-1] * self.data['X'].shape[-2]
            self.num_sub_graphs = 1

        # Instantiate the discretised field graph from which we will draw sub-graphs from
        self.edge_idx, self.edge_disp, self.node_indices = generate_discretised_graph(self.data['X'][0,0], r, bc)

    def __len__(self) -> int:
        return self.N * (self.num_t_steps_per_sample - 1)

    
    def __getitem__(self, index) -> Any:        
        sim_idx   = index // (self.num_t_steps_per_sample - 1)
        frame_idx = index %  (self.num_t_steps_per_sample - 1)

        X_t = self.data['X'][sim_idx, frame_idx]                                         # (H, W)
        pde_params = [float(self.data[k][sim_idx]) for k in self.field_keys]

        # For a given discretised input sample, subsample (num_sub_graphs) subgraphs from our large graph and return them
        
        ## Call function to generate N random node indice samples from edge_idx

        ## Add PDE Param features and sample u(x,y) value from grid to node edge feature vectors

        ### return the N sampled indices
        return
    