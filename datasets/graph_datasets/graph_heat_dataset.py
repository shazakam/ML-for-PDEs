from typing import Any, Iterable
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
        self.edge_idx, self.edge_disp, self.node_spatial_locs = generate_discretised_graph(self.data['X'][0,0], r, bc)

    def __len__(self) -> int:
        return self.N * (self.num_t_steps_per_sample - 1)

    
    def __getitem__(self, index) -> Any:        
        sim_idx   = index // (self.num_t_steps_per_sample - 1)
        frame_idx = index %  (self.num_t_steps_per_sample - 1)

        X_t = self.data['X'][sim_idx, frame_idx]                                         # (H, W)
        pde_params = [float(self.data[k][sim_idx]) for k in self.field_keys]

        ## Call function to generate N random node indice samples from edge_idx
        subgraph_node_indices = self.get_subsample_graph_indices(node_spatial_locs = self.node_spatial_locs)

        ## Add PDE Param features and sample u(x,y) value from grid to node edge feature vectors

        ### return the N sampled indices
        return

    def get_subsample_graph_indices(self) -> Iterable:
        """
        Inputs
        -----
        node_spatial_locs (torch.Tensor) : node spatial locations of shape (N^2, 2)
        Returns a collection of graph node indices.
        """
        sub_graph_node_indices = []

        for _ in range(self.num_sub_graphs):
            sub_graph_size = int(torch.randint(1,self.max_sub_graph_size, (1,1)).squeeze()) # Random subgraph size
            random_node_subsample_idx = torch.randperm(sub_graph_size) # Random Graph node indices
            sub_graph_node_indices.append(random_node_subsample_idx.to(torch.long))

        return sub_graph_node_indices

    def create_edge_features(self, X_t : torch.Tensor, subgraph_node_indices : Iterable, pde_params : list) -> torch.Tensor:
        processed_subgraph_edges = []

        # NOTE:
        # We need both m(x) and m(y) where x is the source node. So what we can do is to essentially 
        # map onto our self.edge_idx the relevant m(x) -> m(y)
        # This can then be used with the mask to attach (m(x), m(y)) to the the edge feature values to be ingested by the model

        for subgraph_nodes_idx in subgraph_node_indices:
            # Get edge displacements features for subgraph
            mask = torch.isin(self.edge_idx[0, :], subgraph_nodes_idx)
            subgraph_node_edges_disp = self.edge_disp[mask, :]

            # Get spatial measurement features for node i.e. measurement for Node A at (x,y)
            node_spatial_locs = self.node_spatial_locs[subgraph_nodes_idx, :]
            node_spatial_measurements = X_t[node_spatial_locs[:, 0], node_spatial_locs[:, 1]]

            # Iterate over PDE Params and append those to nodes as well (need if statement to see if they are constant for a given sim or vary across space)
            for pde_param in pde_params:

            # Append measurement and pde param to edge_disp from source node
        return 
    