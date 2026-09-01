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
        subgraph_node_indices = self.get_subsample_graph_indices()

        ## Add PDE Param features and sample u(x,y) value from grid to node edge feature vectors
        sample_edge_feature_inputs = self.create_edge_features(X_t, subgraph_node_indices, pde_params)

        return sample_edge_feature_inputs

    def get_subsample_graph_indices(self) -> Iterable:
        """
        Outputs
        ------
        sub_graph_node_indices (list[torch.Tensor]) : Iterable containing random collection of sub graph indices to be queried for a given sample
        """
        sub_graph_node_indices = []

        for _ in range(self.num_sub_graphs):
            sub_graph_size = int(torch.randint(1,self.max_sub_graph_size, (1,1)).squeeze()) # Random subgraph size
            random_node_subsample_idx = torch.randperm(sub_graph_size) # Random Graph node indices
            sub_graph_node_indices.append(random_node_subsample_idx.to(torch.long))

        return sub_graph_node_indices

    def create_edge_features(self, X_t : torch.Tensor, subgraph_source_node_indices : Iterable, pde_params : list) -> torch.Tensor:

        """
        Inputs
        -------
        X_t (torch.Tensor, shape: H x W) : Input sample to create subgraph(s) edge features for 
        subgraph_source_node_indices (Iterable) : Iterable containing source node indices for each subgraph
        pde_params (list) : List containing PDE Params for given sample, currently only works with a constant PDE coefficient for a given sample i.e. non-evolving over time

        Outputs
        -------
        subgraph_edge_features (torch.Tensor, shape : number of subgraphs x E x num edge features) : Edge feature inputs for each subgraph for the sample X_t
        """
        processed_subgraph_edges = []

        for subgraph_nodes_idx in subgraph_source_node_indices:

            source_node_mask = torch.isin(self.edge_idx[0, :], subgraph_nodes_idx) # Here we query what the indices are for the edge A -> B
            source_node_dest_node_edge_indices = self.edge_idx[: , source_node_mask] # Indices for source nodes and for edge destination nodes (2, E)
            subgraph_node_edges_disp = self.edge_disp[source_node_mask, :] # Using the indices we get the displacement from node A to node B as (E, 2)

            # Get spatial measurement features for source node i.e. measurement for Node A at (x,y)
            source_node_spatial_locs = self.node_spatial_locs[source_node_dest_node_edge_indices[0, :], :] # (E, 2) containing (x, y) source node coordinates
            source_node_spatial_measurements = X_t[source_node_spatial_locs[:, 0], source_node_spatial_locs[:, 1]] # This is of shape [E, 1]

            # Get spatial measurements for destination nodes
            dest_node_spatial_locs = self.node_spatial_locs[source_node_dest_node_edge_indices[1, :], :] # (E, 2) containing (x, y) dest node coordinates
            dest_node_spatial_measurements = X_t[dest_node_spatial_locs[:, 0], dest_node_spatial_locs[:, 1]] # This is of shape [E, 1]

            # Iterate over PDE Params and append those to nodes as well (currently assumes constant parameter)
            pde_tensors = torch.concatenate([torch.full(dest_node_spatial_measurements.shape, pde_param) for pde_param in pde_params], dim = -1)

            # This should in theory be of shape (E, 4 + however many pde params for the equation)
            edge_feature_inputs = torch.concatenate([subgraph_node_edges_disp, source_node_spatial_measurements, dest_node_spatial_measurements, pde_tensors], dim = -1)

            processed_subgraph_edges.append(edge_feature_inputs)
        return torch.stack(processed_subgraph_edges, dim = 0) # (num subgraphs for sample, E, 4 + however many pde params for the equation)
    