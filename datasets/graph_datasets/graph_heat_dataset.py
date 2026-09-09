from typing import Any, Iterable
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from .graph_data_utils import generate_discretised_graph

# This Graph Dataset assumes only a single discretisation is used

# TODO: NEED TO CHANGE THIS COMPLETELY
# Misunderstood paper in a couple of key areas when it came to A) Subsampling graphs and B) SubGraph Construction
# Instead of instantiating one massive graph from which we sample nodes and generate edges from. We just sample m points and make a subgraph out of that
# It scales better and might actually be easier as well
# What we need to do instead:
# 1. In __init__ we just instantiate node_index_locs -> List of (x,y) coords for X and normalised coordinates node_spatial_locs -> List of (x_spat, y_spat) for distance calc
# 2. For get_item, select random node locations from node_spatial_locs and create periodic_radius_graph with the selected node locations
# 3. Then simplify create_edge_features (do not need to concern ourselves with destination node edges anylonger)

class HeatGraphDataset(Dataset):
    def __init__(self, aggregated_path: str, field_keys: list[str], r : float, bc : str, sub_graph_size : int) -> None:
        super().__init__()
        self.field_keys = field_keys
        self.data = torch.load(aggregated_path, weights_only=False, mmap=True)
        self.N, self.num_t_steps_per_sample = self.data['X'].shape[:2]  # (N, T, H, W)
        self.sub_graph_size = sub_graph_size

        # Instantiate the discretised field graph from which we will draw sub-graphs from
        self.edge_idx, self.edge_disp, self.node_spatial_locs = generate_discretised_graph(self.data['X'][0,0], r, bc)

        self.num_nodes = self.node_spatial_locs.shape[0]

        if sub_graph_size <= 0 or sub_graph_size > self.num_nodes:
            raise ValueError(
                f"sub_graph_size must be in [1, {self.num_nodes}] for this discretisation, got {sub_graph_size}"
            )

    def __len__(self) -> int:
        return self.N * (self.num_t_steps_per_sample - 1)

    
    def __getitem__(self, index) -> Any:   
        """
        Input
        ------
        index (int) : Sample index to retrieve
        """     
        sim_idx   = index // (self.num_t_steps_per_sample - 1)
        frame_idx = index %  (self.num_t_steps_per_sample - 1)

        X_t = self.data['X'][sim_idx, frame_idx] # (H, W)
        X_t1 = self.data['X'][sim_idx, frame_idx+1]
        pde_params = [float(self.data[k][sim_idx]) for k in self.field_keys]

        ## Uniformly sample sub_graph_size nodes (without replacement) from the full grid graph.
        ## Sorted so the selected edges come back in global edge_idx order.
        subgraph_node_indices = torch.randperm(self.num_nodes)[:self.sub_graph_size].sort().values.to(torch.long)

        ## Add PDE Param features and sample u(x,y) value from grid to node edge feature vectors
        sample_edge_feature_inputs_x = self.create_edge_features(X_t, subgraph_node_indices, pde_params)

        y_hat = self.get_subgraph_labels(X_t1, subgraph_node_indices)

        return Data(edge_attr=sample_edge_feature_inputs_x, y=y_hat), sample_edge_feature_inputs_x, y_hat

    def create_edge_features(self, X_t : torch.Tensor, subgraph_source_node_indices : torch.Tensor, pde_params : list) -> torch.Tensor:

        """
        Inputs
        -------
        X_t (torch.Tensor, shape: H x W) : Input sample to create subgraph(s) edge features for 
        subgraph_source_node_indices (Iterable) : Iterable containing source node indices for each subgraph
        pde_params (list) : List containing PDE Params for given sample, currently only works with a constant PDE coefficient for a given sample i.e. non-evolving over time

        Outputs
        -------
        subgraph_edge_features (torch.Tensor, shape : E x num edge features) : Edge feature inputs for a subgraph for the sample X_t
        """

        source_node_mask = torch.isin(self.edge_idx[0, :], subgraph_source_node_indices) # Here we query what the indices are for the edge A -> B

        # Retrieve Source to Edge node distance difference
        subgraph_node_edges_disp = self.edge_disp[source_node_mask, :]

        # Retrieve spatial measurements at source node locations and edge nodes
        source_node_spatial_measurements, dest_node_spatial_measurements = self.get_source_node_and_edge_node_spatial_measurements(X_t, source_node_mask)

        # Iterate over PDE Params and append those to nodes as well (currently assumes constant parameter)
        pde_tensors = torch.concatenate([torch.full(dest_node_spatial_measurements.shape, pde_param) for pde_param in pde_params], dim = -1)

        # This should in theory be of shape (E, 4 + however many pde params for the equation)
        edge_feature_inputs = torch.concatenate([subgraph_node_edges_disp, source_node_spatial_measurements, dest_node_spatial_measurements, pde_tensors], dim = -1)

        return edge_feature_inputs # (E, 4 + however many pde params for the equation)

    def get_source_node_and_edge_node_spatial_measurements(self, X_t, source_node_mask) -> Iterable[torch.Tensor]:
        source_node_dest_node_edge_indices = self.edge_idx[: , source_node_mask] # Indices for source nodes and for edge destination nodes (2, E)

        # Get spatial measurement features for source node i.e. measurement for Node A at (x,y)
        source_node_spatial_locs = self.node_spatial_locs[source_node_dest_node_edge_indices[0, :], :] # (E, 2) containing (x, y) source node coordinates
        source_node_spatial_measurements = X_t[source_node_spatial_locs[:, 0], source_node_spatial_locs[:, 1]].unsqueeze(-1) # This is of shape [E, 1]

        # Get spatial measurements for destination nodes
        dest_node_spatial_locs = self.node_spatial_locs[source_node_dest_node_edge_indices[1, :], :] # (E, 2) containing (x, y) dest node coordinates
        dest_node_spatial_measurements = X_t[dest_node_spatial_locs[:, 0], dest_node_spatial_locs[:, 1]].unsqueeze(-1) # This is of shape [E, 1]

        return source_node_spatial_measurements, dest_node_spatial_measurements

    def get_subgraph_labels(self, X_t1 : torch.Tensor, subgraph_node_indices : torch.Tensor) -> torch.Tensor:
        """
        Inputs
        -------
        X_t1 (torch.Tensor, shape: H x W) : Next-frame field to read targets from
        subgraph_node_indices (torch.Tensor, shape: m) : Global node indices making up the subgraph

        Outputs
        -------
        source_measurement_y (torch.Tensor, shape : m x 1) : u(x, y) at t+1 for each subgraph node
        """
        node_xy_coords = self.node_spatial_locs[subgraph_node_indices] # (m, 2)
        source_measurement_y = X_t1[node_xy_coords[:, 0], node_xy_coords[:, 1]].unsqueeze(-1)
        return source_measurement_y


    