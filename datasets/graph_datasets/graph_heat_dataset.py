from typing import Any, Iterable
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from .graph_data_utils import generate_discretised_graph, create_graph

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

        # Graph construction requirements
        self.sub_graph_size = sub_graph_size
        self.radius = r
        self.boundary_conditions = bc

        H, W = self.data['X'][0,0].shape[-1], self.data['X'][0,0].shape[-2] 
        x_indices = torch.tensor([x for x in range(W)])
        y_indices = torch.tensor([x for x in range(H)])
        self.node_spatial_indices_list = torch.cartesian_prod(x_indices, y_indices) # List of (x, y) grid indices
        self.pos_nodes = torch.cartesian_prod(x_indices / W, y_indices / H) # List of (x, y) spatial positions

        if sub_graph_size <= 0 or sub_graph_size > self.node_spatial_indices.shape[0]:
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
        random_node_indices = torch.randperm(self.num_nodes)[:self.sub_graph_size].sort().values.to(torch.long) # NOTE: This can be made better
        subgraph_node_indices = self.node_spatial_indices_list[random_node_indices] # Subgraph Node index locations on total domain (x idx, y idx)!
        subgraph_spatial_locs = self.pos_nodes[subgraph_node_indices] # Subgraph Node distance values on total domain (x , y ) positions!

        subgraph_edge_index, subgraph_edge_disp = create_graph(node_pos = subgraph_spatial_locs, r = self.radius, boundary_condition = self.boundary_conditions)

        ## Add PDE Param features and sample u(x,y) value from grid to node edge feature vectors
        sample_edge_feature_inputs_x = self.create_edge_features(X_t, subgraph_node_indices, subgraph_edge_disp, pde_params)

        y_hat = X_t1[subgraph_node_indices[:, 0], subgraph_node_indices[:, 1]]

        return Data(edge_idx = subgraph_edge_index.T, edge_attr=sample_edge_feature_inputs_x, y=y_hat), subgraph_edge_index, sample_edge_feature_inputs_x, y_hat

    def create_edge_features(self, X_t : torch.Tensor, subgraph_node_idx_locs : torch.Tensor, subgraph_edge_index : torch.Tensor, subgraph_edge_disp, pde_params : list) -> torch.Tensor:

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

        # Retrieve spatial measurements at source node locations and edge nodes
        node_spatial_measurements = X_t[subgraph_node_idx_locs[:, 0], subgraph_node_idx_locs[:, 1]] # Subgraph node measurements in domain at t

        # Get spatial measurement for source node and get spatial measurement for destination node and concatenate them
        edge_spatial_measurements = torch.concat([node_spatial_measurements[subgraph_edge_index[:, 0]].unsqueeze(-1), node_spatial_measurements[subgraph_edge_index[:, 1]].unsqueeze(-1)], dim = -1) # E x 2

        # Iterate over PDE Params and append those to nodes as well (currently assumes constant parameter)
        pde_tensors = torch.concatenate([torch.full(edge_spatial_measurements.shape, pde_param) for pde_param in pde_params], dim = -1)

        # This should in theory be of shape (E, 3 + however many pde params for the equation)
        edge_feature_inputs = torch.concatenate([subgraph_edge_disp, edge_spatial_measurements, pde_tensors], dim = -1)

        return edge_feature_inputs # (E, 3 + however many pde params for the equation)

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


    