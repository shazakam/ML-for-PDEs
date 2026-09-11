from typing import Any
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from .graph_data_utils import create_graph

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
        self.node_grid_indices = torch.cartesian_prod(x_indices, y_indices) # Collection of (x, y) grid indices
        self.node_spatial_pos = torch.cartesian_prod(x_indices / W, y_indices / H) # Collection of (x, y) spatial positions

        if sub_graph_size <= 0 or sub_graph_size > self.node_grid_indices.shape[0]:
            raise ValueError(
                f"sub_graph_size must be in [1, {self.node_grid_indices.shape[0]}] for this discretisation, got {sub_graph_size}"
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
        random_node_grid_list_indices = torch.randperm(self.node_grid_indices.shape[0])[:self.sub_graph_size].sort().values.to(torch.long) # NOTE: This can be made better

        subgraph_node_indices = self.node_grid_indices[random_node_grid_list_indices] # Subgraph Node index locations on total domain (x idx, y idx)!
        subgraph_spatial_locs = self.node_spatial_pos[random_node_grid_list_indices] # Subgraph Node distance values on total domain (x , y) positions!
        
        subgraph_edge_index, subgraph_edge_disp = create_graph(node_pos = subgraph_spatial_locs, r = self.radius, boundary_condition = self.boundary_conditions)

        ## Add PDE Param features and sample u(x,y) value from grid to node edge feature vectors
        sample_edge_feature_inputs_x = self.create_edge_features(X_t, subgraph_node_indices, subgraph_edge_index, subgraph_edge_disp, pde_params)

        y_hat = X_t1[subgraph_node_indices[:, 0], subgraph_node_indices[:, 1]].unsqueeze(-1)
        
        # num_nodes is declared explicitly: inferring it from edge_index would under-count
        # whenever a sampled node ends up isolated, silently misaligning y under batching.
        return Data(edge_index = subgraph_edge_index,
                    edge_attr = sample_edge_feature_inputs_x,
                    y = y_hat,
                    num_nodes = self.sub_graph_size)
    
    def create_edge_features(self, X_t : torch.Tensor, 
                             subgraph_node_idx_locs : torch.Tensor,
                             subgraph_edge_index : torch.Tensor, 
                             subgraph_edge_disp : torch.Tensor, 
                             pde_params : list) -> torch.Tensor:

        """
        Inputs
        -------
        X_t (torch.Tensor, shape: H x W) : Input sample to create subgraph edge features for
        subgraph_node_idx_locs (torch.Tensor, shape: m x 2) : (x, y) grid indices of the sampled subgraph nodes
        subgraph_edge_index (torch.Tensor, shape: 2 x E) : Local edge index over the m sampled nodes
        subgraph_edge_disp (torch.Tensor, shape: E x 2) : Minimum-image displacement for each edge
        pde_params (list) : List containing PDE Params for given sample, currently only works with a constant PDE coefficient for a given sample i.e. non-evolving over time

        Outputs
        -------
        subgraph_edge_features (torch.Tensor, shape : E x (4 + num pde params)) : Edge features laid out as
        [disp_x, disp_y, u_src, u_dst, *pde_params] for a subgraph of the sample X_t
        """

        # Retrieve spatial measurements at source node locations and edge nodes
        node_spatial_measurements = X_t[subgraph_node_idx_locs[:, 0], subgraph_node_idx_locs[:, 1]] # Subgraph node measurements in domain at t

        # Get spatial measurement for source node and get spatial measurement for destination node and concatenate them
        edge_spatial_measurements = torch.concat([node_spatial_measurements[subgraph_edge_index[0, :]].unsqueeze(-1), node_spatial_measurements[subgraph_edge_index[1, :]].unsqueeze(-1)], dim = -1) # E x 2

        # Iterate over PDE Params and append those to nodes as well (currently assumes constant parameter)
        pde_tensors = torch.concatenate([torch.full((edge_spatial_measurements.shape[0], 1), pde_param) for pde_param in pde_params], dim = -1)

        # (E, 4 + however many pde params for the equation)
        edge_feature_inputs = torch.concatenate([subgraph_edge_disp, edge_spatial_measurements, pde_tensors], dim = -1)

        return edge_feature_inputs # (E, 4 + however many pde params for the equation)


    