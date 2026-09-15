
from typing import Iterable
from torch_geometric.data import Data
import torch
import math

def create_graph(node_pos : torch.Tensor, r : float, boundary_condition : str, box : float = 1.0):
    if boundary_condition == 'periodic':
        edge_index, edge_disp = periodic_radius_graph(node_pos, r, box)
    else:
        raise ValueError("invalid boundary condition input")

    return edge_index, edge_disp

def periodic_radius_graph(pos: torch.Tensor, r: float, box: float = 1.0):
    """
    Input
    -----
    pos (torch.Tensor): [N, 2] coordinates in [0, box)
    r (float) : radius for creating edges across nodes
    box (float) : size of the domain (usually just normalised to 1.0)
    returns edge_index [2, E] and periodic displacement [E, 2]
    """

    d = pos.unsqueeze(1) - pos.unsqueeze(0)        # [N, N, 2]
    d = d - box * torch.round(d / box)             # minimum-image wrap
    dist = d.norm(dim=-1)                          # [N, N]
    mask = (dist <= r) & (dist > 0)                # drop self-loops
    edge_index = mask.nonzero().t().contiguous()   # [2, E]
    src, dst = edge_index
    edge_disp = d[src, dst]                        # [E, 2] -> your edge_attr geometry
    return edge_index, edge_disp

def create_edge_features(X_t : torch.Tensor, 
                            subgraph_node_idx_locs : torch.Tensor,
                            subgraph_edge_index : torch.Tensor, 
                            subgraph_edge_disp : torch.Tensor, 
                            pde_params : list) -> tuple[torch.Tensor, torch.Tensor]:

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

        return edge_feature_inputs, node_spatial_measurements # (E, 4 + however many pde params for the equation)



def partition_domain_into_subgraphs(X_t : torch.Tensor,
                                    X_t1: torch.Tensor,
                                    H : int, 
                                    W : int,
                                    num_subgraph_nodes : int, 
                                    r : float, 
                                    boundary_condition : str,
                                    pde_params : list[float]):
    
    x_indices = torch.tensor([x for x in range(W)])
    y_indices = torch.tensor([x for x in range(H)])

    subgraphs = []
    if num_subgraph_nodes > H*W:
        raise ValueError("Too large subgraph nodes for grid")

    elif num_subgraph_nodes == H*W:
        subgraph = create_graph_data_obj(x_indices, y_indices, H, W, r, boundary_condition, X_t, X_t1, pde_params)
        subgraphs.append(subgraph)

    else:
        cur_loc_i, cur_loc_j = 0,0
        H_slide, W_slide = get_subgraph_grid_size(num_subgraph_nodes)
        num_vertical_slices, num_horizontal_slices = (H // H_slide + 1), (W // W_slide + 1)

        for _ in range(num_vertical_slices):

            if cur_loc_i + H_slide < H:
                x_subgraph_indices = x_indices[cur_loc_i : cur_loc_i + H_slide]
            else:
                x_subgraph_indices = x_indices[-H_slide: ]

            for _ in range(num_horizontal_slices):
                if cur_loc_j + W_slide < W: 
                    y_subgraph_indices = y_indices[cur_loc_j : cur_loc_i + W_slide]
                else:
                    y_subgraph_indices = y_indices[-W_slide:]

                subgraph = create_graph_data_obj(x_subgraph_indices, y_subgraph_indices, H, W, r, boundary_condition, X_t, X_t1, pde_params)
                subgraphs.append(subgraph)

                cur_loc_j += W_slide
            cur_loc_i += H_slide
 
    return subgraphs

def create_graph_data_obj(x_indices, y_indices, H, W, r, boundary_condition, X_t, X_t1, pde_params):
    node_grid_indices = torch.cartesian_prod(x_indices, y_indices) # Collection of (x, y) grid indices
    node_spatial_pos = torch.cartesian_prod(x_indices / W, y_indices / H) 
    subgraph_edge_index, subgraph_edge_disp = create_graph(node_spatial_pos, r, boundary_condition)
    sample_edge_feature_inputs_x, subgraph_node_measurements = create_edge_features(X_t, node_grid_indices,
                                                                                    subgraph_edge_index,
                                                                                    subgraph_edge_disp,
                                                                                    pde_params)

    y = X_t1[node_grid_indices[:, 0], node_grid_indices[:, 1]].unsqueeze(-1)

    return Data(x=torch.concatenate([node_spatial_pos, subgraph_node_measurements.unsqueeze(-1)], dim = -1),
                edge_index = subgraph_edge_index,
                edge_attr = sample_edge_feature_inputs_x,
                y = y,
                num_nodes = node_grid_indices.shape[0])

def get_subgraph_grid_size(num_subgraph_nodes : int):
    sqrt_val = math.sqrt(num_subgraph_nodes)

    if sqrt_val.is_integer():
        return int(sqrt_val), int(sqrt_val)

    else:
        cur_val = int(sqrt_val) + 1
        cur_best_score = -1
        cur_best_pair = [-1,-1]
        done = False

        while not done:
            if cur_val > (num_subgraph_nodes // 2):
                done = True
                break

            if (num_subgraph_nodes % cur_val == 0) and (cur_val <= num_subgraph_nodes // 2):
                summed_factors = (num_subgraph_nodes // cur_val) + cur_val
                if summed_factors < cur_best_score or cur_best_score == -1:
                    cur_best_score = summed_factors
                    cur_best_pair[0] = num_subgraph_nodes // cur_val
                    cur_best_pair[1] = cur_val
                    cur_val += 1

            else:
                cur_val += 1

        return cur_best_pair[0], cur_best_pair[1]