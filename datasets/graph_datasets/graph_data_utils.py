
import torch

def generate_discretised_graph(X : torch.Tensor, r : float, bc : str):
    """
    Params
    ------
    X (torch.Tensor) : Input tensor (really only need the shape anyways)
    r (float) : r is [0, 1] where 1 would meant a fully connected graph and 0 would mean every node is disconnected
    bc (str) : Boundary condition (for now only periodic)

    Outputs
    -------
    Discretised Graph 
    
    """
    H, W = X.shape[-1], X.shape[-2]
    x_y = torch.tensor([x for x in range(H)])
    node_indices = torch.cartesian_prod(x_y, x_y)
    pos_nodes = torch.cartesian_prod(x_y / H, x_y / H)

    if bc == 'periodic':
        edge_index, edge_disp = periodic_radius_graph(pos_nodes, r)

    else: 
        raise ValueError('Bombaclat not correct bc')

    return edge_index, edge_disp, node_indices
        


def periodic_radius_graph(pos: torch.Tensor, r: float, box: float = 1.0):
    """
    pos : [N, 2] coordinates in [0, box)
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