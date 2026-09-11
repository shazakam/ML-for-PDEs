import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from models.model_utils.nn_helpers.ffn import FFN
from ..activations import ACTIVATIONS

class GNOLayer(MessagePassing):
    """
    Graph Neural Operator Layer
    node_feature_dim (int) : Latent space dimension size of projected node features
    layer_activation_function (str) : String value mapping to relevant activation function in ACTIVATIONS
    layer_sizes (list[int]) : Contains layer size for a fully connected feed forward network. First value is input size, last value is output size
    dropout_rate (float | list[float]) : Dropout rates for either all or individual layers
    """
    def __init__(self, node_feature_dim : int, layer_activation_function : str, layer_sizes: list[int], dropout_rate: float | list[float] = 0) -> None:
        super().__init__(aggr = 'mean')
        self.node_feature_dim = node_feature_dim
        self.integral_kernel = FFN(layer_sizes=layer_sizes, activation='relu', dropout_rate=dropout_rate) # Needs to map to node_feature_dim x node_feature_dim
        self.W = nn.Linear(in_features=node_feature_dim, out_features=node_feature_dim)
        if layer_activation_function not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{layer_activation_function}', expected one of {sorted(ACTIVATIONS)}")
        self.activation = ACTIVATIONS[layer_activation_function]()

    def forward(self, v_t : torch.Tensor, edge_index : torch.Tensor, edge_attr : torch.Tensor):
        """
        Input
        -----
        v_t (torch.Tensor) : Node features in latent dimension
        edge_index (torch.Tensor) : Edge connections for subgraph
        edgde_attr (torch.Tensor) : Edge features passed to integral kernel

        Output
        ------
        Torch.Tensor : Final output are the transformed graph node features using Monte Carlo Approximation
        """
        out = self.propagate(edge_index, x=v_t, edge_attr = edge_attr)
        return self.activation(self.W(v_t) + out)

    def message(self, x_j: torch.Tensor, edge_attr : torch.Tensor) -> torch.Tensor:
        k = self.integral_kernel(edge_attr).reshape(-1, self.node_feature_dim, self.node_feature_dim)
        out = torch.einsum('eij,ej -> ei', k, x_j) 
        return out

