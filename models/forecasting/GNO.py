import lightning as L
import torch 
import torch.nn as nn
import torch.nn.functional as F
from ..model_utils.nn_helpers.GNO_layer import GNOLayer
class GNO(L.LightningModule):

    def __init__(self, optimiser: str = 'adamw', 
                 learning_rate: float = 1e-4, 
                 num_node_input_features : int = 3,
                 num_edge_features : int = 5, 
                 num_latent_dim : int = 64, 
                 output_dim : int = 1,
                 num_gno_layers : int = 6,
                 kernel_ffn_layers : list[int] | None = None,
                 kernel_ffn_dropout : float = 0.001, 
                 GNO_layer_activation : str = 'relu'
                 ) -> None:
        
        super().__init__()
        self.save_hyperparameters()

        # Mutable defaults are shared across instances, so build the list here instead.
        if kernel_ffn_layers is None:
            kernel_ffn_layers = [num_edge_features, 128, num_latent_dim ** 2]

        if (kernel_ffn_layers[-1] != num_latent_dim**2) or (kernel_ffn_layers[0] != num_edge_features):
            raise ValueError('Not the correct final output dim or intitial input dim for integral kernel')
        
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.num_gno_layers = num_gno_layers
        self.kernel_ffn_layers = kernel_ffn_layers
        self.kernel_ffn_dropout = kernel_ffn_dropout
        self.GNO_layer_activation = GNO_layer_activation
        self.num_node_input_features = num_node_input_features

        self.P = nn.Linear(num_node_input_features, num_latent_dim, bias = True)
        self.Q = nn.Linear(num_latent_dim, output_dim, bias = True)
        self.GNO_layers = nn.ModuleList([GNOLayer(node_input_dim = num_latent_dim,
                                                   layer_sizes = kernel_ffn_layers, 
                                                   layer_activation_function = GNO_layer_activation,
                                                   dropout_rate=kernel_ffn_dropout) for _ in range(self.num_gno_layers)])


    def forward(self, batch):
        """
        Input
        -----
        batch (torch_geometric.data.Batch) : Batched subgraphs carrying x, edge_index and edge_attr

        Output
        ------
        torch.Tensor (shape: N x output_dim) : Predicted u(x, t+1) at every node in the batch
        """
        # The batch is never written to: mutating batch.x would leave the latent
        # representation in place of the node features and break any second forward pass.
        v_t = self.P(batch.x)
        for layer in self.GNO_layers:
            v_t = layer(v_t, batch.edge_index, batch.edge_attr)

        return self.Q(v_t)

    def training_step(self, batch, batch_idx : int):
        y_hat = self.forward(batch)                 # (N, output_dim)
        output_loss = F.mse_loss(y_hat, batch.y)    # batch.y is (N, 1), one target per node

        self.log("train_loss", output_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return output_loss

    def validation_step(self, batch, batch_idx : int):
        y_hat = self.forward(batch)
        output_loss = F.mse_loss(y_hat, batch.y)

        self.log("val_loss", output_loss, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return output_loss
    
    def configure_optimizers(self):
        if self.optimiser.lower() == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.optimiser.lower() == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        raise ValueError(f"Unsupported optimiser: '{self.optimiser}'. Choose 'adam' or 'adamw'.")   