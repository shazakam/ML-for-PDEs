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
                 kernel_ffn_layers : list[int] = [5, 512, 1024, 4096],
                 kernel_ffn_dropout : float = 0.001, 
                 GNO_layer_activation : str = 'relu'
                 ) -> None:
        
        super().__init__()

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
        self.GNO_layers = nn.Sequential(*[GNOLayer(node_input_dim = num_latent_dim,
                                                   layer_sizes = kernel_ffn_layers, 
                                                   layer_activation_function = GNO_layer_activation,
                                                   dropout_rate=kernel_ffn_dropout) for _ in range(self.num_gno_layers)])


    def forward(self, x): 
        v_t = self.P(x.x)
        x.x = v_t
        v_t = self.GNO_layers(x)
        v_t = self.Q(v_t.x)
       
        return v_t
    
    def training_step(self, batch, batch_idx):

        y = batch.y                                 # (B, 1, H, W)
        y_hat = self.forward(batch)

        output_loss = F.mse_loss(y_hat,y)
        self.log("train_loss", output_loss, on_step=True, on_epoch=True, prog_bar=True)
        return output_loss
    
    def validation_step(self, batch : torch.Tensor, batch_idx : int):
        u_0, target  = batch

        target = target.unsqueeze(1)                                    # (B, 1, H, W)
        u_next = self.forward(u_0)

        output_loss = F.mse_loss(u_next,target)
        self.log("val_loss", output_loss, on_epoch=True, prog_bar=True)

        return output_loss
    
    def configure_optimizers(self):
        if self.optimiser.lower() == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.optimiser.lower() == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        raise ValueError(f"Unsupported optimiser: '{self.optimiser}'. Choose 'adam' or 'adamw'.")   