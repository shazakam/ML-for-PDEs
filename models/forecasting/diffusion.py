import lightning as L
import torch
class DDPM(L.LightningModule):
    
    def __init__(self, denoising_model : torch.nn.Module):
        super().__init__()
        self.denoising_model = denoising_model

    def forward(self, u_0 : torch.Tensor) -> torch.Tensor:
        # Take input to be conditioned on u_0
        # Generate random noise for time step T

        # Denoise T timesteps

        # Return output
        return 
    
    def training_step(self, u_run : torch.Tensor):
        # Takes batch input
        # Get random t between 0 and T
        # Get random noise epsilon
        # Add Noise to target state
        # Concatenate with u_0 so we have [u_0, u_noised_T]

        # Predict Noise
        # Calculate loss and return
        return super().training_step()
    
    def configure_optimizers(self):
        return super().configure_optimizers()
    