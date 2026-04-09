import lightning as L
import torch
import torch.nn.functional as F
class DDPM(L.LightningModule):
    
    def __init__(self, denoising_model : torch.nn.Module, noise_schedule : torch.Tensor):
        super().__init__()
        self.denoising_model = denoising_model
        self.noise_schedule = noise_schedule
        self.T = noise_schedule.shape[0]

    def forward(self, u_0 : torch.Tensor) -> torch.Tensor:
        # Take input to be conditioned on u_0
        # Generate random noise for time step T
        u_noise_curr = torch.randn_like(u_0)
        step = [i for i in range(self.T)]
        step = step[::-1]
        # Denoise T timesteps
        for t in step:
            
            if t > 1:
                z = torch.randn_like(u_0)
            else:
                z = torch.zeros_like(u_0)

            eps_pred = self.denoising_model(u_noise_curr)
            alpha_t = self.noise_schedule[t]
            sigma_t = (1-self.noise_schedule[t-1])/(1-alpha_t)
            u_noise_curr = (1/torch.sqrt(alpha_t))*(u_noise_curr - ((1 - alpha_t)/torch.sqrt((1 - alpha_t)))*eps_pred+z*sigma_t)

        return u_noise_curr
    
    def training_step(self, batch : torch.Tensor, batch_idx : int):
        # Takes batch input - param should contain pde specific coefficients
        u_0, target,  t, param  = batch

        # Get random noise epsilon
        eps = torch.randn_like(u_0)

        # Add Noise to target state
        u_noisy = torch.sqrt(self.noise_schedule[t])*u_0 + torch.sqrt(1 - self.noise_schedule[t])*eps

        # Concatenate with u_0 so we have [u_0, u_noised_T] along channel axis
        u_in = torch.cat([u_0, u_noisy], dim=1)

        # Predict Noise
        eps_pred = self.denoising_model(u_in, t)

        # Calculate loss and return
        output_loss = F.mse_loss(eps_pred, eps)

        return output_loss
    
    def configure_optimizers(self):
        return super().configure_optimizers()
    