import lightning as L
import torch
import torch.nn.functional as F

from models.unet.unet import UNet

class DDPM(L.LightningModule):
    
    def __init__(self, denoising_model: UNet, noise_schedule: torch.Tensor):
        super().__init__()
        self.denoising_model = denoising_model
        self.noise_schedule = noise_schedule
        self.T = noise_schedule.shape[0] - 1  # schedule has T+1 points (t=0..T); steps are 1..T-1

    def forward(self, u_0 : torch.Tensor) -> torch.Tensor:
        # Take input to be conditioned on u_0
        # Generate random noise for time step T
        out_ch = self.denoising_model.final_filters
        u_noise_curr = torch.randn(u_0.shape[0], out_ch, u_0.shape[2], u_0.shape[3], device=self.device)
        step = [i for i in range(1, self.T)]
        step = step[::-1]

        # Denoise T timesteps
        for t in step:
            if t > 1:
                z = torch.randn_like(u_noise_curr)
            else:
                z = torch.zeros_like(u_noise_curr)

            u_in = torch.cat([u_0, u_noise_curr], dim=1)

            T = torch.full((u_in.shape[0],), t).to(self.device)
            eps_pred = self.denoising_model(u_in, T)
            alpha_bar_t = self.noise_schedule[t]
            beta_t = 1 - self.noise_schedule[t] / self.noise_schedule[t-1]
            alpha_t = 1 - beta_t

            sigma_t = torch.sqrt(beta_t * (1 - self.noise_schedule[t-1]) / (1 - alpha_bar_t))

            u_noise_curr = (1/torch.sqrt(alpha_t))*(u_noise_curr - (beta_t/torch.sqrt((1 - alpha_bar_t)))*eps_pred+z*sigma_t)

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
    