import lightning as L
import torch
import torch.nn.functional as F

from models.unet.unet import UNet

class DDPM(L.LightningModule):
    
    def __init__(self, denoising_model: UNet, noise_schedule: torch.Tensor,
                 optimiser: str, learning_rate: float):
        super().__init__()
        self.denoising_model = denoising_model
        self.register_buffer("noise_schedule", noise_schedule)
        self.T = noise_schedule.shape[0] - 1  # schedule has T+1 points (t=0..T); steps are 1..T-1
        self.optimiser = optimiser
        self.learning_rate = learning_rate

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
            alpha_bar_t = self.noise_schedule[t]#type:ignore
            beta_t = 1 - self.noise_schedule[t] / self.noise_schedule[t-1]#type:ignore
            alpha_t = 1 - beta_t

            sigma_t = torch.sqrt(beta_t * (1 - self.noise_schedule[t-1]) / (1 - alpha_bar_t))#type:ignore

            u_noise_curr = (1/torch.sqrt(alpha_t))*(u_noise_curr - (beta_t/torch.sqrt((1 - alpha_bar_t)))*eps_pred+z*sigma_t)

        return u_noise_curr
    
    def training_step(self, batch : torch.Tensor, batch_idx : int):
        # u_0    : (B, C, H, W) — conditioning frame(s) + PDE param channels
        # target : (B, H, W)   — next PDE state to forecast
        # t      : (B,)        — diffusion timestep sampled by the dataset
        u_0, target, t = batch

        target = target.unsqueeze(1)                                    # (B, 1, H, W)
        eps = torch.randn_like(target)

        # Noise the target (not the conditioning)
        alpha_bar_t = self.noise_schedule[t].view(-1, 1, 1, 1)         # broadcast over (B, 1, H, W) #type:ignore
        target_noisy = torch.sqrt(alpha_bar_t) * target + torch.sqrt(1 - alpha_bar_t) * eps

        # Condition on u_0, denoise target: [u_0, target_noisy] -> predict eps
        u_in = torch.cat([u_0, target_noisy], dim=1)
        eps_pred = self.denoising_model(u_in, t)

        output_loss = F.mse_loss(eps_pred, eps)
        self.log("train_loss", output_loss, on_step=True, on_epoch=True, prog_bar=True)
        return output_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        u_0, target, t = batch

        target = target.unsqueeze(1)
        eps = torch.randn_like(target)

        alpha_bar_t = self.noise_schedule[t].view(-1, 1, 1, 1) #type:ignore
        target_noisy = torch.sqrt(alpha_bar_t) * target + torch.sqrt(1 - alpha_bar_t) * eps

        u_in = torch.cat([u_0, target_noisy], dim=1)
        eps_pred = self.denoising_model(u_in, t)

        val_loss = F.mse_loss(eps_pred, eps)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        if self.optimiser.lower() == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.optimiser.lower() == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        raise ValueError(f"Unsupported optimiser: '{self.optimiser}'. Choose 'adam' or 'adamw'.")
    