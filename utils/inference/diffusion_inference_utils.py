import torch

from models.unet.unet import UNet
from models.forecasting.diffusion import DDPM
from models.model_utils.noise_scheduler import CosineScheduler


def diffusion_forecast(model_path: str, model_cfg: dict, num_steps: int,
                       device: torch.device, X: torch.Tensor) -> torch.Tensor:
    diffusion_cfg = model_cfg

    # --- Model ---
    unet = UNet(
        in_channels=diffusion_cfg['in_channels'],
        out_channels=diffusion_cfg['out_channels'],
        kernel_size=diffusion_cfg['kernel_size'],
        final_filters=diffusion_cfg['final_filters'],
        encoder_dropout=diffusion_cfg['encoder_dropout'],
        input_size=diffusion_cfg['input_size'],
    )

    # noise_schedule is a registered buffer on DDPM and is restored by load_state_dict,
    # but DDPM still requires one at construction time.
    noise_schedule = CosineScheduler(T=diffusion_cfg['num_timesteps'], s=diffusion_cfg['cosine_shift']).schedule()

    diffusion_model = DDPM(
        denoising_model=unet,
        noise_schedule=noise_schedule,
        optimiser=diffusion_cfg['optimiser'],
        learning_rate=diffusion_cfg['learning_rate']
    )

    diffusion_model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    diffusion_model.to(device)
    diffusion_model.eval()

    # Clone so the caller's tensor is not mutated by the in-place feedback below.
    X = X.clone()
    multi_step_output = []
    for _ in range(num_steps):
        with torch.no_grad():
            output = diffusion_model(X.to(device).unsqueeze(0))   # (1, 1, H, W)
            multi_step_output.append(output.cpu())
            X[0] = output.squeeze().cpu()                          # feed prediction back as new state

    return torch.concat(multi_step_output)                        # (num_steps, 1, H, W)
