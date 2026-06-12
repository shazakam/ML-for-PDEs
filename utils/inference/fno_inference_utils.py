import yaml
from ...models.unet.unet import UNet
from ...models.forecasting.FNO import FNO
from ...models.model_utils.nn_helpers.ffn import FFN
import torch

def fno_forecast(model_path : str, model_cfg, num_steps : int, device : torch.Device, X : torch.Tensor) -> torch.Tensor:
    cfg = model_cfg

    # --- Model ---
    P_FFN = FFN(layer_sizes=cfg.num_p_layers, activation=cfg.p_activations)
    Q_FFN = FFN(layer_sizes=cfg.num_q_layers, activation=cfg.q_activations)

    model = FNO(d_v=cfg.projection_dim, 
                num_fourier_layers=cfg.num_fourier_layers,
                num_fourier_modes=cfg.num_fourier_modes,
                P=P_FFN,
                Q=Q_FFN,
                optimiser=cfg.optimiser,
                learning_rate=cfg.learning_rate)


    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.to(device)
    model.eval()

    multi_step_output = []
    for t in range(num_steps):
        with torch.no_grad():
            output = model(X.to(device).unsqueeze(0))
            multi_step_output.append(output.cpu())
            X[0] = output.squeeze().cpu()

    multi_step_output = torch.concat(multi_step_output)
    return multi_step_output