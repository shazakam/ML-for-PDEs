import yaml
from ...models.unet.unet import UNet
from ...models.forecasting.deep_onet import DeepONet
import torch

def deep_onet_forecast(model_path : str, model_cfg, num_steps : int, device : torch.Device, X : torch.Tensor) -> torch.Tensor:
    cfg = model_cfg

    # --- Model ---
    model = DeepONet(
        conv_branch_layers=cfg.conv_branch_layers,
        conv_branch_activations=cfg.conv_branch_activations,
        stride_branch=cfg.stride_branch,
        ffn_branch_layers=cfg.ffn_branch_layers,
        ffn_branch_activations=cfg.ffn_branch_activations,
        ffn_trunk_layers=cfg.ffn_trunk_layers,
        ffn_trunk_activations=cfg.ffn_trunk_activations,
        dropout=cfg.dropout,
        optimiser=cfg.optimiser,
        learning_rate=cfg.learning_rate,
    )


    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.to(device)
    model.eval()

    multi_step_output = []
    for t in range(1, num_steps):
        dt = t / num_steps
        with torch.no_grad():
            output = deep_onet_single_image_inference(model, X.to(device).unsqueeze(0), dt)
            multi_step_output.append(output.cpu())
            X[0] = output.squeeze().cpu()

    multi_step_output = torch.concat(multi_step_output)
    return multi_step_output

def deep_onet_single_image_inference(model, X : torch.Tensor, t : float) -> torch.Tensor:
    H = X.shape[-2]
    W = X.shape[-1]
    final_output = []
    for h in range(H):
        row_output = []
        for w in range(W):
            trunk   = torch.tensor([w / (W - 1), h / (H - 1), t], dtype=torch.float32)
            pix_output = model.forward(X, trunk)
            row_output.append(pix_output)

        final_output.append(row_output)

    return torch.tensor(final_output)
