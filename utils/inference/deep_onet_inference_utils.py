import torch

from models.forecasting.deep_onet import DeepONet


def deep_onet_forecast(model_path: str, model_cfg: dict, num_steps: int,
                       device: torch.device, X: torch.Tensor) -> torch.Tensor:
    cfg = model_cfg

    # --- Model ---
    model = DeepONet(
        conv_branch_layers=cfg['conv_branch_layers'],
        conv_branch_activations=cfg['conv_branch_activations'],
        stride_branch=cfg['stride_branch'],
        ffn_branch_layers=cfg['ffn_branch_layers'],
        ffn_branch_activations=cfg['ffn_branch_activations'],
        ffn_trunk_layers=cfg['ffn_trunk_layers'],
        ffn_trunk_activations=cfg['ffn_trunk_activations'],
        dropout=cfg['dropout'],
        optimiser=cfg['optimiser'],
        learning_rate=cfg['learning_rate'],
    )

    model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    model.to(device)
    model.eval()

    # Autoregressive single-step rollout. The model is a fixed one-step operator
    # u_n -> u_{n+1} (spatial-only trunk, predicts the next-frame field directly), so
    # multi-step forecasting feeds each prediction back in as the next state. Only the
    # field channel is updated; the PDE-param channels stay fixed.
    X = X.clone()
    multi_step_output = []
    for _ in range(num_steps):
        with torch.no_grad():
            output = deep_onet_single_image_inference(model, X.to(device).unsqueeze(0), device)
            multi_step_output.append(output.unsqueeze(0).unsqueeze(0).cpu())   # (1, 1, H, W)
            X[0] = output.cpu()                                                # feed prediction back

    return torch.concat(multi_step_output)                                     # (num_steps, 1, H, W)


def deep_onet_single_image_inference(model: DeepONet, X: torch.Tensor,
                                     device: torch.device) -> torch.Tensor:
    # X: (1, C, H, W). Evaluate the one-step operator at every (x, y) query point.
    H = X.shape[-2]
    W = X.shape[-1]

    # All query coordinates, ordered row-major so a (H, W) reshape restores the grid.
    rows, cols = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    xs = cols.reshape(-1).float() / (W - 1)         # normalised x (width)
    ys = rows.reshape(-1).float() / (H - 1)         # normalised y (height)
    trunk = torch.stack([xs, ys], dim=1).unsqueeze(0)   # (1, H*W, 2) — matches training [col, row]

    # One forward over all pixels at once; the model predicts the next-frame field directly.
    pred = model(X, trunk)                          # (1, H*W, 1)

    return pred.reshape(H, W)
