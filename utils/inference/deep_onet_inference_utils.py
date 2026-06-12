import torch

from models.forecasting.deep_onet import DeepONet


def deep_onet_forecast(model_path: str, model_cfg: dict, num_steps: int,
                       device: torch.device, X: torch.Tensor,
                       step_dt: float) -> torch.Tensor:
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

    # Clone so the caller's tensor is not mutated by the in-place feedback below.
    X = X.clone()
    multi_step_output = []
    for _ in range(num_steps):
        with torch.no_grad():
            # Autoregressive single-step rollout: every step advances the state by one
            # frame, so the trunk time-offset is the *constant* single-step Δt. This
            # must match the (tgt_frame - src_frame) / (T - 1) normalisation used in
            # training, i.e. step_dt = 1 / (num_frames_per_sim - 1).
            output = deep_onet_single_image_inference(model, X.to(device).unsqueeze(0), step_dt, device)
            multi_step_output.append(output.unsqueeze(0).unsqueeze(0).cpu())   # (1, 1, H, W)
            X[0] = output.cpu()                                                # feed prediction back

    return torch.concat(multi_step_output)                                     # (num_steps, 1, H, W)


def deep_onet_single_image_inference(model: DeepONet, X: torch.Tensor, t: float,
                                     device: torch.device) -> torch.Tensor:
    # X: (1, C, H, W). Evaluate the operator at every (x, y) query point for time offset t.
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
    ts = torch.full_like(xs, float(t))
    trunk = torch.stack([xs, ys, ts], dim=1).unsqueeze(0)   # (1, H*W, 3) — matches training [col, row, Δt]

    # One forward over all pixels at once; the branch is encoded once for the single image
    # and the bias b0 is applied inside forward.
    pred = model(X, trunk)                          # (1, H*W, 1)

    return pred.reshape(H, W)
