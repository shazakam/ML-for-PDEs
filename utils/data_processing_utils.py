import torch
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

def aggregate_files(folder_path: str, save_path: str) -> None:

    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pt')])
    if not files:
        raise ValueError(f"No .pt files found in {folder_path}")

    file_paths = [f"{folder_path}/{f}" for f in files]
    n = len(file_paths)

    first = torch.load(file_paths[0], weights_only=False, mmap=True)
    X_shape = first['X'].shape  # (T, H, W)
    dtype = first['X'].dtype
    scalar_keys = [k for k in first.keys() if k != 'X']

    # Map torch dtype → numpy dtype for the memmap staging file
    np_dtype = torch.empty(0, dtype=dtype).numpy().dtype
    tmp_path = save_path + '.tmp.bin'

    try:
        staging = np.memmap(tmp_path, dtype=np_dtype, mode='w+', shape=(n, *X_shape))

        scalar_data: dict[str, list] = {k: [] for k in scalar_keys}
        for i, fp in enumerate(tqdm(file_paths)):
            data = torch.load(fp, weights_only=False, mmap=True)
            assert data['X'].shape == X_shape, \
                f"Shape mismatch at {fp}: {data['X'].shape} vs {X_shape}"
            staging[i] = data['X'].numpy()
            for k in scalar_keys:
                scalar_data[k].append(float(data[k]))

        staging.flush()

        result = {'X': torch.from_numpy(np.array(staging))}
        for k in scalar_keys:
            result[k] = torch.tensor(scalar_data[k], dtype=dtype)

        torch.save(result, save_path)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _find_aggregated_file(folder_path: str) -> str:
    """Return the path to the single aggregated .pt file in folder_path.

    Looks first for files matching aggregated_*.pt, then falls back to any
    .pt file.  Raises FileNotFoundError if nothing is found.

    :param folder_path: Directory that contains the aggregated file.
    :type folder_path: str
    :returns: Absolute path to the aggregated .pt file.
    :rtype: str
    """
    candidates = sorted(Path(folder_path).glob("aggregated_*.pt"))
    if not candidates:
        candidates = sorted(Path(folder_path).glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt file found in {folder_path}")
    return str(candidates[0])


def min_max_aggregated(
    folder_path: str,
    tensor_keys: list[str],
    pde_param_keys: list[str],
    train_output_dir: str,
    test_output_dir: str,
    dataset_name: str,
    train_ratio: float = 0.8,
) -> None:
    """Normalise an aggregated dataset to [0, 1] and split into train / test files.

    Statistics are computed on the training split only and then applied to both
    splits, preventing data leakage from the test set.

    :param folder_path: Directory containing the aggregated .pt file.
    :type folder_path: str
    :param tensor_keys: Keys whose values are multi-dimensional tensors (shape (N, ...)).
    :type tensor_keys: list[str]
    :param pde_param_keys: Keys whose values are 1-D scalar parameter tensors (shape (N,)).
    :type pde_param_keys: list[str]
    :param train_output_dir: Directory where the normalised train file will be saved.
    :type train_output_dir: str
    :param test_output_dir: Directory where the normalised test file will be saved.
    :type test_output_dir: str
    :param train_ratio: Fraction of samples used for training (default 0.8).
    :type train_ratio: float
    """
    aggregated_path = _find_aggregated_file(folder_path)
    data = torch.load(aggregated_path, weights_only=False, mmap=True)

    N = data['X'].shape[0]
    n_train = int(N * train_ratio)

    # Compute stats on train split only
    tensor_mins, tensor_maxs, param_mins, param_maxs = {}, {}, {}, {}
    for k in tensor_keys:
        tensor_mins[k] = data[k][:n_train].min()
        tensor_maxs[k] = data[k][:n_train].max()
    for k in pde_param_keys:
        param_mins[k] = data[k][:n_train].min()
        param_maxs[k] = data[k][:n_train].max()

    def _normalise(split: dict) -> dict:
        out = {}
        for k in tensor_keys:
            out[k] = (split[k] - tensor_mins[k]) / (tensor_maxs[k] - tensor_mins[k])
        for k in pde_param_keys:
            out[k] = (split[k] - param_mins[k]) / (param_maxs[k] - param_mins[k])
        return out

    train_data = _normalise({k: data[k][:n_train].clone() for k in tensor_keys + pde_param_keys})
    test_data  = _normalise({k: data[k][n_train:].clone() for k in tensor_keys + pde_param_keys})

    Path(train_output_dir).mkdir(parents=True, exist_ok=True)
    Path(test_output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(train_data, f"{train_output_dir}/{dataset_name}_minmax.pt")
    torch.save(test_data,  f"{test_output_dir}/{dataset_name}_minmax.pt")


def z_normal_aggregated(
    folder_path: str,
    tensor_keys: list[str],
    pde_param_keys: list[str],
    train_output_dir: str,
    test_output_dir: str,
    dataset_name: str,
    train_ratio: float = 0.8,
) -> None:
    """Z-normalise an aggregated dataset and split into train / test files.

    Statistics (mean, std) are computed on the training split only.

    :param folder_path: Directory containing the aggregated .pt file.
    :type folder_path: str
    :param tensor_keys: Keys whose values are multi-dimensional tensors (shape (N, ...)).
    :type tensor_keys: list[str]
    :param pde_param_keys: Keys whose values are 1-D scalar parameter tensors (shape (N,)).
    :type pde_param_keys: list[str]
    :param train_output_dir: Directory where the normalised train file will be saved.
    :type train_output_dir: str
    :param test_output_dir: Directory where the normalised test file will be saved.
    :type test_output_dir: str
    :param train_ratio: Fraction of samples used for training (default 0.8).
    :type train_ratio: float
    """
    aggregated_path = _find_aggregated_file(folder_path)
    data = torch.load(aggregated_path, weights_only=False, mmap=True)

    N = data['X'].shape[0]
    n_train = int(N * train_ratio)

    # Compute stats on train split only
    tensor_means, tensor_stds, param_means, param_stds = {}, {}, {}, {}
    for k in tensor_keys:
        train_slice = data[k][:n_train].float()
        tensor_means[k] = train_slice.mean()
        tensor_stds[k]  = train_slice.std()
    for k in pde_param_keys:
        train_slice = data[k][:n_train].float()
        param_means[k] = train_slice.mean()
        param_stds[k]  = train_slice.std()

    def _normalise(split: dict) -> dict:
        out = {}
        for k in tensor_keys:
            std = tensor_stds[k]
            out[k] = (split[k] - tensor_means[k]) / std if std.item() != 0 else torch.zeros_like(split[k])
        for k in pde_param_keys:
            std = param_stds[k]
            out[k] = (split[k] - param_means[k]) / std if std.item() != 0 else torch.zeros_like(split[k])
        return out

    train_data = _normalise({k: data[k][:n_train].clone() for k in tensor_keys + pde_param_keys})
    test_data  = _normalise({k: data[k][n_train:].clone() for k in tensor_keys + pde_param_keys})

    Path(train_output_dir).mkdir(parents=True, exist_ok=True)
    Path(test_output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(train_data, f"{train_output_dir}/{dataset_name}_z.pt")
    torch.save(test_data,  f"{test_output_dir}/{dataset_name}_z.pt")

def load_single_sample(field_keys, data_file_path, input_idx):
    dataset = torch.load(data_file_path, mmap=True)

    sim_idx   = input_idx // (dataset.shape[1])
    frame_idx = input_idx %  (dataset.shape[1])

    X_t = dataset['X'][sim_idx, frame_idx]                         # (H, W)
    pde_params = [float(dataset[k][sim_idx]) for k in field_keys]
    X = torch.stack([X_t] + [torch.full_like(X_t, p) for p in pde_params], dim=0) 

    return X
