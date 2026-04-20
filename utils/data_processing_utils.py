import torch
import os
from tqdm import tqdm

def min_max_dataset(folder_path, tensor_keys, pde_param_keys, save_dir):
    """Apply min-max normalisation to all files in a dataset and save the results.

    Computes global per-key min and max across the full dataset, then scales
    every value to [0, 1]. Normalised files are written to save_dir
    with the prefix min_max_.

    :param folder_path: Path to directory containing raw .pt simulation files.
    :type folder_path: str
    :param tensor_keys: Keys in each file whose values are multi-dimensional tensors.
    :type tensor_keys: list[str]
    :param pde_param_keys: Keys in each file whose values are scalar PDE parameters.
    :type pde_param_keys: list[str]
    :param save_dir: Directory where normalised files will be written.
    :type save_dir: str
    """
    # Find minimum and maximum for each channel across raw dataset
    tensor_mins, tensor_maxs, pde_param_mins, pde_param_maxs = get_min_max(folder_path, tensor_keys, pde_param_keys)

    # Apply min max normalisation to each file 
    files = [f"{folder_path}/{file}" for file in os.listdir(folder_path)]
    
    pde_param_keys = pde_param_mins.keys()
    tensor_keys = tensor_mins.keys()

    for file in tqdm(files):
        data = torch.load(file)
        data = min_max_file(data, tensor_keys, pde_param_keys, tensor_mins, tensor_maxs, pde_param_mins, pde_param_maxs)
    
        # Save each file to specified directory
        torch.save(data, f"{save_dir}/min_max_{file.split('/')[-1]}")

def get_min_max(folder_path, tensor_keys, pde_param_keys):
    """Compute the per-key global min and max across all files in a dataset.

    Iterates over every .pt file in folder_path, tracking the running
    minimum and maximum for each tensor key (element-wise) and each scalar PDE
    parameter key.

    :param folder_path: Path to directory containing .pt simulation files.
    :type folder_path: str
    :param tensor_keys: Keys whose values are multi-dimensional tensors.
    :type tensor_keys: list[str]
    :param pde_param_keys: Keys whose values are scalar PDE parameters.
    :type pde_param_keys: list[str]
    :returns: Tuple of (tensor_mins, tensor_maxs, pde_param_mins, pde_param_maxs),
              each a dict keyed by the respective key lists.
    :rtype: tuple[dict, dict, dict, dict]
    """
    pde_param_maxs = dict(zip(pde_param_keys, [None] * len(pde_param_keys)))
    pde_param_mins = dict(zip(pde_param_keys, [None] * len(pde_param_keys)))
    tensor_maxs = dict(zip(tensor_keys, len(tensor_keys) * [-torch.inf]))
    tensor_mins = dict(zip(tensor_keys, len(tensor_keys) * [torch.inf]))

    files = [f"{folder_path}/{file}" for file in os.listdir(folder_path)]

    for file in files:
        data = torch.load(file)

        for pde_key in pde_param_keys:

            # If None then we are in the first iteration
            if pde_param_mins[pde_key] is None:
                pde_param_mins[pde_key] = data[pde_key]
                pde_param_maxs[pde_key] = data[pde_key]
            else:

                if pde_param_mins[pde_key] > data[pde_key] :
                    pde_param_mins[pde_key] = data[pde_key]

                if pde_param_maxs[pde_key] < data[pde_key]:
                    pde_param_maxs[pde_key] = data[pde_key]
                
        for tensor_key in tensor_keys:
            if tensor_mins[tensor_key] > data[tensor_key].min():
                tensor_mins[tensor_key] = data[tensor_key].min()
            
            if tensor_maxs[tensor_key] < data[tensor_key].max():
                tensor_maxs[tensor_key] = data[tensor_key].max() 
    return tensor_mins, tensor_maxs, pde_param_mins, pde_param_maxs

def min_max_file(data, tensor_keys, pde_param_keys, tensor_mins, tensor_maxs, pde_param_mins, pde_param_maxs):
    """Apply min-max normalisation to a single data dict.

    Scales each value to [0, 1] using (x - min) / (max - min).

    :param data: Dict loaded from a .pt file.
    :type data: dict
    :param tensor_keys: Keys whose values are multi-dimensional tensors.
    :type tensor_keys: list[str]
    :param pde_param_keys: Keys whose values are scalar PDE parameters.
    :type pde_param_keys: list[str]
    :param tensor_mins: Per-key global minimums for tensor fields.
    :type tensor_mins: dict
    :param tensor_maxs: Per-key global maximums for tensor fields.
    :type tensor_maxs: dict
    :param pde_param_mins: Per-key global minimums for scalar PDE parameters.
    :type pde_param_mins: dict
    :param pde_param_maxs: Per-key global maximums for scalar PDE parameters.
    :type pde_param_maxs: dict
    :returns: Data dict with normalised values.
    :rtype: dict
    """
    for pde_param_key in pde_param_keys:
        x = data[pde_param_key]
        min, max = pde_param_mins[pde_param_key], pde_param_maxs[pde_param_key]
        x_scaled = (x-min)/(max-min)
        data[pde_param_key] = x_scaled
    
    for tensor_key in tensor_keys:
        x = data[tensor_key]
        min, max = tensor_mins[tensor_key], tensor_maxs[tensor_key]
        x_scaled = (x-min)/(max-min)
        data[tensor_key] = x_scaled

    return data

def z_normal_dataset(folder_path, tensor_keys, pde_param_keys, save_dir):
    """Compute z-normalisation statistics across the dataset and save normalised files.

    :param folder_path: Path to directory containing raw .pt simulation files.
    :type folder_path: str
    :param tensor_keys: Keys in each file whose values are multi-dimensional tensors.
    :type tensor_keys: list[str]
    :param pde_param_keys: Keys in each file whose values are scalar PDE parameters.
    :type pde_param_keys: list[str]
    :param save_dir: Directory where normalised files will be written.
    :type save_dir: str
    """
    pde_param_means, tensor_means = calculate_channel_means(folder_path, tensor_keys, pde_param_keys)
    pde_param_stds, tensor_stds = calculate_channel_std(folder_path, tensor_keys, pde_param_keys, pde_param_means, tensor_means)

    files = [f"{folder_path}/{file}" for file in os.listdir(folder_path) if file.endswith('.pt')]

    for file in tqdm(files):
        data = torch.load(file, weights_only=False)
        data = z_norm_file(data, tensor_keys, pde_param_keys, tensor_means, tensor_stds, pde_param_means, pde_param_stds)
        basename = os.path.basename(file)
        torch.save(data, f"{save_dir}/z_norm_{basename}")

def calculate_channel_means(folder_path, tensor_keys, pde_param_keys):
    """Compute the per-key mean across all files in a dataset.

    For tensor keys the mean is computed over all elements across all files.
    For PDE parameter keys (scalars) the mean is computed across all files.

    :param folder_path: Path to directory containing .pt simulation files.
    :type folder_path: str
    :param tensor_keys: Keys whose values are multi-dimensional tensors.
    :type tensor_keys: list[str]
    :param pde_param_keys: Keys whose values are scalar PDE parameters.
    :type pde_param_keys: list[str]
    :returns: Tuple of (pde_param_means, tensor_means) dicts keyed by their respective keys.
    :rtype: tuple[dict, dict]
    """
    pde_param_means = dict(zip(pde_param_keys, [0.0] * len(pde_param_keys)))
    tensor_means = dict(zip(tensor_keys, len(tensor_keys) * [torch.tensor(0.0)]))

    files = [f"{folder_path}/{file}" for file in os.listdir(folder_path) if file.endswith('.pt')]
    tensor_numel = torch.load(files[0], weights_only=False)[tensor_keys[0]].numel()

    for file in files:
        data = torch.load(file, weights_only=False)

        for pde_key in pde_param_keys:
            pde_param_means[pde_key] = pde_param_means[pde_key] + float(data[pde_key])

        for tensor_key in tensor_keys:
            tensor_means[tensor_key] = tensor_means[tensor_key] + data[tensor_key].sum()

    for pde_key in pde_param_keys:
        pde_param_means[pde_key] = pde_param_means[pde_key] / len(files)

    for tensor_key in tensor_keys:
        tensor_means[tensor_key] = tensor_means[tensor_key] / (tensor_numel * len(files))

    return pde_param_means, tensor_means

def calculate_channel_std(folder_path, tensor_keys, pde_param_keys, pde_param_means, tensor_means):
    """Compute per-key standard deviation across all files given pre-computed means.

    Uses a two-pass approach: means must be supplied from :func:`calculate_channel_means`.

    :param folder_path: Path to directory containing .pt simulation files.
    :type folder_path: str
    :param tensor_keys: Keys whose values are multi-dimensional tensors.
    :type tensor_keys: list[str]
    :param pde_param_keys: Keys whose values are scalar PDE parameters.
    :type pde_param_keys: list[str]
    :param pde_param_means: Pre-computed scalar means for each PDE parameter key.
    :type pde_param_means: dict
    :param tensor_means: Pre-computed tensor means for each tensor key.
    :type tensor_means: dict
    :returns: Tuple of (pde_param_stds, tensor_stds) dicts keyed by their respective keys.
    :rtype: tuple[dict, dict]
    """
    pde_param_sq_devs = dict(zip(pde_param_keys, [0.0] * len(pde_param_keys)))
    tensor_sq_devs = dict(zip(tensor_keys, len(tensor_keys) * [torch.tensor(0.0)]))

    files = [f"{folder_path}/{file}" for file in os.listdir(folder_path) if file.endswith('.pt')]
    tensor_numel = torch.load(files[0], weights_only=False)[tensor_keys[0]].numel()

    for file in files:
        data = torch.load(file, weights_only=False)

        for pde_key in pde_param_keys:
            pde_param_sq_devs[pde_key] = pde_param_sq_devs[pde_key] + (float(data[pde_key]) - pde_param_means[pde_key]) ** 2

        for tensor_key in tensor_keys:
            tensor_sq_devs[tensor_key] = tensor_sq_devs[tensor_key] + ((data[tensor_key] - tensor_means[tensor_key]) ** 2).sum()

    pde_param_stds = {}
    for pde_key in pde_param_keys:
        pde_param_stds[pde_key] = (pde_param_sq_devs[pde_key] / len(files)) ** 0.5

    tensor_stds = {}
    for tensor_key in tensor_keys:
        tensor_stds[tensor_key] = torch.sqrt(tensor_sq_devs[tensor_key] / (tensor_numel * len(files)))

    return pde_param_stds, tensor_stds

def z_norm_file(data, tensor_keys, pde_param_keys, tensor_means, tensor_stds, pde_param_means, pde_param_stds):
    """Apply z-normalisation to a single data dict.

    :param data: Dict loaded from a .pt file.
    :type data: dict
    :param tensor_keys: Keys whose values are multi-dimensional tensors.
    :type tensor_keys: list[str]
    :param pde_param_keys: Keys whose values are scalar PDE parameters.
    :type pde_param_keys: list[str]
    :param tensor_means: Per-key means for tensor fields.
    :type tensor_means: dict
    :param tensor_stds: Per-key standard deviations for tensor fields.
    :type tensor_stds: dict
    :param pde_param_means: Per-key means for scalar PDE parameters.
    :type pde_param_means: dict
    :param pde_param_stds: Per-key standard deviations for scalar PDE parameters.
    :type pde_param_stds: dict
    :returns: Data dict with normalised values.
    :rtype: dict
    """
    for pde_key in pde_param_keys:
        x = float(data[pde_key])
        mean, std = pde_param_means[pde_key], pde_param_stds[pde_key]
        data[pde_key] = (x - mean) / std if std != 0 else 0.0

    for tensor_key in tensor_keys:
        x = data[tensor_key]
        mean, std = tensor_means[tensor_key], tensor_stds[tensor_key]
        data[tensor_key] = (x - mean) / std if std.item() != 0 else torch.zeros_like(x)

    return data


