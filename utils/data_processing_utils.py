import torch
import os

def min_max_dataset(folder_path, tensor_keys, pde_param_keys, save_dir):

    # Find minimum and maximum for each channel across raw dataset

    # Apply min max normalisation to each file 
    
        # Save each file to specified directory
    pass

def get_min_max(folder_path, tensor_keys, pde_param_keys):
    return

def z_normal_dataset(folder_path, tensor_keys, pde_param_keys):
    # Calculate mean across each channel

    # Calculate std across each channel

    # Apply z_norm to each file 
        # Save each file to specified directory

    return

def calculate_channel_means(folder_path, tensor_keys, pde_param_keys):
    pde_param_means = dict(zip(pde_param_keys, [0.0] * len(pde_param_keys)))
    tensor_means = dict(zip(tensor_keys, len(tensor_keys) * [torch.tensor(0.0)]))

    files = [f"{folder_path}/{file}" for file in os.listdir(folder_path)]
    tensor_count = [1*x for x in torch.load(files[0])[pde_param_keys[0]].shape][0]

    for file in files:
        data = torch.load(file)

        for pde_key in pde_param_keys:
            pde_param_means[pde_key] = pde_param_means[pde_key] + float(data[pde_key])

        for tensor_key in tensor_keys:
            tensor_means[tensor_key] = tensor_means[tensor_key] + data[tensor_key].sum()

    for pde_key in pde_param_keys:
        pde_param_means[pde_key] = pde_param_means[pde_key] / len(files)

    for tensor_key in tensor_keys:
        tensor_means = tensor_means[tensor_key] / (tensor_count*len(files))
        
    return pde_param_means, tensor_means

def calculate_channel_std(folder_path, tensor_keys, pde_param_keys):
    return

