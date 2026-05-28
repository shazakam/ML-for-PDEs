import torch.nn as nn

ACTIVATIONS: dict[str, nn.Module] = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "softplus": nn.Softplus(),
    "mish": nn.Mish(),
    "silu": nn.SiLU(),
    "identity": nn.Identity(),
}
