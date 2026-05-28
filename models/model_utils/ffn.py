import torch
import torch.nn as nn

from models.model_utils.config import ACTIVATIONS


class FFN(nn.Module):

    def __init__(self, layer_sizes: list[int], activation: str | list[str],  dtype:torch.dtype, dropout_rate: float | list[float] = 0) -> None:
        super().__init__()

        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self.activation = [ACTIVATIONS[a] for a in activation]
        else:
            self.activation = ACTIVATIONS[activation]

        if isinstance(dropout_rate, list):
            if len(layer_sizes) - 1 != len(dropout_rate):
                raise ValueError(
                    f"Number of dropout rates must be equal to {len(layer_sizes) - 1}"
                )
            self.dropout_rate = dropout_rate
        else:
            self.dropout_rate = [dropout_rate] * (len(layer_sizes) - 1)

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=dtype
                )
            )

    def forward(self, x):
        for j, linear in enumerate(self.linears[:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
            if self.dropout_rate[j] > 0:
                x = torch.nn.functional.dropout(
                    x, p=self.dropout_rate[j], training=self.training
                )
        x = self.linears[-1](x)

        return
