from typing import Any
import torch.nn as nn
from ..activations import ACTIVATIONS

class Conv2DBlock(nn.Module):
    def __init__(self, c_in, c_out, stride, activation) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            ACTIVATIONS[activation](inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x):
        return self.conv_block(x)