from typing import Any
import torch
import torch.nn as nn

class FNOLayer(nn.Module):

    def __init__(self, num_modes : int, d_v : int, ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.d_v = d_v
        self.R = nn.Parameter(torch.randn(num_modes, num_modes, d_v, d_v))
        # self.W = 

    def forward(self, x): # x has shape (dv, H, W)
        # Note to self:
        # switch to fft2 -> need both positive and negative frequencies
        # batch needs to be included in einsum
        # zero pad before applying inverse transform

        # x = torch.fft.rfft2(x)
        # x = x[:, :self.num_modes, :self.num_modes]
        # x = torch.einsum('kxlj,kxj->kxl', self.R, x)
        # x = torch.fft.irfft2(x)
        return