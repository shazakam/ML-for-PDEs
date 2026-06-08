from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
class FNOLayer(nn.Module):

    def __init__(self, num_modes : int, d_v : int, ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.d_v = d_v
        self.R_high = nn.Parameter(torch.randn(num_modes, num_modes, d_v, d_v)) 
        self.R_low = nn.Parameter(torch.randn(num_modes, num_modes, d_v, d_v))
        self.W =  nn.Conv2d(in_channels=d_v, out_channels=d_v, kernel_size=1)

    def forward(self, v_t):  # v_t: (B, d_v, H, W)
        _, _, H, W = v_t.shape
        
        # FFT
        x = torch.fft.rfft2(v_t)
        
        # Extract corners
        x_pp = x[:, :, :self.num_modes, :self.num_modes]          
        x_mp = x[:, :, -self.num_modes:, :self.num_modes]      
        
        # Apply spectral weights per mode
        x_pp = torch.einsum('mnoi,bimn->bomn', self.R_low,  x_pp)
        x_mp = torch.einsum('mnoi,bimn->bomn', self.R_high, x_mp)
        
        # Scatter back
        x_out = torch.zeros_like(x)
        x_out[:, :, :self.num_modes, :self.num_modes] = x_pp
        x_out[:, :, -self.num_modes:, :self.num_modes] = x_mp
        
        # Inverse FFT
        spectral_out = torch.fft.irfft2(x_out, s=(H, W))  # (B, d_v, H, W)
        
        # Local skip
        local_out = self.W(v_t)  # (B, d_v, H, W)
        
        # Add and activate
        return F.relu(spectral_out + local_out)