import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import torch.fft


class SSRModule(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.spatial_filter = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)

        self.freq_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.freq_weight, std=.02)

        self.norm_pre = nn.GroupNorm(num_groups=1, num_channels=dim, eps=1e-6)
        self.norm_post = nn.GroupNorm(num_groups=1, num_channels=dim, eps=1e-6)

    def forward(self, x):
        x = self.norm_pre(x)

        x_spatial, x_spectral = torch.chunk(x, 2, dim=1)

        x_spatial = self.spatial_filter(x_spatial)

        x_spectral = x_spectral.to(torch.float32)
        B, C_half, H, W = x_spectral.shape

        spectrum = torch.fft.rfft2(x_spectral, dim=(2, 3), norm='ortho')

        weight = self.freq_weight
        target_h, target_w = spectrum.shape[2], spectrum.shape[3]
        if not weight.shape[1:3] == (target_h, target_w):
            weight = F.interpolate(weight.permute(3, 0, 1, 2),
                                   size=(target_h, target_w),
                                   mode='bilinear',
                                   align_corners=True).permute(1, 2, 3, 0)

        modulator = torch.view_as_complex(weight.contiguous())
        spectrum = spectrum * modulator

        x_spectral = torch.fft.irfft2(spectrum, s=(H, W), dim=(2, 3), norm='ortho')

        x_out = torch.cat([x_spatial, x_spectral], dim=1)
        x_out = self.norm_post(x_out)

        return x_out
