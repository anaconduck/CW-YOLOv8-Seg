"""
CARAFE: Content-Aware ReAssembly of FEatures (ICCV 2019)
Pure PyTorch implementation with autograd and CUDA acceleration.
Designed for seamless integration into YOLOv8 feature pyramid networks.
Reference: Wang et al., "CARAFE: Content-Aware ReAssembly of FEatures", ICCV 2019.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    """
    Content-Aware ReAssembly of FEatures (CARAFE) upsampling operator.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (usually equal to in_channels).
        scale_factor (int): Upsampling scale factor (default: 2).
        up_kernel (int): Reassembly kernel size (default: 5).
        encoder_kernel (int): Kernel size for generating reassembly weights (default: 3).
        compressed_channels (int): Compressed channel count for lightweight kernel generation (default: 64).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        scale_factor: int = 2,
        up_kernel: int = 5,
        encoder_kernel: int = 3,
        compressed_channels: int = 64
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.encoder_kernel = encoder_kernel

        # Channel compression to reduce computational cost
        self.compressed_channels = min(compressed_channels, in_channels)
        self.channel_compressor = nn.Conv2d(
            in_channels,
            self.compressed_channels,
            kernel_size=1,
            bias=False
        )

        # Content encoder: generates reassembly kernels
        encoder_padding = encoder_kernel // 2
        self.kernel_encoder = nn.Conv2d(
            self.compressed_channels,
            (scale_factor * up_kernel) ** 2,
            kernel_size=encoder_kernel,
            padding=encoder_padding,
            bias=False
        )

        # Optional 1x1 conv if in_channels != out_channels
        if self.in_channels != self.out_channels:
            self.post_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.post_conv = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in [self.channel_compressor, self.kernel_encoder]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CARAFE.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Upsampled tensor of shape (B, C_out, H * scale, W * scale).
        """
        B, C, H, W = x.shape
        S = self.scale_factor
        K = self.up_kernel
        pad = K // 2

        # 1. Kernel prediction
        compressed = self.channel_compressor(x)  # (B, C_comp, H, W)
        kernel = self.kernel_encoder(compressed)  # (B, (S * K)^2, H, W)

        # Reshape & rearrange kernel to (B, H*S, W*S, K*K)
        # Pixel-shuffle-like permutation for spatial assignment
        kernel = F.pixel_shuffle(kernel, S)  # (B, K*K, H*S, W*S)
        kernel = kernel.permute(0, 2, 3, 1)  # (B, H*S, W*S, K*K)
        kernel = F.softmax(kernel, dim=-1)   # Normalize kernel weights per location

        # 2. Content-Aware Reassembly
        # Pad input for receptive field K
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='replicate')

        # Extract sliding local blocks of size K x K from input
        # unfold produces (B, C * K * K, H * W)
        patches = F.unfold(x_padded, kernel_size=K, stride=1)
        patches = patches.view(B, C, K * K, H, W)

        # Repeat patches across scale factor S
        # Shape: (B, C, K*K, H, 1, W, 1) -> (B, C, K*K, H, S, W, S) -> (B, C, K*K, H*S, W*S)
        patches = patches.unsqueeze(4).unsqueeze(6).repeat(1, 1, 1, 1, S, 1, S)
        patches = patches.view(B, C, K * K, H * S, W * S)
        patches = patches.permute(0, 3, 4, 1, 2)  # (B, H*S, W*S, C, K*K)

        # Multiply and sum: (B, H*S, W*S, C, K*K) * (B, H*S, W*S, 1, K*K) -> sum over K*K
        kernel = kernel.unsqueeze(3)  # (B, H*S, W*S, 1, K*K)
        out = torch.sum(patches * kernel, dim=-1)  # (B, H*S, W*S, C)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, C, H*S, W*S)

        return self.post_conv(out)
