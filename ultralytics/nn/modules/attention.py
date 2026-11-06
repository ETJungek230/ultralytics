import math
import torch
import torch.nn as nn


class SE(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        hidden_channels = max(1, channels // reduction)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape

        # (B, C, H, W) -> (B, C)
        y = self.pool(x).view(b, c)
        # (B, C) -> (B, C, 1, 1)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y


class ECA(nn.Module):
    """Efficient Channel Attention."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, C, 1, 1)
        y = self.pool(x)
        # (B, C, 1, 1) -> (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)
        # Local cross-channel interaction
        y = self.conv(y)
        # (B, 1, C) -> (B, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y


class CA(nn.Module):
    """Coordinate Attention (CA) module.

    Splits channel attention into two 1D spatial directions (height and width)
    to capture long-range dependencies along each spatial axis precisely.
    This allows the model to encode complete position information in one axis
    and selectively focus on the other.

    References:
        https://arxiv.org/abs/2103.02907 (CoordAtt, AAAI 2021)
    """

    def __init__(self, channels: int, reduction: int = 16):
        """Initialize Coordinate Attention module.

        Args:
            channels (int): Number of input channels.
            reduction (int): Channel reduction ratio.
        """
        super().__init__()
        hidden_channels = max(1, channels // reduction)

        # Global Average Pool along Height (keep width dimension)
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        # Global Average Pool along Width (keep height dimension)
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))

        # Shared MLP for channel compression
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
        )

        # Generate attention weights for each spatial direction
        self.attn_w = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.attn_h = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Coordinate Attention to input tensor.

        Optimized for PyTorch 2.15: removes unnecessary shape unpacking.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            (torch.Tensor): Attention-weighted output tensor of shape (B, C, H, W).
        """
        # Pool along width axis -> (B, C, 1, W)
        x_w = self.avg_pool_w(x)
        # Pool along height axis -> (B, C, H, 1)
        x_h = self.avg_pool_h(x)

        # Shared MLP for channel compression
        y_w = self.fc(x_w)  # (B, hidden, 1, W)
        y_h = self.fc(x_h)  # (B, hidden, H, 1)

        # Generate attention maps and apply
        attn_w = self.attn_w(y_w)  # (B, C, 1, W)
        attn_h = self.attn_h(y_h)  # (B, C, H, 1)

        return x * attn_w * attn_h
