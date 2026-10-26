# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Native PyTorch implementations of selected PaddleDetection model blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .head import Detect


def _make_divisible(value: float, divisor: int = 8) -> int:
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    return rounded + divisor if rounded < 0.9 * value else rounded


class HardSwish(nn.Module):
    """Hard-swish activation used by PPLCNet."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3, inplace=True) / 6


class ConvBNAct(nn.Sequential):
    """Convolution, batch normalization and Paddle-compatible activation."""

    def __init__(self, c1: int, c2: int, k: int, s: int = 1, groups: int = 1, act: str = "hswish"):
        activation = HardSwish() if act == "hswish" else nn.ReLU(inplace=True) if act == "relu" else nn.Identity()
        super().__init__(
            nn.Conv2d(c1, c2, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            activation,
        )


class SqueezeExcitation(nn.Module):
    """PPLCNet channel squeeze-and-excitation block."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = F.hardsigmoid(self.fc2(scale), inplace=True)
        return x * scale


class LCNetBlock(nn.Module):
    """Depthwise-separable PPLCNet block with optional SE."""

    def __init__(self, c1: int, c2: int, k: int, stride: int, use_se: bool):
        super().__init__()
        self.dw = ConvBNAct(c1, c1, k, stride, groups=c1)
        self.se = SqueezeExcitation(c1) if use_se else nn.Identity()
        self.pw = ConvBNAct(c1, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.se(self.dw(x)))


class LCNet(nn.Module):
    """PPLCNet backbone returning stride-8/16/32 feature maps."""

    def __init__(self, c1: int = 3, scale: float = 0.75):
        super().__init__()
        # NET_CONFIG blocks2..6 and feature_maps=[3,4,5] from Paddle LCNet.
        stages = (
            [(3, 16, 32, 1, False)],
            [(3, 32, 64, 2, False), (3, 64, 64, 1, False)],
            [(3, 64, 128, 2, False), (3, 128, 128, 1, False)],
            [(3, 128, 256, 2, False)] + [(5, 256, 256, 1, False)] * 5,
            [(5, 256, 512, 2, True), (5, 512, 512, 1, True)],
        )
        self.stem = ConvBNAct(c1, _make_divisible(16 * scale), 3, 2)
        self.stages = nn.ModuleList()
        channels = _make_divisible(16 * scale)
        self.out_channels = []
        for stage_index, stage in enumerate(stages):
            blocks = []
            for kernel, _, out, stride, se in stage:
                out = _make_divisible(out * scale)
                blocks.append(LCNetBlock(channels, out, kernel, stride, se))
                channels = out
            self.stages.append(nn.Sequential(*blocks))
            if stage_index >= 2:  # Paddle feature_maps [3, 4, 5] => blocks4, blocks5, blocks6.
                self.out_channels.append(channels)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        outputs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i >= 2:
                outputs.append(x)
        return outputs


class LCPAN(nn.Module):
    """Depthwise LC-PAN with top-down and bottom-up feature fusion."""

    def __init__(self, channels: list[int] | tuple[int, ...], out_channels: int = 96, num_features: int = 4):
        super().__init__()
        if len(channels) != 3 or num_features != 4:
            raise ValueError("LCPAN expects three LCNet inputs and four output levels")
        self.lateral = nn.ModuleList(ConvBNAct(c, out_channels, 1) for c in channels)
        self.top = nn.ModuleList(
            nn.Sequential(
                LCNetBlock(2 * out_channels, 2 * out_channels, 5, 1, False),
                LCNetBlock(2 * out_channels, out_channels, 5, 1, False),
            )
            for _ in range(2)
        )
        self.down = nn.ModuleList(ConvBNAct(out_channels, out_channels, 5, 2) for _ in range(2))
        self.bottom = nn.ModuleList(
            nn.Sequential(
                LCNetBlock(2 * out_channels, 2 * out_channels, 5, 1, False),
                LCNetBlock(2 * out_channels, out_channels, 5, 1, False),
            )
            for _ in range(2)
        )
        self.first_top = ConvBNAct(out_channels, out_channels, 5, 2)
        self.second_top = ConvBNAct(out_channels, out_channels, 5, 2)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        p3, p4, p5 = [layer(x) for layer, x in zip(self.lateral, xs)]
        inner5 = p5
        inner4 = self.top[0](torch.cat((F.interpolate(inner5, scale_factor=2, mode="nearest"), p4), 1))
        inner3 = self.top[1](torch.cat((F.interpolate(inner4, scale_factor=2, mode="nearest"), p3), 1))
        out3 = inner3
        out4 = self.bottom[0](torch.cat((self.down[0](out3), inner4), 1))
        out5 = self.bottom[1](torch.cat((self.down[1](out4), inner5), 1))
        out6 = self.first_top(p5) + self.second_top(out5)
        return [out3, out4, out5, out6]


class PicoFeat(nn.Module):
    """PicoDet feature tower: repeated depthwise 5x5 + pointwise 1x1 blocks."""

    def __init__(self, channels: int = 96, num_convs: int = 2, use_se: bool = True):
        super().__init__()
        self.blocks = nn.Sequential(
            *(nn.Sequential(ConvBNAct(channels, channels, 5, groups=channels), ConvBNAct(channels, channels, 1))
              for _ in range(num_convs))
        )
        self.se = SqueezeExcitation(channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.blocks(x))


class TinyPoseHeatmap(nn.Module):
    """Top-down keypoint heatmap head used after a LiteHRNet feature map."""

    def __init__(self, c1: int, num_joints: int = 17, upsample: int = 4):
        super().__init__()
        self.num_joints = num_joints
        self.final = nn.Conv2d(c1, num_joints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(x)


class _ShuffleUnit(nn.Module):
    """NAIVE LiteHRNet shuffle unit from Paddle's wider_naive configuration."""

    def __init__(self, channels: int, stride: int):
        super().__init__()
        half = channels // 2
        self.stride = stride
        if stride == 2:
            self.branch1 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 2, 1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.Conv2d(channels, half, 1, bias=False),
                nn.BatchNorm2d(half),
                nn.ReLU(inplace=True),
            )
            branch_in = channels
        else:
            self.branch1 = None
            branch_in = half
        self.branch2 = nn.Sequential(
            nn.Conv2d(branch_in, half, 1, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, 3, stride, 1, groups=half, bias=False),
            nn.BatchNorm2d(half),
            nn.Conv2d(half, half, 1, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 2:
            x = torch.cat((self.branch1(x), self.branch2(x)), 1)
        else:
            a, b = x.chunk(2, 1)
            x = torch.cat((a, self.branch2(b)), 1)
        b, c, h, w = x.shape
        return x.reshape(b, 2, c // 2, h, w).transpose(1, 2).reshape(b, c, h, w)


class LiteHRModule(nn.Module):
    """Parallel NAIVE branches and the multi-resolution fusion used by LiteHRNet."""

    def __init__(self, channels: list[int], num_blocks: int = 2):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(*[_ShuffleUnit(c, 1) for _ in range(num_blocks)]) for c in channels
        )
        fuse = []
        for target, target_channels in enumerate(channels):
            row = nn.ModuleList()
            for source, source_channels in enumerate(channels):
                if source == target:
                    row.append(nn.Identity())
                elif source > target:
                    row.append(
                        nn.Sequential(
                            nn.Conv2d(source_channels, target_channels, 1, bias=False),
                            nn.BatchNorm2d(target_channels),
                            nn.Upsample(scale_factor=2 ** (source - target), mode="nearest"),
                        )
                    )
                else:
                    ops = []
                    in_ch = source_channels
                    for step in range(target - source):
                        out_ch = target_channels if step == target - source - 1 else in_ch
                        ops.extend(
                            [
                                nn.Conv2d(in_ch, in_ch, 3, 2, 1, groups=in_ch, bias=False),
                                nn.BatchNorm2d(in_ch),
                                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                nn.BatchNorm2d(out_ch),
                            ]
                        )
                        if step != target - source - 1:
                            ops.append(nn.ReLU(inplace=True))
                        in_ch = out_ch
                    row.append(nn.Sequential(*ops))
            fuse.append(row)
        self.fuse = nn.ModuleList(fuse)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        xs = [branch(x) for branch, x in zip(self.branches, xs)]
        outputs = []
        for i, row in enumerate(self.fuse):
            y = row[0](xs[0])
            for j in range(len(xs)):
                if j == 0:
                    y = y + y
                elif j == i:
                    y = y + xs[j]
                else:
                    y = y + row[j](xs[j])
            outputs.append(F.relu(y, inplace=True))
        return outputs


class LiteHRNetWiderNaive(nn.Module):
    """Paddle LiteHRNet ``wider_naive`` with [2,4,2] modules and [40,80,160,320] channels."""

    def __init__(self, c1: int = 3, width: int = 40):
        super().__init__()
        # Official Stem: conv1 then split/dw-stride2 and channel shuffle.
        self.stem_conv = ConvBNAct(c1, 32, 3, 2, act="relu")
        self.stem_left = nn.Sequential(
            ConvBNAct(16, 16, 3, 2, groups=16, act="identity"), ConvBNAct(16, 16, 1, act="relu")
        )
        self.stem_right = nn.Sequential(
            ConvBNAct(16, 32, 1, act="relu"),
            ConvBNAct(32, 32, 3, 2, groups=32, act="identity"),
            ConvBNAct(32, 16, 1, act="relu"),
        )
        self.transition0 = nn.ModuleList([self._transition(32, width), self._transition(32, width * 2, down=True)])
        self.stage0 = nn.Sequential(*[LiteHRModule([width, width * 2]) for _ in range(2)])
        self.transition1 = nn.ModuleList(
            [nn.Identity(), nn.Identity(), self._transition(width * 2, width * 4, down=True)]
        )
        self.stage1 = nn.Sequential(*[LiteHRModule([width, width * 2, width * 4]) for _ in range(4)])
        self.transition2 = nn.ModuleList(
            [nn.Identity(), nn.Identity(), nn.Identity(), self._transition(width * 4, width * 8, down=True)]
        )
        self.stage2 = nn.Sequential(
            *[LiteHRModule([width, width * 2, width * 4, width * 8]) for _ in range(2)]
        )
        self.head_dw = nn.ModuleList()
        self.head_pw = nn.ModuleList()
        rev = [width * 8, width * 4, width * 2, width]
        for i, ch in enumerate(rev):
            out_ch = rev[i + 1] if i < 3 else ch
            self.head_dw.append(ConvBNAct(ch, ch, 3, groups=ch, act="identity"))
            self.head_pw.append(ConvBNAct(ch, out_ch, 1, act="relu"))
        self.out_channels = width

    @staticmethod
    def _transition(c1: int, c2: int, down: bool = False) -> nn.Module:
        if down:
            return nn.Sequential(
                ConvBNAct(c1, c1, 3, 2, groups=c1, act="identity"),
                ConvBNAct(c1, c2, 1, act="relu"),
            )
        return nn.Sequential(
            ConvBNAct(c1, c1, 3, groups=c1, act="identity"), ConvBNAct(c1, c2, 1, act="relu")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_conv(x)
        left, right = x.chunk(2, 1)
        x = torch.cat((self.stem_left(left), self.stem_right(right)), 1)
        b1, b2 = self.transition0[0](x), self.transition0[1](x)
        xs = self.stage0([b1, b2])
        xs = [self.transition1[i](xs[-1] if i >= len(xs) else xs[i]) for i in range(3)]
        xs = self.stage1(xs)
        xs = [self.transition2[i](xs[-1] if i >= len(xs) else xs[i]) for i in range(4)]
        xs = self.stage2(xs)
        # Official iterative head fuses coarse semantic maps back into the 1/4-resolution branch.
        rev = xs[::-1]
        last = None
        projected = []
        for i, feat in enumerate(rev):
            if last is not None:
                last = F.interpolate(last, size=feat.shape[-2:], mode="bilinear", align_corners=True)
                feat = feat + last
            last = self.head_pw[i](self.head_dw[i](feat))
            projected.append(last)
        return projected[-1]
