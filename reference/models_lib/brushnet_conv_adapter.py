"""Convolutional LoRA-style side branches for the (attention-free) BrushNet.

BrushNet in this repository is a pure conv/resnet conditioning network with no
attention layers, so the classical linear LoRA does not apply. This module
provides the convolutional analogue with the same guarantees:

- one side branch per targeted Conv2d: 1x1 down-projection (rank r),
  a 3x3 depthwise spatial mix, and a 1x1 up-projection;
- the up-projection is zero-initialised, so the modified network is exactly
  equivalent to the frozen backbone at step 0;
- only the side branches carry gradients; the frozen conv weights are never
  touched, so a failure is attributable to this single variable.
"""

import torch
from torch import nn


class ConvSideBranch(nn.Module):
    def __init__(self, channels, rank=8, kernel=3):
        super().__init__()
        self.down = nn.Conv2d(channels, rank, 1, bias=False)
        self.spatial = nn.Conv2d(
            rank, rank, kernel, padding=kernel // 2, groups=rank, bias=False
        )
        self.up = nn.Conv2d(rank, channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.spatial(self.down(x)))


class BrushNetConvAdapter(nn.Module):
    """Wrap selected BrushNet convs with additive side branches."""

    def __init__(self, brushnet, rank=8, targets=("conv1", "conv2")):
        super().__init__()
        self.branches = nn.ModuleDict()
        self.targets = set(targets)
        for name, module in brushnet.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            leaf = name.split(".")[-1]
            if leaf not in self.targets:
                continue
            if not (name.startswith("down_blocks") or name.startswith("mid_block")):
                continue
            safe_key = name.replace(".", "_")
            self.branches[safe_key] = ConvSideBranch(module.out_channels, rank=rank)
        self._hooks = []
        for name, module in brushnet.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            leaf = name.split(".")[-1]
            if leaf not in self.targets:
                continue
            if not (name.startswith("down_blocks") or name.startswith("mid_block")):
                continue
            safe_key = name.replace(".", "_")

            def hook(mod, inputs, output, key=safe_key):
                return output + self.branches[key](output)

            self._hooks.append(module.register_forward_hook(hook))

    def forward(self, x):
        return x
