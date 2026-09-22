import torch
import torch.nn as nn


class RefNetBrushnetFusion(nn.Module):
    """Phase-2 plugin: fuse ReferenceNet features into BrushNet's residual
    stream via zero-initialized 1x1 convs (per matched scale).

    Zero-init guarantees step-0 output == the pure-BrushNet baseline exactly,
    so the plugin can be hot-swapped onto a trained Phase-1 checkpoint without
    perturbing it; training then gradually learns to read the source-photo
    features. This is the ControlNet "add-a-condition" convention applied to
    feature fusion — no attention surgery, single-writer preserved (everything
    still enters SD through BrushNet's add_res only).

    Channel matching (RefNet output scales vs BrushNet residual stream):
      scale 0: 320ch @ 64x64   scale 1: 640ch @ 32x32
      scale 2: 1280ch @ 16x16  scale 3: 1280ch @ 8x8
    """

    def __init__(self, channels=(320, 640, 1280, 1280)):
        super().__init__()
        self.projs = nn.ModuleList(
            [nn.Conv2d(c, c, kernel_size=1) for c in channels]
        )
        for conv in self.projs:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, brushnet_res_samples, refnet_feats):
        """brushnet_res_samples: tuple of residual tensors from BrushNet down
        path (the same tuples that feed add_res). refnet_feats: dict from
        ReferenceNet.extract_features — {ch: [tensors]} keyed by channel count.

        Returns new tuples with the (zero-init) ref features added per scale.
        """
        out = []
        for sample in brushnet_res_samples:
            ch = sample.shape[1]
            candidates = refnet_feats.get(ch) if refnet_feats else None
            if not candidates:
                out.append(sample)
                continue
            feat = candidates[0] if isinstance(candidates, (list, tuple)) else candidates
            if feat.shape[-2:] != sample.shape[-2:]:
                feat = nn.functional.interpolate(
                    feat, size=sample.shape[-2:], mode="bilinear", align_corners=False
                )
            proj = None
            for p in self.projs:
                if p.in_channels == ch:
                    proj = p
                    break
            if proj is None:
                out.append(sample)
                continue
            out.append(sample + proj(feat.to(sample.dtype)))
        return tuple(out)
