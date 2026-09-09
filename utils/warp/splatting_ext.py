# Diffusion-port extension of the ORIGINAL Warper (utils/warp/Splatting.py).
#
# The original file is kept untouched (faithful-port base); this subclass only
# adds `inverse_warp` — a line-faithful port of the MAIN project's
# utils/warp/Splatting.py::Warper.inverse_warp (validated v2 contract:
# streak-free grid-sample back-projection, exactly one sample per target
# pixel; internal lap 0.028 ≈ source photo 0.037).
#
# Used to build the clean visible-region epsilon anchor for the real-batch
# novel pass (ORIG_FAITHFUL_PORT_SPEC.md §5.3).

from typing import Tuple

import torch
import torch.nn.functional as F

from utils.warp.Splatting import Warper


class WarperExt(Warper):

    def inverse_warp(self, img2: torch.Tensor, depth1: torch.Tensor,
                     depth2: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor,
                     depth_eps: float = 6e-2, return_mismatch: bool = False):
        """Backward warp: render view-1 pixels by grid-sampling img2 (view 2).

        For every view-1 (target) pixel with depth1, unproject, reproject into
        view 2, and bilinearly sample img2 there. Unlike forward_warp's
        splatting this is streak-free (exactly one sample per target pixel).

        v7: return_mismatch=True yields a CONTINUOUS validity field
        clamp(1 - |depth2_sampled - z| / depth_eps, 0, 1) * in_bounds instead
        of the binary verdict — the threshold becomes an attenuation scale
        (no cliff -> no salt-and-pepper fragmentation; see STEP0 §8.18).

        Geometry conventions are identical to forward_warp's defaults
        (pixel-unit intrinsics f=4.2647*res, cam2world from c[:, :16]).
        Returns (warped, valid[, mismatch]) with valid in [0,1].
        """
        b, c, h, w = img2.shape
        if depth1.shape[2:4] != (h, w):
            depth1 = F.interpolate(depth1, size=(h, w), mode='bicubic', align_corners=False)
        if depth2.shape[2:4] != (h, w):
            depth2 = F.interpolate(depth2, size=(h, w), mode='bicubic', align_corners=False)
        ex1 = c1[:, :16].view(b, 4, 4)  # cam2world of view 1 (target/novel)
        ex2 = c2[:, :16].view(b, 4, 4)  # cam2world of view 2 (source)
        intrinsic = torch.tensor(
            [[4.2647 * w, 0, 0.5 * w], [0, 4.2647 * h, 0.5 * h], [0, 0, 1]]
        ).unsqueeze(0).repeat(b, 1, 1).to(img2)
        transformation = torch.linalg.inv(ex2) @ ex1  # view-1 cam -> view-2 cam

        x1d = torch.arange(0, w)[None].to(img2)
        y1d = torch.arange(0, h)[:, None].to(img2)
        x2d = x1d.repeat([h, 1])
        y2d = y1d.repeat([1, w])
        ones_2d = torch.ones_like(x2d)
        pos_vectors = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]  # (1,h,w,3,1)
        inv_k = torch.linalg.inv(intrinsic)[:, None, None]  # (b,1,1,3,3)
        rays = torch.matmul(inv_k, pos_vectors).squeeze(-1)  # (b,h,w,3)
        pts_view1 = rays * depth1[:, 0][:, :, :, None]  # (b,h,w,3)
        pts_homo = torch.cat(
            [pts_view1, torch.ones_like(pts_view1[..., :1])], dim=-1
        )  # (b,h,w,4)
        pts_view2 = torch.matmul(
            transformation[:, None, None], pts_homo.unsqueeze(-1)
        ).squeeze(-1)[..., :3]  # (b,h,w,3)
        proj = torch.matmul(
            intrinsic[:, None, None], pts_view2.unsqueeze(-1)
        ).squeeze(-1)  # (b,h,w,3)
        z = proj[..., 2].clamp(min=1e-6)
        u = (proj[..., 0] / z / w * 2 + 1.0 / w - 1.0)  # align_corners=False norm
        v = (proj[..., 1] / z / h * 2 + 1.0 / h - 1.0)
        grid = torch.stack([u, v], dim=-1)  # (b,h,w,2)

        in_bounds = (
            (grid[..., 0] > -1) & (grid[..., 0] < 1)
            & (grid[..., 1] > -1) & (grid[..., 1] < 1)
        ).float()
        warped = F.grid_sample(img2, grid, align_corners=False)
        depth2_sampled = F.grid_sample(depth2, grid, align_corners=False)[:, 0]
        depth_ok = ((depth2_sampled - z).abs() < depth_eps).float() * (z > 0).float()
        valid = (in_bounds * depth_ok).unsqueeze(1)  # (b,1,h,w)
        if return_mismatch:
            # continuous confidence: ramp from 1 (exact depth match) to 0 (at
            # the eps threshold); still zero out-of-bounds / non-positive-z
            conf = (1.0 - (depth2_sampled - z).abs() / depth_eps).clamp(0, 1) \
                * in_bounds * (z > 0).float()
            return warped, valid, conf.unsqueeze(1)
        return warped, valid
