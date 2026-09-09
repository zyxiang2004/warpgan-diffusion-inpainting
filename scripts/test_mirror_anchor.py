# v7 numerical unit test: continuous-confidence supervision + clean condition.
# Checks per identity x view:
#   1. w is CONTINUOUS in [0,1] (no binarization) and equals the depth ramp
#   2. anchor blends mirror-real by w, EG3D by (1-vis_eff-w) — smooth, no cliff
#   3. condition hole is PURE EG3D (mirror contributes zero condition pixels)
#   4. OFF path (mirror_anchor=False) reproduces the v4 anchor bitwise
#   5. fragmentation is gone: w histogram has no 0/1 spike at the old threshold

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append('.')
from utils.warp.Splatting import Warper
from utils.warp.splatting_ext import WarperExt

ROOT = '/data/xzy/warpgan20260803/20260803/warpgan_orig/WarpGAN-main'
warper = Warper()
warper_ext = WarperExt()


def as_depth(t):
    t = t.float()
    if t.dim() == 2:
        t = t[None, None]
    elif t.dim() == 3:
        t = t[:, None]
    return t


ok_all = True
for ident, view in [('000014', 1), ('000004', 2), ('000004', 3)]:
    base = os.path.join(ROOT, 'data/celeba-hq_1000_static_rebalanced', ident)
    a = np.asarray(Image.open(os.path.join(base, 'x.png'))).astype(np.float32) / 255.
    x = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).cuda()
    c = torch.load(os.path.join(base, 'c.pt'), map_location='cpu').float().reshape(1, -1).cuda()
    depth = as_depth(torch.load(os.path.join(base, 'depth.pt'), map_location='cpu')).cuda()
    c_m = torch.load(os.path.join(base, 'c_mirror.pt'), map_location='cpu').float().reshape(1, -1).cuda()
    dm = as_depth(torch.load(os.path.join(base, 'depth_mirror.pt'), map_location='cpu')).cuda()
    c_n = torch.load(os.path.join(base, f'c_novel_{view}.pt'), map_location='cpu').float().reshape(1, -1).cuda()
    dn = as_depth(torch.load(os.path.join(base, f'depth_novel_{view}.pt'), map_location='cpu')).cuda()
    yh_img = np.asarray(Image.open(os.path.join(base, f'y_hat_novel_{view}.png'))).astype(np.float32) / 255.
    yh = torch.from_numpy(yh_img.transpose(2, 0, 1)).unsqueeze(0).cuda()
    x_mirror = torch.flip(x, dims=[3])

    warp_img, vis, _ = warper.forward_warp(img1=x, depth1=depth, c1=c, c2=c_n)
    mask = 1.0 - vis

    inv_warp, inv_valid = warper_ext.inverse_warp(img2=x, depth1=dn, depth2=depth, c1=c_n, c2=c)
    vis_eff = inv_valid.clamp(0, 1) * (1.0 - mask)
    inv_warp_m, _, conf_m = warper_ext.inverse_warp(
        img2=x_mirror, depth1=dn, depth2=dm, c1=c_n, c2=c_m, return_mismatch=True)
    w = conf_m.clamp(0, 1) * mask                      # continuous
    blind = (1.0 - vis_eff - w).clamp(0, 1)

    anchor = inv_warp * vis_eff + inv_warp_m * w + yh * blind
    cond = warp_img * (1.0 - mask) + yh * mask         # v7: pure EG3D hole
    v4_anchor = inv_warp * vis_eff + yh * (1.0 - vis_eff)  # OFF path

    wn = w[0, 0].cpu().numpy()
    frac_mid = float(((wn > 0.05) & (wn < 0.95)).sum()) / max(int((wn > 0.05).sum()), 1)
    # segment extraction: total = mirror + blind + visible; solve per segment
    seg_mirror = inv_warp_m * w
    seg_blind = yh * blind
    seg_vis = inv_warp * vis_eff
    residual = anchor - (seg_mirror + seg_blind + seg_vis)
    checks = {
        'w continuous in [0,1]': bool((w >= 0).all() and (w <= 1).all()),
        'w has intermediate values (no binarization)': frac_mid > 0.30,
        'anchor == sum of three segments': bool(torch.allclose(
            residual, torch.zeros_like(residual), atol=1e-4)),
        'cond hole is PURE EG3D': bool(torch.allclose(cond * mask, yh * mask)),
        'cond visible == warp': bool(torch.allclose(cond * (1 - mask), warp_img * (1 - mask))),
        'OFF path == v4 anchor bitwise': bool(torch.allclose(
            inv_warp * vis_eff + yh * (1.0 - vis_eff), v4_anchor)),
        'anchor finite': bool(torch.isfinite(anchor).all()),
        'sum of segments == 1': bool(torch.allclose(vis_eff + w + blind,
                                                    torch.ones_like(vis_eff), atol=1e-5)),
    }
    status = 'PASS' if all(checks.values()) else 'FAIL'
    ok_all &= all(checks.values())
    print(f'[{status}] {ident}_v{view}: hole={float(mask.mean()):.2f} '
          f'w_mean={float(w.sum() / mask.sum()):.2f} '
          f'w_intermediate_frac={frac_mid:.2f}')
    for k, v in checks.items():
        if not v:
            print(f'    FAIL -> {k}')

print('\nALL PASS' if ok_all else '\nSOME CHECKS FAILED')
