# -*- coding: utf-8 -*-
"""Anchor median-cleanup preview (v18.7). Standalone — loads ONLY the warpers
(~3G GPU), replicates Coach._build_novel_view verbatim (mirror_anchor=True
path), and renders [photo | render | mask | anchor_raw | anchor_med3 |
anchor_med5] plus 2x hole-centroid crops for visual verification BEFORE any
training deployment. Run: CUDA_VISIBLE_DEVICES=0 python scripts/preview_anchor_cleanup.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torchvision.utils as vutils
from kornia.filters import median_blur
from utils.warp.Splatting import Warper
from utils.warp.splatting_ext import WarperExt

DEV = 'cuda:0'
DATA_ROOT = './data/celeba-hq_1000_static_rebalanced'
OUT_FULL = 'train_logs/anchor_cleanup_full.png'
OUT_CROP = 'train_logs/anchor_cleanup_crops.png'
IDS = ['000004', '000550', '001418']
VIEW = 1


def as_depth(t):
    t = t.float()
    if t.dim() == 2:
        t = t[None, None]
    elif t.dim() == 3:
        t = t[:, None]
    return t


def load_case(ident, v):
    base = os.path.join(DATA_ROOT, ident)

    def img(name):
        a = np.asarray(Image.open(os.path.join(base, name))).astype(np.float32) / 255.
        return torch.from_numpy(a.transpose(2, 0, 1))

    x = img('x.png').unsqueeze(0).to(DEV)
    depth = as_depth(torch.load(os.path.join(base, 'depth.pt'), map_location='cpu')).to(DEV)
    c_n = torch.load(os.path.join(base, f'c_novel_{v}.pt'), map_location='cpu').float().reshape(1, -1).to(DEV)
    yh = img(f'y_hat_novel_{v}.png').unsqueeze(0).to(DEV)
    dn = as_depth(torch.load(os.path.join(base, f'depth_novel_{v}.pt'), map_location='cpu')).to(DEV)
    x_mirror = torch.flip(x, dims=[3])
    c_m = torch.load(os.path.join(base, 'c_mirror.pt'), map_location='cpu').float().reshape(1, -1).to(DEV)
    d_m = as_depth(torch.load(os.path.join(base, 'depth_mirror.pt'), map_location='cpu')).to(DEV)
    return dict(x=x, depth=depth, c_n=c_n, yh=yh, dn=dn,
                x_mirror=x_mirror, c_m=c_m, d_m=d_m)


@torch.no_grad()
def build_anchor(case, ident):
    """Coach._build_novel_view (mirror_anchor=True path), verbatim math."""
    x, depth = case['x'], case['depth']
    c = torch.load(os.path.join(DATA_ROOT, ident, 'c.pt'),
                   map_location='cpu').float().reshape(1, -1).to(DEV)
    c_n, yh, dn = case['c_n'], case['yh'], case['dn']
    x_mirror, c_m, d_m = case['x_mirror'], case['c_m'], case['d_m']

    warper = Warper()
    warper_ext = WarperExt()
    _, vis_mask, _ = warper.forward_warp(img1=x, depth1=depth, c1=c, c2=c_n)
    mask = 1.0 - vis_mask
    inv_warp, inv_valid = warper_ext.inverse_warp(
        img2=x, depth1=dn, depth2=depth, c1=c_n, c2=c)
    vis_eff = inv_valid.clamp(0, 1) * (1.0 - mask)
    inv_warp_m, _, conf_m = warper_ext.inverse_warp(
        img2=x_mirror, depth1=dn, depth2=d_m, c1=c_n, c2=c_m, return_mismatch=True)
    w_mirror = conf_m.clamp(0, 1) * mask
    hole_real = w_mirror
    blind = (1.0 - vis_eff - hole_real).clamp(0, 1)
    anchor = inv_warp * vis_eff + inv_warp_m * hole_real + yh * blind
    return anchor, mask


def main():
    rows_full, rows_crop = [], []
    for ident in IDS:
        case = load_case(ident, VIEW)
        anchor, mask = build_anchor(case, ident)
        med3 = median_blur(anchor, (3, 3))
        med5 = median_blur(anchor, (5, 5))
        row = [case['x'][0].cpu(), case['yh'][0].cpu(),
               mask[0].expand(3, -1, -1).cpu(), anchor[0].cpu(),
               med3[0].cpu(), med5[0].cpu()]
        rows_full.append(torch.stack(row, 0))
        # 2x zoom at hole centroid (eval_anchor_compare crop logic)
        m = mask[0, 0]
        ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
        cy, cx = (int(ys.float().mean()), int(xs.float().mean())) if len(ys) else (256, 256)
        r1, c1 = min(512, cy + 80), min(512, cx + 80)
        r0, c0 = r1 - 160, c1 - 160
        crops = []
        for t in (anchor, med3, med5):
            cr = t[0, :, r0:r1, c0:c1].cpu()
            crops.append(F.interpolate(cr.unsqueeze(0), scale_factor=2,
                                       mode='bilinear', align_corners=False)[0])
        rows_crop.append(torch.stack(crops, 0))
        print(f'[anchor-preview] {ident} hole={float(mask.mean()):.2f} done', flush=True)

    g1 = torch.cat(rows_full, 0)
    vutils.save_image(g1, OUT_FULL, nrow=6, padding=2)
    g2 = torch.cat(rows_crop, 0)
    vutils.save_image(g2, OUT_CROP, nrow=3, padding=2)
    print(f'full  -> {OUT_FULL}  (cols: photo|render|mask|anchor_raw|med3|med5)')
    print(f'crops -> {OUT_CROP}  (2x hole zoom: anchor_raw|med3|med5)')


if __name__ == '__main__':
    main()
