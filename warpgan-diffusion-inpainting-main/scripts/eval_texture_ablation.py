# Texture/domain ablation on the FROZEN step1b 24999 checkpoint.
#
# Motivated by the user's eyeball review of final_review:
#   000009_v3 (54% hole): face broken/twisted
#   000004_v2 (57% hole): natural-but-EG3D-like skin texture (oil-paint)
#   000014_v3: good (control)   000014_v1: pinhole (white dot) in hair
#
# Hypothesis: the hole's "texture teacher" is almost entirely EG3D
# (synth target = render, real weak anchor hole@0.1), while the visible region
# is real-photo domain -> (a) oil-paint hole texture, (b) mask-like seam.
# This script tests INFERENCE-SIDE fixes that need no training:
#   A baseline       : RefNet input = y_hat_novel (as trained)
#   B refnet_src     : RefNet input = source photo x  (teacher① / ablation #2)
#   C refnet_mirror  : RefNet = y_hat_novel + x_mirror parallel (teacher④)
#   D hole_lowpass   : condition hole filled with LOWPASS render (copy-prior
#                      can only copy color/structure, texture left to SD+RefNet)
#   E lowpass+mirror : D + C combined
# Also: pinhole diagnostics (dark/bright connected components) for 000014_v1.

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append('.')
sys.path.append('..')

from omegaconf import OmegaConf
import torchvision.utils as vutils

from training.coach_inpainting_diffusion import Coach

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    './experiments/train_inpainting_diffusion/[20260823-180346]_step1b_epsonly_25k/checkpoints/iteration_0024999.pt'
CKPT = os.path.abspath(CKPT)
EXP = os.path.dirname(os.path.dirname(CKPT))
opts = OmegaConf.load(os.path.join(EXP, 'config.yaml'))
opts.checkpoint_path = CKPT
opts.exp_dir = EXP
opts.smoke.check = False

coach = Coach(opts)
coach.denoising_unet.eval()
dev = coach.device
TEST_ROOT = str(opts.paths.dataset.test)
OUT = os.path.join(EXP, 'final_review')
os.makedirs(OUT, exist_ok=True)

SAMPLES = [('000009', 3), ('000004', 2), ('000014', 3), ('000014', 1)]
LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def lap_energy(img, m):
    g = img.mean(dim=1, keepdim=True)
    l = F.conv2d(F.pad(g, (1, 1, 1, 1), mode='replicate'), LAP.to(img))
    return float((l.abs() * m).sum() / (m.sum() + 1e-6))


def as_depth(t):
    t = t.float()
    if t.dim() == 2:
        t = t[None, None]
    elif t.dim() == 3:
        t = t[:, None]
    return t


def load_img(ident, name):
    a = np.asarray(Image.open(os.path.join(TEST_ROOT, ident, name))).astype(np.float32) / 255.0
    return torch.from_numpy(a.transpose(2, 0, 1))


def load_sample(ident, view):
    base = os.path.join(TEST_ROOT, ident)
    x = load_img(ident, 'x.png').unsqueeze(0).to(dev)
    c = torch.load(os.path.join(base, 'c.pt'), map_location='cpu').float().reshape(1, -1).to(dev)
    codes = torch.load(os.path.join(base, 'codes.pt'), map_location='cpu').float()
    codes = codes.reshape(1, *codes.shape[-2:]).to(dev)
    depth = as_depth(torch.load(os.path.join(base, 'depth.pt'), map_location='cpu')).to(dev)
    c_novel = torch.load(os.path.join(base, f'c_novel_{view}.pt'), map_location='cpu').float().reshape(1, -1).to(dev)
    y_hat_novel = load_img(ident, f'y_hat_novel_{view}.png').unsqueeze(0).to(dev)
    depth_novel = as_depth(torch.load(os.path.join(base, f'depth_novel_{view}.pt'), map_location='cpu')).to(dev)
    warp_img, vis_mask, _ = coach.warper.forward_warp(img1=x, depth1=depth, c1=c, c2=c_novel)
    mask = 1.0 - vis_mask
    cond = warp_img * (1.0 - mask) + y_hat_novel * mask
    inv, inv_valid = coach.warper_ext.inverse_warp(img2=x, depth1=depth_novel, depth2=depth, c1=c_novel, c2=c)
    vis_eff = inv_valid.clamp(0, 1) * (1.0 - mask)
    anchor = inv * vis_eff + y_hat_novel * (1.0 - vis_eff)
    return dict(x=x, codes=codes, mask=mask, cond=cond, anchor=anchor, y_hat_novel=y_hat_novel)


VARIANTS = ['A_baseline', 'B_refnet_src', 'C_refnet_mirror', 'D_hole_lowpass', 'E_lowpass+mirror']
lines = ['sample        variant             full     hole   visible  hole_lap  (x_lap / anchor_hole_lap)']
pinhole_note = []

for ident, view in SAMPLES:
    s = load_sample(ident, view)
    x, mask, cond, anchor = s['x'], s['mask'], s['cond'], s['anchor']
    y_hat_novel, codes = s['y_hat_novel'], s['codes']
    x_mirror = torch.flip(x, dims=[3])
    lowpass = F.avg_pool2d(y_hat_novel, kernel_size=31, stride=1, padding=15)
    cond_lp = cond * (1.0 - mask) + lowpass * mask

    x_lap = lap_energy(x, mask)
    anchor_hole_lap = lap_energy(anchor, mask)

    rows = []
    for v in VARIANTS:
        if v == 'A_baseline':
            c_img, refs, tags = cond, [y_hat_novel], ['y_hat_novel']
        elif v == 'B_refnet_src':
            c_img, refs, tags = cond, [x], ['x_src']
        elif v == 'C_refnet_mirror':
            c_img, refs, tags = cond, [y_hat_novel, x_mirror], ['y_hat_novel', 'x_mirror']
        elif v == 'D_hole_lowpass':
            c_img, refs, tags = cond_lp, [y_hat_novel], ['y_hat_novel']
        else:
            c_img, refs, tags = cond_lp, [y_hat_novel, x_mirror], ['y_hat_novel', 'x_mirror']
        gen = coach._sample_novel(c_img, mask, codes, refs, tags, seed=42)
        l1 = (gen - anchor).abs().mean(dim=1, keepdim=True)
        m = {'full': float(l1.mean()),
             'hole': float((l1 * mask).sum() / (mask.sum() + 1e-6)),
             'visible': float((l1 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-6)),
             'hole_lap': lap_energy(gen, mask)}
        lines.append(f'{ident}_v{view}  {v:18s}  {m["full"]:.4f}  {m["hole"]:.4f}  '
                     f'{m["visible"]:.4f}  {m["hole_lap"]:.4f}   ({x_lap:.4f} / {anchor_hole_lap:.4f})')
        rows.append(gen[0].cpu())

    header = torch.cat([x[0].cpu(), s['y_hat_novel'][0].cpu(), cond[0].cpu(),
                        mask[0].expand(3, -1, -1).cpu(), anchor[0].cpu(), rows[0]], dim=2)
    grid = header.unsqueeze(0)
    for r in rows[1:]:
        blank = torch.zeros(3, r.shape[-2], header.shape[-1] - r.shape[-1])
        grid = torch.cat([grid, torch.cat([blank, r], dim=2).unsqueeze(0)], dim=0)
    vutils.save_image(grid, os.path.join(OUT, f'texture_ablation_{ident}_v{view}.png'))

    if (ident, view) == ('000014', 1):
        gen = rows[0]  # baseline variant
        try:
            from scipy import ndimage
            hole = mask[0, 0].cpu().numpy() > 0.5
            g = gen.mean(dim=0).numpy()
            dark = g < 0.08
            bright = (g > 0.97) & hole
            lab_a, n_d = ndimage.label(dark)
            lab_b, n_b = ndimage.label(bright)
            dark_sizes = sorted(np.bincount(lab_a.ravel())[1:], reverse=True)[:8] if n_d > 1 else []
            bright_sizes = sorted(np.bincount(lab_b.ravel())[1:], reverse=True)[:8] if n_b > 1 else []
            lab_m, n_m = ndimage.label(hole)
            mask_sizes = sorted(np.bincount(lab_m.ravel())[1:], reverse=True)[:8] if n_m > 1 else []
            pinhole_note.append(
                f'000014_v1 pinhole diagnostics:\n'
                f'  generation: dark_px_ratio={float(dark.mean()):.4f}, '
                f'bright_in_hole_px={int(bright.sum())}, '
                f'bright_small_components(<=16px)={sum(1 for s_ in bright_sizes if s_ <= 16)}\n'
                f'  bright component sizes: {bright_sizes}\n'
                f'  dark component sizes: {dark_sizes}\n'
                f'  CONDITION mask components (splat fragmentation): n={n_m}, sizes={mask_sizes}')
        except ImportError:
            pinhole_note.append('scipy unavailable - skipped connected-component analysis')

report = os.path.join(OUT, 'texture_ablation_metrics.txt')
with open(report, 'w') as f:
    f.write('\n'.join(lines) + '\n\n' + '\n'.join(pinhole_note) + '\n')
print('\n'.join(lines))
print()
print('\n'.join(pinhole_note))
print(f'\nsaved -> {OUT}/texture_ablation_*.png')

