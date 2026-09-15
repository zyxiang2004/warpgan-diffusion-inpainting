# Step-2 v2 S-curve-distortion causal test on FROZEN step2v2@27999.
#
# User reports GLOBAL regular S-shaped warping (not hole-local). Two suspects:
#   H1: the tiny-but-growing mirror injection mixes MIRRORED geometry into
#       generation (attention retrieves flipped-content features);
#   H2: the trained BrushNet weights themselves drifted (training dynamics,
#       e.g. optimizer-state reset), independent of injection.
# Same id/view/seed on the same checkpoint:
#   A as_trained   : mirror on (reproduces the S-warp if H1)
#   B mirror_off   : no extra features at inference
#   C src_extra    : extra branch fed the UN-MIRRORED source photo
#                    (H1 refines: mirror CONTENT vs any extra content)
#   D ref_step1b   : baseline 24999 model (pre-v2 reference)
# If A distorted but B clean        -> H1 (injection causes it)
# If A == B distorted, D clean      -> H2 (weights drifted)
# If A & C distorted but B clean    -> extra-branch usage itself, geometry-agnostic

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append('.')
sys.path.append('..')

from omegaconf import OmegaConf
import torchvision.utils as vutils

BASE = '/data/xzy/warpgan20260803/20260803/warpgan_orig/WarpGAN-main/experiments/train_inpainting_diffusion'
CKPT = os.path.join(BASE, 'step2v2_mirror_gate0_10k/checkpoints/iteration_0027999.pt')
BASE_CKPT = os.path.join(BASE, '[20260823-180346]_step1b_epsonly_25k/checkpoints/iteration_0024999.pt')
OUT = os.path.join(BASE, 'step2v2_mirror_gate0_10k', 'scurve_diagnosis')
os.makedirs(OUT, exist_ok=True)


def build_coach(ckpt):
    exp = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    cfg = os.path.join(exp, 'config.yaml')
    if not os.path.exists(cfg):
        cfg = os.path.join(exp, 'config_resume.yaml')
    opts = OmegaConf.load(cfg)
    opts.checkpoint_path = os.path.abspath(ckpt)
    opts.exp_dir = exp
    opts.smoke.check = False
    from training.coach_inpainting_diffusion import Coach
    return Coach(opts)


def as_depth(t):
    t = t.float()
    if t.dim() == 2:
        t = t[None, None]
    elif t.dim() == 3:
        t = t[:, None]
    return t


root = './data/celeba-hq_1000_static_rebalanced'
SAMPLES = [('000004', 2), ('000009', 3), ('000014', 1)]
LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def load_sample(ident, view, coach):
    base = os.path.join(root, ident)

    def img(name):
        a = np.asarray(Image.open(os.path.join(root, ident, name))).astype(np.float32) / 255.
        return torch.from_numpy(a.transpose(2, 0, 1))

    dev = coach.device
    x = img('x.png').unsqueeze(0).to(dev)
    c = torch.load(os.path.join(base, 'c.pt'), map_location='cpu').float().reshape(1, -1).to(dev)
    codes = torch.load(os.path.join(base, 'codes.pt'), map_location='cpu').float()
    codes = codes.reshape(1, *codes.shape[-2:]).to(dev)
    depth = as_depth(torch.load(os.path.join(base, 'depth.pt'), map_location='cpu')).to(dev)
    c_n = torch.load(os.path.join(base, f'c_novel_{view}.pt'), map_location='cpu').float().reshape(1, -1).to(dev)
    yh = img(f'y_hat_novel_{view}.png').unsqueeze(0).to(dev)
    dn = as_depth(torch.load(os.path.join(base, f'depth_novel_{view}.pt'), map_location='cpu')).to(dev)
    warp_img, vis, _ = coach.warper.forward_warp(img1=x, depth1=depth, c1=c, c2=c_n)
    mask = 1.0 - vis
    cond = warp_img * (1.0 - mask) + yh * mask
    inv, iv = coach.warper_ext.inverse_warp(img2=x, depth1=dn, depth2=depth, c1=c_n, c2=c)
    ve = iv.clamp(0, 1) * (1.0 - mask)
    anchor = inv * ve + yh * (1.0 - ve)
    return x, codes, mask, cond, anchor, yh


lines = []
coach = build_coach(CKPT)
print(f'[scurve] step2v2@{coach.global_step} mirror={coach.use_mirror}')
rows_all = []
for ident, view in SAMPLES:
    x, codes, mask, cond, anchor, yh = load_sample(ident, view, coach)
    x_mirror = torch.flip(x, dims=[3])
    variants = [
        ('A_as_trained', [yh, x_mirror]),
        ('B_mirror_off', [yh]),
        ('C_src_extra', [yh, x]),
    ]
    rows = []
    for name, refs in variants:
        tags = ['y_hat_novel'] + [f'extra_{i}' for i in range(len(refs) - 1)]
        gen = coach._sample_novel(cond, mask, codes, refs, tags, seed=42)
        l1 = (gen - anchor).abs().mean(dim=1, keepdim=True)
        hole = float((l1 * mask).sum() / (mask.sum() + 1e-6))
        vis_ = float((l1 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-6))
        lines.append(f'{ident}_v{view}  {name:14s} hole={hole:.4f} visible={vis_:.4f}')
        rows.append(gen[0].cpu())
    header = torch.cat([x[0].cpu(), yh[0].cpu(), cond[0].cpu(),
                        mask[0].expand(3, -1, -1).cpu(), anchor[0].cpu(), rows[0]], dim=2)
    grid = header.unsqueeze(0)
    for r in rows[1:]:
        blank = torch.zeros(3, r.shape[-2], header.shape[-1] - r.shape[-1])
        grid = torch.cat([grid, torch.cat([blank, r], dim=2).unsqueeze(0)], dim=0)
    vutils.save_image(grid, os.path.join(OUT, f'scurve_{ident}_v{view}.png'))

del coach
torch.cuda.empty_cache()

coach = build_coach(BASE_CKPT)
for ident, view in SAMPLES:
    x, codes, mask, cond, anchor, yh = load_sample(ident, view, coach)
    gen = coach._sample_novel(cond, mask, codes, [yh], ['y_hat_novel'], seed=42)
    l1 = (gen - anchor).abs().mean(dim=1, keepdim=True)
    hole = float((l1 * mask).sum() / (mask.sum() + 1e-6))
    vis_ = float((l1 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-6))
    lines.append(f'{ident}_v{view}  D_ref_step1b   hole={hole:.4f} visible={vis_:.4f}')

with open(os.path.join(OUT, 'scurve_metrics.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'\n[scurve] saved -> {OUT}')
