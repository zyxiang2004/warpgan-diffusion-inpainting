# Step-2 failure diagnosis on the FROZEN step2_mirror@31999 checkpoint.
#
# Training with mirror ON degraded output (hole 0.036 -> ~0.10 stuck; user
# eyeball: much worse). Gate/adapters barely moved (0.435 -> 0.435), so the
# injection itself (untrained-use tokens) is the suspected culprit, NOT the
# adapter weights. Variants on the SAME checkpoint, same id/view/seed:
#   A as_trained   : mirror tokens ON  (reproduces the degraded output)
#   B no_mirror    : mirror tokens OFF at inference (RefNet = y_hat_novel only)
#   C no_refnet    : RefNet branch fully OFF
#   D + gate0      : mirror ON but reference gate forced to 0 (=C effectively)
# If B ~= step1b baseline quality -> the DAMAGE lives in BrushNet's trained
# weights; if A >> B on this ckpt -> the damage is the injected tokens.

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
CKPT = os.path.join(BASE, 'step2_mirror_15k/checkpoints/iteration_0031999.pt')
BASE_CKPT = os.path.join(BASE, '[20260823-180346]_step1b_epsonly_25k/checkpoints/iteration_0024999.pt')
OUT = os.path.join(BASE, 'step2_mirror_15k', 'failure_diagnosis')
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


@torch.no_grad()
def sample_variant(coach, s, mode):
    x, codes, mask, cond, anchor, yh = s
    x_mirror = torch.flip(x, dims=[3])
    if mode == 'A_as_trained':
        refs, tags = [yh, x_mirror], ['y_hat_novel', 'x_mirror']
    elif mode == 'B_no_mirror':
        refs, tags = [yh], ['y_hat_novel']
    else:  # C_no_refnet
        refs, tags = [], []
    return coach._sample_novel(cond, mask, codes, refs, tags, seed=42)


lines = []
coach = build_coach(CKPT)
print(f'[diag] loaded step2@{coach.global_step} (mirror={coach.use_mirror})')
rows_all = []
for ident, view in SAMPLES:
    s = load_sample(ident, view, coach)
    x, mask, anchor = s[0], s[3 - 2], s[4]  # x, mask, anchor
    mask = s[2]
    rows = []
    for mode in ['A_as_trained', 'B_no_mirror', 'C_no_refnet']:
        gen = sample_variant(coach, s, mode)
        l1 = (gen - anchor).abs().mean(dim=1, keepdim=True)
        hole = float((l1 * mask).sum() / (mask.sum() + 1e-6))
        vis_ = float((l1 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-6))
        lines.append(f'{ident}_v{view}  {mode:14s} hole={hole:.4f} visible={vis_:.4f}')
        rows.append(gen[0].cpu())
    header = torch.cat([x[0].cpu(), s[5][0].cpu(), s[3][0].cpu(), mask[0].expand(3, -1, -1).cpu(),
                        anchor[0].cpu(), rows[0]], dim=2)
    grid = header.unsqueeze(0)
    for r in rows[1:]:
        blank = torch.zeros(3, r.shape[-2], header.shape[-1] - r.shape[-1])
        grid = torch.cat([grid, torch.cat([blank, r], dim=2).unsqueeze(0)], dim=0)
    vutils.save_image(grid, os.path.join(OUT, f'diag_{ident}_v{view}.png'))
    rows_all.append(rows)

del coach
torch.cuda.empty_cache()

# step1b baseline reference gens for the same samples (bottom comparison row)
coach = build_coach(BASE_CKPT)
ref_lines = []
i = 0
for ident, view in SAMPLES:
    s = load_sample(ident, view, coach)
    gen = coach._sample_novel(s[3], s[2], s[1], [s[5]], ['y_hat_novel'], seed=42)
    l1 = (gen - s[4]).abs().mean(dim=1, keepdim=True)
    hole = float((l1 * s[2]).sum() / (s[2].sum() + 1e-6))
    vis_ = float((l1 * (1 - s[2])).sum() / ((1 - s[2]).sum() + 1e-6))
    lines.append(f'{ident}_v{view}  REF_step1b     hole={hole:.4f} visible={vis_:.4f}')
    i += 1
del coach

with open(os.path.join(OUT, 'diag_metrics.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'\n[diag] saved -> {OUT}')
