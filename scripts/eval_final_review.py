# Final review (spec §8 25K terminal audit): FIXED identities x FIXED views.
#
# The orig dataset samples novel_view = random.randint(1,3) on every load, so
# in-training val points test DIFFERENT views (hole fractions 0.57/0.37/0.22).
# This script reads the test-set files directly (bypassing the dataset class)
# so every identity is evaluated on ALL THREE views with seed=42 — comparable,
# and per-view metrics expose the large-hole (large-yaw) weakness directly.
#
# Usage (repo root):
#   python scripts/eval_final_review.py [ckpt]
#   (default ckpt = step1b iteration_0024999.pt)

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

coach = Coach(opts)   # __init__ -> resume() from the checkpoint
coach.denoising_unet.eval()
dev = coach.device

TEST_ROOT = str(opts.paths.dataset.test)
ids = sorted(d for d in os.listdir(TEST_ROOT) if d.isdigit())[:3]
VIEWS = [1, 2, 3]
print(f'[final review] identities={ids} views={VIEWS} seed=42 ckpt_step={coach.global_step}')

OUT = os.path.join(EXP, 'final_review')
os.makedirs(OUT, exist_ok=True)


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


@torch.no_grad()
def evaluate(ident, view):
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

    gen = coach._sample_novel(cond, mask, codes, [y_hat_novel], ['y_hat_novel'], seed=42)

    l1 = (gen - anchor).abs().mean(dim=1, keepdim=True)
    m = {
        'hole_frac': float(mask.mean()),
        'full': float(l1.mean()),
        'hole': float((l1 * mask).sum() / (mask.sum() + 1e-6)),
        'visible': float((l1 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-6)),
    }
    panel = torch.cat([x[0], y_hat_novel[0], cond[0], mask[0].expand(3, -1, -1),
                       anchor[0], gen[0]], dim=2).cpu()
    return m, panel, gen[0].cpu()


rows, lines = [], ['ident    view  hole_frac   full     hole   visible']
for ident in ids:
    gens = []
    for v in VIEWS:
        m, panel, gen = evaluate(ident, v)
        gens.append(gen)
        lines.append(f'{ident}   v{v}    {m["hole_frac"]:.3f}    '
                     f'{m["full"]:.4f}  {m["hole"]:.4f}  {m["visible"]:.4f}')
        vutils.save_image(panel, os.path.join(OUT, f'review_{ident}_v{v}.png'))
    rows.append(torch.cat(gens, dim=2))

grid = torch.cat([r.unsqueeze(0) for r in rows], dim=0)
vutils.save_image(grid, os.path.join(OUT, 'overview_gen_3x3.png'))

report = os.path.join(OUT, 'metrics.txt')
with open(report, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'\n[final review] panels -> {OUT}')
print(f'[final review] 3x3 generated-view grid -> {OUT}/overview_gen_3x3.png')
