# Step-2 A/B: baseline (no mirror) vs mirror-trained, SAME ids x views x seed.
#
# Loads the two checkpoints SEQUENTIALLY (baseline Coach freed before the
# mirror Coach builds — two 618M stacks never coexist on the 24GB card).
# Produces per-sample 7-panel rows (x | y_hat_novel | cond | mask | anchor |
# gen_BASELINE | gen_MIRROR), an overview grid, and a metrics table.

import gc
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
BASE_CKPT = os.path.join(BASE, '[20260823-180346]_step1b_epsonly_25k/checkpoints/iteration_0024999.pt')
MIR_DIR = os.path.join(BASE, '[20260829-234740]_step3_v16_photoref_50k/checkpoints')
if not os.path.isdir(MIR_DIR):
    # addtime2path wraps exp_dir with a [timestamp]_ prefix
    cands_dir = [d for d in os.listdir(BASE) if 'step3_v16_photoref' in d]
    MIR_DIR = os.path.join(BASE, sorted(cands_dir)[-1], 'checkpoints')
cands = sorted(f for f in os.listdir(MIR_DIR) if f.endswith('.pt'))
MIR_CKPT = os.path.join(MIR_DIR, cands[-1])
OUT = os.path.join(os.path.dirname(MIR_DIR), 'ab_review')
os.makedirs(OUT, exist_ok=True)

LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def lap_energy(img, m):
    g = img.mean(dim=1, keepdim=True)
    l = torch.nn.functional.conv2d(
        torch.nn.functional.pad(g, (1, 1, 1, 1), mode='replicate'), LAP.to(img))
    return float((l.abs() * m).sum() / (m.sum() + 1e-6))


def as_depth(t):
    t = t.float()
    if t.dim() == 2:
        t = t[None, None]
    elif t.dim() == 3:
        t = t[:, None]
    return t


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


def sample_all(coach, ids, views, root):
    dev = coach.device
    out = {}
    for ident in ids:
        base = os.path.join(root, ident)

        def img(name):
            a = np.asarray(Image.open(os.path.join(root, ident, name))).astype(np.float32) / 255.
            return torch.from_numpy(a.transpose(2, 0, 1))

        for v in views:
            x = img('x.png').unsqueeze(0).to(dev)
            c = torch.load(os.path.join(base, 'c.pt'), map_location='cpu').float().reshape(1, -1).to(dev)
            codes = torch.load(os.path.join(base, 'codes.pt'), map_location='cpu').float()
            codes = codes.reshape(1, *codes.shape[-2:]).to(dev)
            depth = as_depth(torch.load(os.path.join(base, 'depth.pt'), map_location='cpu')).to(dev)
            c_n = torch.load(os.path.join(base, f'c_novel_{v}.pt'), map_location='cpu').float().reshape(1, -1).to(dev)
            yh = img(f'y_hat_novel_{v}.png').unsqueeze(0).to(dev)
            dn = as_depth(torch.load(os.path.join(base, f'depth_novel_{v}.pt'), map_location='cpu')).to(dev)
            x_mirror = torch.flip(x, dims=[3])
            c_m = torch.load(os.path.join(base, 'c_mirror.pt'), map_location='cpu').float().reshape(1, -1).to(dev)
            d_m = as_depth(torch.load(os.path.join(base, 'depth_mirror.pt'), map_location='cpu')).to(dev)
            # v6: use the coach's OWN novel-view construction — train/inference
            # consistency by construction. Baseline coaches carry
            # mirror_anchor=False, so they get the exact old inline behavior.
            nv = coach._build_novel_view(x, depth, c, c_n, yh, dn,
                                         x_mirror=x_mirror, c_mirror=c_m, depth_mirror=d_m)
            mask, cond, anchor = nv['mask'], nv['cond'], nv['anchor']
            # v9: reference order MUST follow the checkpoint's own config
            # (real_primary=mirror swaps: [x_mirror, y_hat_novel]); using the
            # coach's own method keeps eval consistent with train/validate.
            refs, tags = coach._ref_inputs_real(yh, x_mirror if coach.use_mirror else None)
            gen = coach._sample_novel(cond, mask, codes, refs, tags, seed=42,
                                      mask_cond=nv.get('mask_cond'))
            out[(ident, v)] = dict(x=x, yh=yh, cond=cond, mask=mask, anchor=anchor, gen=gen)
    return out

def metrics(res):
    gen, mask, anchor = res['gen'], res['mask'], res['anchor']
    l1 = (gen - anchor).abs().mean(dim=1, keepdim=True)
    return dict(hole_frac=float(mask.mean()), full=float(l1.mean()),
                hole=float((l1 * mask).sum() / (mask.sum() + 1e-6)),
                visible=float((l1 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-6)),
                hole_lap=lap_energy(gen, mask))


ids = ['000004', '000009', '000014']
views = [1, 2, 3]
root = './data/celeba-hq_1000_static_rebalanced'

coach = build_coach(BASE_CKPT)
print(f'[AB] baseline loaded (mirror={coach.use_mirror}, step={coach.global_step})')
base_res = sample_all(coach, ids, views, root)
del coach
gc.collect()
torch.cuda.empty_cache()

coach = build_coach(MIR_CKPT)
print(f'[AB] mirror loaded (mirror={coach.use_mirror}, step={coach.global_step}, ckpt={MIR_CKPT})')
mir_res = sample_all(coach, ids, views, root)
del coach
gc.collect()
torch.cuda.empty_cache()

lines = ['sample       | BASE full/hole/vis/lap          | MIRROR full/hole/vis/lap        | d_hole']
rows = []
for ident in ids:
    gens = []
    for v in views:
        b, m = base_res[(ident, v)], mir_res[(ident, v)]
        mb, mm = metrics(b), metrics(m)
        lines.append(f'{ident}_v{v} ({mb["hole_frac"]:.2f}) | '
                     f'{mb["full"]:.4f}/{mb["hole"]:.4f}/{mb["visible"]:.4f}/{mb["hole_lap"]:.4f} | '
                     f'{mm["full"]:.4f}/{mm["hole"]:.4f}/{mm["visible"]:.4f}/{mm["hole_lap"]:.4f} | '
                     f'{mm["hole"] - mb["hole"]:+.4f}')
        panel = torch.cat([b['x'][0].cpu(), b['yh'][0].cpu(), b['cond'][0].cpu(),
                           b['mask'][0].expand(3, -1, -1).cpu(), b['anchor'][0].cpu(),
                           b['gen'][0].cpu(), m['gen'][0].cpu()], dim=2)
        vutils.save_image(panel, os.path.join(OUT, f'ab_{ident}_v{v}.png'))
        gens.append(panel)
    rows.append(torch.cat(gens, dim=1))
grid = torch.cat([r.unsqueeze(0) for r in rows], dim=0)
vutils.save_image(grid, os.path.join(OUT, 'ab_overview.png'))

with open(os.path.join(OUT, 'ab_metrics.txt'), 'w') as f:
    f.write('panels: x | y_hat_novel | cond | mask(white=hole) | anchor | BASELINE_gen | MIRROR_gen\n\n')
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'\n[AB] saved -> {OUT}/ab_overview.png (+ per-sample panels, ab_metrics.txt)')

