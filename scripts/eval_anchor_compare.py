# -*- coding: utf-8 -*-
"""ANCHOR COMPARISON: orig FFC SVINet vs diffusion v16 vs diffusion v12.

Teacher request (2026-09-08): compare against the ORIGINAL WarpGAN result,
many cases, in detail.

Orig FFC inference = faithful replay of coach_inpainting_static.py:
  warp L223/L226; process_mask L186-198 (erode3+blur21, sigma fixed 1.05);
  hybrid L177-178; 14ch input L152-156 (both branches cat_inv=True);
  pred = FFCStyleResNetGenerator(inp, codes), sigmoid full frame
  (style = ws[:, -5:] slices, ffc_style.py L159-203).
diffusion v16/v12 = Coach._build_novel_view + _sample_novel (50-step DDIM,
seed=42), same as scripts/eval_step2_ab.py. SAME ids/views/seed/anchor x3.

Outputs (experiments/anchor_compare/): anchor_metrics.txt,
cmp_<id>_v<v>.png (x|render|mask|anchor|FFC|v16|v12),
hole_<id>_v<v>.png (2x hole-centroid crops FFC|v16|v12|anchor),
overview_<id>.png, overview_holes_<id>.png
"""
import gc
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = '/data/xzy/warpgan20260803/20260803/warpgan_orig/WarpGAN-main'
os.chdir(ROOT)
sys.path.append('.')
sys.path.append('..')

from omegaconf import OmegaConf
import torchvision.utils as vutils
import kornia.morphology as km
from torchvision import transforms as tvt

from utils.warp.Splatting import Warper

DEV = 'cuda:0'
DATA_ROOT = './data/celeba-hq_1000_static_rebalanced'
FFC_CKPT = './pretrained_models/inpaintor/inpaintor.pt'
OUT = './experiments/anchor_compare'
os.makedirs(OUT, exist_ok=True)
V16_DIR = './experiments/train_inpainting_diffusion/[20260903-032126]_step3_v16_photoref_50k/checkpoints'
V12_DIR = './experiments/train_inpainting_diffusion/[20260828-205831]_step3_v12_dualband_50k/checkpoints'
LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def log(msg):
    print(msg, flush=True)


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


def load_case(ident, v):
    base = os.path.join(DATA_ROOT, ident)

    def img(name):
        a = np.asarray(Image.open(os.path.join(base, name))).astype(np.float32) / 255.
        return torch.from_numpy(a.transpose(2, 0, 1))

    x = img('x.png').unsqueeze(0).to(DEV)
    c = torch.load(os.path.join(base, 'c.pt'), map_location='cpu').float().reshape(1, -1).to(DEV)
    codes = torch.load(os.path.join(base, 'codes.pt'), map_location='cpu').float()
    codes = codes.reshape(1, *codes.shape[-2:]).to(DEV)
    depth = as_depth(torch.load(os.path.join(base, 'depth.pt'), map_location='cpu')).to(DEV)
    c_n = torch.load(os.path.join(base, f'c_novel_{v}.pt'), map_location='cpu').float().reshape(1, -1).to(DEV)
    yh = img(f'y_hat_novel_{v}.png').unsqueeze(0).to(DEV)
    dn = as_depth(torch.load(os.path.join(base, f'depth_novel_{v}.pt'), map_location='cpu')).to(DEV)
    x_mirror = torch.flip(x, dims=[3])
    c_m = torch.load(os.path.join(base, 'c_mirror.pt'), map_location='cpu').float().reshape(1, -1).to(DEV)
    d_m = as_depth(torch.load(os.path.join(base, 'depth_mirror.pt'), map_location='cpu')).to(DEV)
    return dict(x=x, c=c, codes=codes, depth=depth, c_n=c_n, yh=yh, dn=dn,
                x_mirror=x_mirror, c_m=c_m, d_m=d_m)




def soften_vis(vis):
    """orig coach process_mask L186-198: erode visible 3px + GaussianBlur(21)."""
    vis = km.erosion(vis, torch.ones(3, 3, device=vis.device, dtype=vis.dtype))
    vis = tvt.GaussianBlur(21, sigma=1.05)(vis)
    return vis


class FFCRunner:
    def __init__(self):
        cfg = OmegaConf.load('./configs/train_inpainting.yaml')
        gen = OmegaConf.to_container(cfg.generator, resolve=True)
        kind = gen.pop('kind')
        from models.saicinpainting.training.modules import make_generator
        self.net = make_generator(None, kind, **gen).to(DEV).eval()
        sd = torch.load(FFC_CKPT, map_location='cpu')['inpaintor_state_dict']
        missing, unexpected = self.net.load_state_dict(sd, strict=False)[:2]
        log(f'[FFC] loaded; missing={len(missing)} unexpected={len(unexpected)}')
        if missing:
            log(f'[FFC][WARN] missing e.g. {missing[:5]}')
        if unexpected:
            log(f'[FFC][WARN] unexpected e.g. {unexpected[:5]}')
        self.warper = Warper()

    @torch.no_grad()
    def __call__(self, case):
        x, depth, c, c_n, yh = case['x'], case['depth'], case['c'], case['c_n'], case['yh']
        warp_img, vis, _ = self.warper.forward_warp(img1=x, depth1=depth, c1=c, c2=c_n)
        warp_m, vis_m, _ = self.warper.forward_warp(
            img1=case['x_mirror'], depth1=case['d_m'], c1=case['c_m'], c2=c_n)
        mask, mask_m = 1.0 - vis, 1.0 - vis_m
        mask_s = 1.0 - soften_vis(vis)
        mask_ms = 1.0 - soften_vis(vis_m)
        hybrid = warp_img * (1.0 - mask_s) + yh * mask_s
        hybrid_m = warp_m * (1.0 - mask_ms) + yh * mask_ms
        inp = torch.cat([hybrid, yh, mask_s, hybrid_m, yh, mask_ms], dim=1)
        pred = self.net(inp, case['codes']).clamp(0, 1)
        return dict(pred=pred, mask=mask)


def hole_frac(ident, v):
    case = load_case(ident, v)
    w = Warper()
    _, vis, _ = w.forward_warp(img1=case['x'], depth1=case['depth'],
                               c1=case['c'], c2=case['c_n'])
    del case
    return float((1.0 - vis).mean())


def select_ids():
    fixed = ['000004', '000009', '000014']
    all_ids = sorted(d for d in os.listdir(DATA_ROOT)
                     if d.isdigit() and len(d) == 6 and d not in fixed)
    rng = random.Random(7)
    cands = rng.sample(all_ids, 24)
    scored = []
    for ident in cands:
        fr = np.mean([hole_frac(ident, v) for v in (1, 2, 3)])
        scored.append((fr, ident))
        log(f'[scan] {ident} mean_hole={fr:.3f}')
    scored.sort()
    picks = [scored[int(q)][1] for q in np.linspace(0, len(scored) - 1, 5)]
    return fixed + picks


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


def latest_ckpt(d):
    cands = sorted(f for f in os.listdir(d) if f.endswith('.pt'))
    return os.path.join(d, cands[-1])


@torch.no_grad()
def diffusion_sample(coach, case):
    nv = coach._build_novel_view(
        case['x'], case['depth'], case['c'], case['c_n'], case['yh'], case['dn'],
        x_mirror=case['x_mirror'], c_mirror=case['c_m'], depth_mirror=case['d_m'])
    refs, tags = coach._ref_inputs_real(case['yh'],
                                        case['x_mirror'] if coach.use_mirror else None)
    gen = coach._sample_novel(nv['cond'], nv['mask'], case['codes'], refs, tags,
                              seed=42, mask_cond=nv.get('mask_cond'))
    return gen, nv


def metrics(gen, mask, anchor, yh, x):
    l1 = (gen - anchor).abs().mean(dim=1, keepdim=True)
    l1y = (gen - yh).abs().mean(dim=1, keepdim=True)
    m, im = mask, (1.0 - mask)
    return dict(
        full=float(l1.mean()), hole=float((l1 * m).sum() / (m.sum() + 1e-6)),
        vis=float((l1 * im).sum() / (im.sum() + 1e-6)),
        hole_vs_render=float((l1y * m).sum() / (m.sum() + 1e-6)),
        gen_hole_lap=lap_energy(gen, m), gen_vis_lap=lap_energy(gen, im),
        anchor_hole_lap=lap_energy(anchor, m), render_hole_lap=lap_energy(yh, m),
        photo_vis_lap=lap_energy(x, im))


def main():
    torch.manual_seed(0)
    ids = select_ids()
    views = [1, 2, 3]
    log(f'[cases] ids={ids} views={views}')
    ffc = FFCRunner()

    results = {}
    for tag, ckpt_dir in (('v16', V16_DIR), ('v12', V12_DIR)):
        ckpt = latest_ckpt(ckpt_dir)
        log(f'[{tag}] building coach from {ckpt}')
        coach = build_coach(ckpt)
        log(f'[{tag}] loaded (mirror={coach.use_mirror}, step={coach.global_step})')
        for ident in ids:
            for v in views:
                case = load_case(ident, v)
                gen, nv = diffusion_sample(coach, case)
                results.setdefault((ident, v), {'case': case, 'nv': nv})
                results[(ident, v)][tag] = gen
        del coach
        gc.collect()
        torch.cuda.empty_cache()
        log(f'[{tag}] done, freed')

    for ident in ids:
        for v in views:
            key = (ident, v)
            results[key]['FFC'] = ffc(results[key]['case'])['pred']

    lines = ['panel row: x | y_hat_novel | mask(white=hole) | anchor | FFC | v16 | v12']
    lines.append('aF/aH/aV = L1 vs anchor full/hole/visible; rH = L1 vs render (hole);')
    lines.append('lapH/lapV = laplacian energy in hole/visible (texture proxy)\n')
    agg = {'FFC': [], 'v16': [], 'v12': []}
    for ident in ids:
        for v in views:
            r = results[(ident, v)]
            case, nv = r['case'], r['nv']
            mask, anchor, yh, x = nv['mask'], nv['anchor'], case['yh'], case['x']
            row = [f'{ident}_v{v} h={float(mask.mean()):.2f}']
            for tag in ('FFC', 'v16', 'v12'):
                mm = metrics(r[tag], mask, anchor, yh, x)
                agg[tag].append(mm)
                row.append(f"{tag} aF/aH/aV={mm['full']:.3f}/{mm['hole']:.3f}/{mm['vis']:.3f} "
                           f"rH={mm['hole_vs_render']:.3f} lapH={mm['gen_hole_lap']:.3f} "
                           f"lapV={mm['gen_vis_lap']:.3f}")
            row.append(f"REFlap aH/rH/pV={mm['anchor_hole_lap']:.3f}/"
                       f"{mm['render_hole_lap']:.3f}/{mm['photo_vis_lap']:.3f}")
            lines.append(' | '.join(row))
    with open(os.path.join(OUT, 'anchor_metrics.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log('[metrics] table written')

    for ident in ids:
        rows_main, rows_hole = [], []
        for v in views:
            r = results[(ident, v)]
            case, nv = r['case'], r['nv']
            mask, anchor = nv['mask'], nv['anchor']
            panel = torch.cat([case['x'][0].cpu(), case['yh'][0].cpu(),
                               mask[0].expand(3, -1, -1).cpu(), anchor[0].cpu(),
                               r['FFC'][0].cpu(), r['v16'][0].cpu(),
                               r['v12'][0].cpu()], dim=2)
            vutils.save_image(panel, os.path.join(OUT, f'cmp_{ident}_v{v}.png'))
            rows_main.append(panel)
            m = mask[0, 0]
            ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
            cy, cx = (int(ys.float().mean()), int(xs.float().mean())) if len(ys) > 0 else (256, 256)
            r1, c1 = min(512, cy + 80), min(512, cx + 80)
            r0, c0 = r1 - 160, c1 - 160
            crops = []
            for t in (r['FFC'], r['v16'], r['v12'], anchor):
                c = t[0, :, r0:r1, c0:c1].cpu()
                crops.append(F.interpolate(c.unsqueeze(0), scale_factor=2,
                                           mode='bilinear', align_corners=False)[0])
            hpanel = torch.cat(crops, dim=2)
            vutils.save_image(hpanel, os.path.join(OUT, f'hole_{ident}_v{v}.png'))
            rows_hole.append(hpanel)
        grid = torch.cat([r.unsqueeze(0) for r in rows_main], dim=0)
        vutils.save_image(grid, os.path.join(OUT, f'overview_{ident}.png'))
        gh = torch.cat([r.unsqueeze(0) for r in rows_hole], dim=0)
        vutils.save_image(gh, os.path.join(OUT, f'overview_holes_{ident}.png'))

    summary = ['\nmean over all cases:']
    for tag in ('FFC', 'v16', 'v12'):
        ms = agg[tag]
        summary.append(
            f"  {tag}: anchorL1 full/hole/vis="
            f"{np.mean([m['full'] for m in ms]):.4f}/"
            f"{np.mean([m['hole'] for m in ms]):.4f}/"
            f"{np.mean([m['vis'] for m in ms]):.4f} | "
            f"renderL1 hole={np.mean([m['hole_vs_render'] for m in ms]):.4f} | "
            f"lap hole={np.mean([m['gen_hole_lap'] for m in ms]):.4f} "
            f"vis={np.mean([m['gen_vis_lap'] for m in ms]):.4f}")
    ms = agg['FFC']
    summary.append(
        f"  REF: anchor_hole_lap={np.mean([m['anchor_hole_lap'] for m in ms]):.4f} "
        f"render_hole_lap={np.mean([m['render_hole_lap'] for m in ms]):.4f} "
        f"photo_vis_lap={np.mean([m['photo_vis_lap'] for m in ms]):.4f}")
    tail = '\n'.join(summary)
    log(tail)
    with open(os.path.join(OUT, 'anchor_metrics.txt'), 'a') as f:
        f.write(tail + '\n')
    log(f'[done] all outputs -> {OUT}')


if __name__ == '__main__':
    main()


