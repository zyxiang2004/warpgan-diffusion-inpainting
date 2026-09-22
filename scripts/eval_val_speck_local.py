#!/usr/bin/env python
"""Local re-implementation of the PROJECT_STATUS.md §7.4/§7.6 speck/blotch audit.

Metric definitions (verbatim from §7.6):
  - val panel = 6 tiles concatenated horizontally: [x | render | cond | mask | anchor | gen]
    (source: training/coach_inpainting_diffusion.py L2757-2760, torch.cat dim=2, no padding)
  - mask tile: white (>128) = hole
  - speck3  = count(|img - MedianFilter(3)| > 20) per 1000 px, inside hole
  - blotch7 = count(|img - MedianFilter(7)| > 20) per 1000 px, inside hole
  - vis speck = same speck3 outside hole (visible region)
  - anchor blotch7 = blotch7 of the anchor tile inside hole (anchor cleanliness)
  - x vis speck = speck3 of the source-photo tile outside hole (photo reference)

Sample grouping: the 3 rotating val samples are clustered by an 8x8 average hash
of the `x` tile (identical samples -> identical hash).

Output: one row per panel: step, sample_id, val_hole_l1 (from log if given),
hole_speck3, hole_blotch7, anchor_blotch7, vis_speck3, x_vis_speck3.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
from PIL import Image, ImageFilter

PANEL_ORDER = ['x', 'render', 'cond', 'mask', 'anchor', 'gen']
TH = 20  # intensity threshold, 0-255 grayscale
# Hole convention (CORRECTED 2026-09-19 to match server calib_user_calibration.py
# L29 and PROJECT_STATUS §7.6 "暗像素计数"): mask tile BLACK (<128) = hole.
# Verified: per-sample hole fractions 78.2/63.5/42.8% == server's
# 112170/166416/204907 px @512² (78.1/63.5/42.8%).


def gray(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def med_diff_masked(g, k, m):
    """count of |g - MedianFilter(g,k)| > TH inside mask m, per 1000 px of m"""
    im = Image.fromarray(g.astype(np.uint8), mode='L')
    filt = np.asarray(im.filter(ImageFilter.MedianFilter(size=k)), dtype=np.float32)
    hit = (np.abs(g - filt) > TH) & m
    return float(hit.sum()) / (m.sum() / 1000.0)


def ahash(g):
    im = Image.fromarray(g.astype(np.uint8), mode='L').resize((8, 8), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    return tuple((a > a.mean()).flatten())


def analyse(path):
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
    h, w, _ = a.shape
    tw = w // 6
    tiles = {}
    for i, name in enumerate(PANEL_ORDER):
        tiles[name] = a[:, i * tw:(i + 1) * tw]
    g = {k: gray(v) for k, v in tiles.items()}
    mask = g['mask'] < 128.0  # BLACK = hole (server calib convention)
    if mask.sum() < 50 or (~mask).sum() < 50:
        return None
    gen_hole_speck3 = med_diff_masked(g['gen'], 3, mask)
    gen_hole_blotch7 = med_diff_masked(g['gen'], 7, mask)
    anchor_blotch7 = med_diff_masked(g['anchor'], 7, mask)
    gen_vis_speck3 = med_diff_masked(g['gen'], 3, ~mask)
    x_vis_speck3 = med_diff_masked(g['x'], 3, ~mask)
    # tex: server calib_user_calibration.py convention — mean |dx| of gen inside hole
    dx = np.abs(np.diff(g['gen'], axis=1))
    gm = mask[:, :-1] & mask[:, 1:]
    hole_tex = float(dx[gm].mean())
    hole_frac = float(mask.mean())
    step = int(re.search(r'val_step(\d+)', os.path.basename(path)).group(1))
    return dict(step=step, ahash=ahash(g['x']), hole_frac=hole_frac,
                hole_speck3=gen_hole_speck3, hole_blotch7=gen_hole_blotch7,
                anchor_blotch7=anchor_blotch7, vis_speck3=gen_vis_speck3,
                x_vis_speck3=x_vis_speck3, hole_tex=hole_tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('val_dir')
    ap.add_argument('--log', default=None, help='train log with [val step N] lines')
    args = ap.parse_args()

    l1 = {}
    if args.log and os.path.exists(args.log):
        with open(args.log) as f:
            for line in f:
                m = re.search(r'\[val step (\d+)\].*val_novel_hole=([\d.]+)', line)
                if m:
                    l1[int(m.group(1))] = float(m.group(2))

    rows = []
    pattern = os.path.join(glob.escape(args.val_dir), 'val_step*.png')
    for p in sorted(glob.glob(pattern)):
        r = analyse(p)
        if r:
            rows.append(r)

    # group samples by discrete hole fraction (3 rotating samples have fixed
    # per-sample hole fractions; ahash was not discriminative for face images)
    hf_vals = sorted(set(round(r['hole_frac'], 3) for r in rows))
    rank = {v: i for i, v in enumerate(hf_vals)}
    for r in rows:
        r['sample'] = rank[round(r['hole_frac'], 3)]
    order = list(range(len(hf_vals)))

    print(f"{'step':>7} {'smp':>3} {'hole%':>6} {'L1':>6} {'tex':>5} | {'hole_spk3':>9} {'hole_blt7':>9} "
          f"{'anch_blt7':>9} {'vis_spk3':>8} {'xvis_spk3':>9}")
    for r in sorted(rows, key=lambda r: r['step']):
        print(f"{r['step']:>7} {r['sample']:>3} {r['hole_frac']*100:>5.1f}% "
              f"{l1.get(r['step'], float('nan')):>6.4f} {r['hole_tex']:>5.1f} | {r['hole_speck3']:>9.1f} {r['hole_blotch7']:>9.1f} "
              f"{r['anchor_blotch7']:>9.1f} {r['vis_speck3']:>8.1f} {r['x_vis_speck3']:>9.1f}")

    print("\nper-sample summary (mean over all steps / mean over last 6 vals):")
    for s in order:
        rs = sorted([r for r in rows if r['sample'] == s], key=lambda r: r['step'])
        last = rs[-6:]
        def m(k, rr): return np.mean([x[k] for x in rr])
        print(f"  sample {s} (hole {m('hole_frac', rs)*100:.0f}%): "
              f"tex {m('hole_tex', rs):.1f}/{m('hole_tex', last):.1f}  "
              f"hole_spk3 {m('hole_speck3', rs):.1f}/{m('hole_speck3', last):.1f}  "
              f"hole_blt7 {m('hole_blotch7', rs):.1f}/{m('hole_blotch7', last):.1f}  "
              f"anch_blt7 {m('anchor_blotch7', rs):.1f}/{m('anchor_blotch7', last):.1f}  "
              f"vis_spk3 {m('vis_speck3', rs):.1f}/{m('vis_speck3', last):.1f}  "
              f"xvis_spk3 {m('x_vis_speck3', rs):.1f}/{m('x_vis_speck3', last):.1f}  "
              f"(all/last6, n={len(rs)})")


if __name__ == '__main__':
    sys.exit(main())
