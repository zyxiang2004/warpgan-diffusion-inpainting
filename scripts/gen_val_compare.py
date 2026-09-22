#!/usr/bin/env python
"""Side-by-side val comparison: W6-A baseline vs W7, per rotating sample.

For each of the 3 rotating val samples, picks the latest panel of each arm and
renders one row: [anchor reference | W6-A gen | W7 gen], with the HOLE region
outlined in red on the gen tiles (mask tile: BLACK < 128 = hole — server calib
convention). Anchor is shown WITHOUT outline (it is the supervision target).
Output: a single PNG for human visual adjudication.
"""
import argparse
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W6A_VAL = 'experiments/train_inpainting_diffusion/[20260915-111722]_v19_w6a_100k/logs/images/val'
W7_VAL = 'experiments/train_inpainting_diffusion/v19_w7_holed_300k/logs/images/val'
TILE = 512


def load_tiles(path):
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
    assert a.shape[1] == TILE * 6, a.shape
    return [a[:, i * TILE:(i + 1) * TILE] for i in range(6)]


def hole_mask(tiles):
    g = 0.299 * tiles[3][..., 0].astype(np.float32) + 0.587 * tiles[3][..., 1] + 0.114 * tiles[3][..., 2]
    return g < 128.0


def outline(gen, mask, color=(255, 0, 0)):
    m = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    er = np.asarray(m.filter(ImageFilter.MinFilter(5)), dtype=np.float32) > 128
    edge = mask & (~er)
    out = gen.copy()
    out[edge] = color
    return out


def latest_with_holefrac(val_dir, frac):
    best = None
    for f in sorted(os.listdir(val_dir)):
        if not f.startswith('val_step'):
            continue
        p = os.path.join(val_dir, f)
        t = load_tiles(p)
        hf = hole_mask(t).mean()
        if abs(hf * 100 - frac) < 1.0:
            best = (p, t)  # sorted -> last match = latest
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='train_logs/w7_vs_w6a_val_compare.png')
    args = ap.parse_args()
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
        font_s = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    except OSError:
        font = font_s = ImageFont.load_default()

    cases = [('SMALL hole 43% (sample 112170)', 42.8), ('MID hole 63% (sample 166416)', 63.5),
             ('LARGE hole 78% (sample 204907)', 78.2)]
    H, W = 3 * (TILE + 46) + 46, 3 * TILE + 2 * 24 + 340
    canvas = Image.new('RGB', (W, H), (24, 24, 24))
    d = ImageDraw.Draw(canvas)
    d.text((24, 8), 'Hole region = RED outline.  Columns: [ anchor (target) | W6-A baseline gen | W7 gen ].  '
                    'Same sample per row; compare texture inside the red outline.', font=font_s, fill=(230, 230, 230))
    cols = ['anchor (target)', 'W6-A gen', 'W7 gen']
    y0 = 56
    for ci, c in enumerate(cols):
        d.text((340 + ci * (TILE + 24) + TILE // 2 - 90, y0), c, font=font, fill=(255, 210, 80))
    for ri, (label, frac) in enumerate(cases):
        y = y0 + 40 + ri * (TILE + 46)
        d.text((24, y + TILE // 2 - 30), label, font=font, fill=(160, 220, 255))
        w6 = latest_with_holefrac(W6A_VAL, frac)
        w7 = latest_with_holefrac(W7_VAL, frac)
        assert w6 and w7, f'missing panels for frac {frac}'
        for ci, (pair, is_gen) in enumerate([(w6, False), (w6, True), (w7, True)]):
            p, t = pair
            tile = t[4] if ci == 0 else t[5]
            if is_gen:
                tile = outline(t[5], hole_mask(t))
            x = 340 + ci * (TILE + 24)
            canvas.paste(Image.fromarray(tile), (x, y))
            d.text((x + 6, y + TILE - 26), os.path.basename(p).replace('val_step', 'step '),
                   font=font_s, fill=(200, 200, 200))
    canvas.save(args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()
