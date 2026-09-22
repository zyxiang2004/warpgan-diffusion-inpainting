#!/usr/bin/env python
"""W10 val-frame scorer: learned validator (§7.30, user labels 2026-09-21).
Scores every val_step*.png in an experiment dir with P(GOOD), writes CSV +
timeline PNG. Zero model loading. Usage:
  python scripts/score_val_frames.py <exp_dir>
"""
import os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validator_feats import panels, feats_of

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
VAL_ASSET = os.path.join(ROOT, 'experiments/analysis_osc_20260921/validator.npz')


def main():
    exp = sys.argv[1].rstrip('/')
    valdir = os.path.join(exp, 'logs', 'images', 'val')
    if not os.path.isdir(valdir):
        print(f'no val dir: {valdir}'); return
    d = np.load(VAL_ASSET, allow_pickle=True)
    keys, mu, sd, coef = list(d['keys']), d['mu'], d['sd'], d['coef']
    rows = []
    for f in sorted(os.listdir(valdir)):
        if not (f.startswith('val_step') and f.endswith('.png')):
            continue
        step = int(re.search(r'val_step(\d+)', f).group(1))
        rows.append((step, feats_of(panels(os.path.join(valdir, f)))))
    if not rows:
        print('no val frames'); return
    X = np.array([[r[1].get(k, 0.0) for k in keys] for r in rows])
    Z = (X - mu) / (sd + 1e-9)
    p = 1 / (1 + np.exp(-(Z @ coef)))
    with open(os.path.join(exp, 'logs', 'validator_scores.csv'), 'w') as fh:
        fh.write('step,p_good\n')
        for (s, _), pv in zip(rows, p):
            fh.write(f'{s},{pv:.4f}\n')
    good = float(np.mean(p >= 0.5))
    last = p[-10:] if len(p) >= 10 else p
    print(f'{os.path.basename(exp)}: frames={len(p)} good_rate={good:.2f} '
          f'last10 P(G) mean={np.mean(last):.3f} std={np.std(last):.3f}')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        st = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(st, p, '.-', lw=.8)
        ax.axhline(0.5, color='r', ls='--', lw=.8)
        ax.set_xlabel('step'); ax.set_ylabel('P(GOOD)')
        ax.set_title(os.path.basename(exp))
        fig.tight_layout()
        fig.savefig(os.path.join(exp, 'logs', 'validator_timeline.png'), dpi=100)
        plt.close(fig)
    except Exception as e:
        print('plot skipped:', e)


if __name__ == '__main__':
    main()
