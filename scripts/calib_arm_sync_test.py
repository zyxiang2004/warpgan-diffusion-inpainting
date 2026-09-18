# -*- coding: utf-8 -*-
"""两臂翻转同步性检验：w6a vs w6b 在相同样本、相同步数上的洞区纹理能量序列
是否相关。两臂同种子同时启动 → 若数据顺序相同且翻转同步 → 驱动是课程数据；
若不同步 → 驱动是各臂自身的权重噪声。同时验证 166416 全序列。"""
import re, glob
import numpy as np
from PIL import Image

B = 'experiments/train_inpainting_diffusion/'
RUNS = {'w6a': B + '[20260915-085452]_v19_w6a_300k',
        'w6b': B + '[20260915-085455]_v19_w6b_300k'}


def tex_series(run):
    out = {}
    for png in sorted(glob.glob(glob.escape(run) + '/logs/images/val/*.png')):
        step = int(png.split('val_step')[-1].split('.')[0])
        arr = np.array(Image.open(png).convert('L'), dtype=np.float32)
        mask, gen = arr[:, 3 * 512:4 * 512], arr[:, 5 * 512:6 * 512]
        hole = mask < 128
        g = np.abs(np.diff(gen, axis=1))
        gm = hole[:, :-1] & hole[:, 1:]
        out[step] = (int(hole.sum()), float(g[gm].mean()))
    return out


A, Bb = tex_series(RUNS['w6a']), tex_series(RUNS['w6b'])
print('== 两臂同步性（相同样本、相同步数配对）==')
for sample in sorted({v[0] for v in A.values()}):
    steps = [s for s in sorted(set(A) & set(Bb)) if A[s][0] == sample]
    ta = np.array([A[s][1] for s in steps])
    tb = np.array([Bb[s][1] for s in steps])
    if len(steps) < 6:
        continue
    pear = np.corrcoef(ta, tb)[0, 1]
    agree = np.mean([(x >= 9) == (y >= 9) for x, y in zip(ta, tb)])
    print('样本 %d: n=%d  Pearson r=%.2f  GOOD判定一致率=%.0f%%' % (sample, len(steps), pear, agree * 100))
    print('  w6a:', ' '.join('%.1f' % x for x in ta))
    print('  w6b:', ' '.join('%.1f' % x for x in tb))
    print('  步:', ' '.join('%dK' % (s // 1000) for s in steps))
