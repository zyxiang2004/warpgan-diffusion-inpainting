# -*- coding: utf-8 -*-
"""用户校准集生成器 — 把"我的指标标准"选出的好图集中到一个文件夹，
供用户肉眼校准。规则预注册（PROJECT_STATUS §7.9）：
  GOOD  = 洞区纹理能量 tex >= 9.0 （丰富端；用户已知正例 @82K tex=11.6）
  SMEAR = tex <= 5.0 （涂抹端，对照组）
  GOOD 内分 tier：A = speck3 <= 60（丰富且斑少），B = speck3 > 60（可能过噪）
"""
import re, glob, os, shutil
import numpy as np
from PIL import Image, ImageFilter

B = 'experiments/train_inpainting_diffusion/'
ARMS = {
    'w6a': (B + '[20260915-085452]_v19_w6a_300k', 'train_logs/v19_w6a.log'),
    'w6b': (B + '[20260915-085455]_v19_w6b_300k', 'train_logs/v19_w6b.log'),
    'w3p': (B + '[20260911-143749]_v18w3_w3p_30k', 'train_logs/v18w3_w3p.log'),
}
OUT = 'user_calibration_20260918'
D_GOOD = os.path.join(OUT, '01_预测好_丰富端')
D_SMEAR = os.path.join(OUT, '02_对照_预测涂抹端')
os.makedirs(D_GOOD, exist_ok=True)
os.makedirs(D_SMEAR, exist_ok=True)


def metrics(png):
    arr = np.array(Image.open(png).convert('L'), dtype=np.float32)
    x, render, cond, mask, anchor, gen = [arr[:, i * 512:(i + 1) * 512] for i in range(6)]
    hole = mask < 128

    def sp(img, k):
        med = np.array(Image.fromarray(img.astype(np.uint8)).filter(ImageFilter.MedianFilter(k)),
                       dtype=np.float32)
        return float(((np.abs(img - med) > 20) & hole).sum() / hole.sum() * 1000)

    g = np.abs(np.diff(gen, axis=1))
    gm = hole[:, :-1] & hole[:, 1:]
    return int(hole.sum()), sp(gen, 3), sp(gen, 7), float(g[gm].mean())


rows = []
for arm, (run, logf) in ARMS.items():
    l1 = {}
    for line in open(logf, errors='ignore'):
        m = re.match(r'\[val step (\d+)\].*val_novel_hole=([\d.]+)', line)
        if m:
            l1[int(m.group(1))] = float(m.group(2))
    for png in sorted(glob.glob(glob.escape(run) + '/logs/images/val/*.png')):
        step = int(png.split('val_step')[-1].split('.')[0])
        hp, s3, s7, tex = metrics(png)
        rows.append(dict(arm=arm, step=step, sample=hp, s3=s3, s7=s7,
                         tex=tex, l1=l1.get(step), png=png))


def verdict(r):
    if r['tex'] >= 9.0:
        return 'GOOD'
    if r['tex'] <= 5.0:
        return 'SMEAR'
    return 'MID'


n = {'GOOD': 0, 'SMEAR': 0}
for r in rows:
    v = verdict(r)
    r['v'] = v
    if v == 'MID':
        continue
    l1s = '%.3f' % r['l1'] if r['l1'] is not None else 'nan'
    if v == 'GOOD':
        tier = 'A' if r['s3'] <= 60 else 'B'
        dst = os.path.join(D_GOOD, '%s_tier%s_s%d_st%07d_tex%.1f_sp%.0f_bl%.0f_L1%s.png'
                           % (r['arm'], tier, r['sample'], r['step'], r['tex'], r['s3'], r['s7'], l1s))
    else:
        dst = os.path.join(D_SMEAR, '%s_s%d_st%07d_tex%.1f_sp%.0f_bl%.0f_L1%s.png'
                           % (r['arm'], r['sample'], r['step'], r['tex'], r['s3'], r['s7'], l1s))
    shutil.copy2(r['png'], dst)
    n[v] += 1

texs = np.array([r['tex'] for r in rows])
print('面板总数=%d  GOOD(tex>=9)=%d  SMEAR(tex<=5)=%d  MID=%d'
      % (len(rows), n['GOOD'], n['SMEAR'], len(rows) - n['GOOD'] - n['SMEAR']))
print('tex分布: min=%.1f p25=%.1f 中位=%.1f p75=%.1f max=%.1f'
      % (texs.min(), np.percentile(texs, 25), np.median(texs),
         np.percentile(texs, 75), texs.max()))
for arm in ARMS:
    sub = [r for r in rows if r['arm'] == arm]
    g = sum(1 for r in sub if r['v'] == 'GOOD')
    s = sum(1 for r in sub if r['v'] == 'SMEAR')
    print('  %s: GOOD=%d SMEAR=%d / 共%d' % (arm, g, s, len(sub)))

with open(os.path.join(OUT, 'MANIFEST.md'), 'w') as f:
    f.write('# 用户校准集 — 我的指标标准 vs 你的肉眼标准 (2026-09-18)\n\n')
    f.write('**预注册规则（选图前先定死）**：GOOD = 洞区纹理能量 tex>=9（丰富端，'
            '你的已知正例 w6a@81999 tex=11.6）；SMEAR = tex<=5（涂抹端）；中间不复制。\n')
    f.write('GOOD 内 tier：A = speck3<=60（丰富且斑少），B = speck3>60（丰富但可能过噪）。\n')
    f.write('文件名携带全部指标。请你看完两个文件夹后告诉我：\n')
    f.write('① 01 里有几张是你认可的"好"；② 02 里有几张其实不差；③ 你被 @82K 打动的原因如果是别的（比如结构/发际线），请描述。\n\n')
    f.write('| arm | step | 样本 | tex | speck3 | blotch7 | L1hole | 判定 |\n|---|---|---|---|---|---|---|---|\n')
    for r in sorted(rows, key=lambda r: (r['arm'], r['step'])):
        l1s = '%.3f' % r['l1'] if r['l1'] is not None else '-'
        mark = ' **<== 你的已知好例**' if r['arm'] == 'w6a' and r['step'] == 81999 else ''
        f.write('| %s | %d | %d | %.2f | %.1f | %.1f | %s | %s%s |\n'
                % (r['arm'], r['step'], r['sample'], r['tex'], r['s3'], r['s7'], l1s, r['v'], mark))
print('MANIFEST 写入', os.path.join(OUT, 'MANIFEST.md'))
