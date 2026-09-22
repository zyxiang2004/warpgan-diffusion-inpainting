#!/usr/bin/env python
"""Shared validator feature pipeline (37 features, §7.30). Pure numpy/PIL."""
import numpy as np
from PIL import Image, ImageFilter

PANELS = ['x', 'yhat', 'cond', 'mask', 'anchor', 'gen']

def panels(p):
    im = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.
    w = im.shape[1] // 6
    return {k: im[:, i * w:(i + 1) * w] for i, k in enumerate(PANELS)}

def gray(a):
    return a[..., 0] * .299 + a[..., 1] * .587 + a[..., 2] * .114

def erode(m, it=1):
    m = m > 0.5
    for _ in range(it):
        e = m.copy()
        e[1:, :] &= m[:-1, :]; e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
        m = e
    return m

def dilate(m, it=1):
    m = m > 0.5
    for _ in range(it):
        e = m.copy()
        e[1:, :] |= m[:-1, :]; e[:-1, :] |= m[1:, :]
        e[:, 1:] |= m[:, :-1]; e[:, :-1] |= m[:, 1:]
        m = e
    return m

def med(a, k):
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
                      .filter(ImageFilter.MedianFilter(k)), dtype=np.float32) / 255.

def radial(g, m):
    mm = (m > 0.5).astype(np.float32)
    F = np.log1p(np.fft.fftshift(np.abs(np.fft.fft2((g - g.mean()) * mm)) ** 2))
    yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    r = np.hypot(yy - g.shape[0] // 2, xx - g.shape[1] // 2)
    edges = np.arange(0, 129, 8)
    prof = np.array([F[(r >= edges[i]) & (r < edges[i + 1])].mean()
                     for i in range(len(edges) - 1)])
    return prof, (edges[:-1] + edges[1:]) / 2

def coherency(g, m):
    gx = np.gradient(g, axis=1); gy = np.gradient(g, axis=0)
    kb = np.ones((5, 5)) / 25.
    def box(a):
        ap = np.pad(a, 2, mode='edge')
        return sum(kb[i, j] * ap[i:i + a.shape[0], j:j + a.shape[1]]
                   for i in range(5) for j in range(5))
    Sxx, Syy, Sxy = box(gx * gx), box(gy * gy), box(gx * gy)
    tr = Sxx + Syy + 1e-8
    coh = np.sqrt((Sxx - Syy) ** 2 + 4 * Sxy ** 2) / tr
    mm = erode(m) > 0.5
    return float(coh[mm].mean()), float(np.sqrt(gx * gx + gy * gy)[mm].mean())

def lskew_lstd(g, m):
    from scipy import stats as st
    k = 8
    gp = np.pad(g, 4, mode='edge')
    loc = np.zeros((g.shape[0] // k, g.shape[1] // k))
    for i in range(loc.shape[0]):
        for j in range(loc.shape[1]):
            loc[i, j] = gp[i * k:i * k + k, j * k:j * k + k].std()
    loc = loc.flatten()
    return float(st.skew(loc)), float(loc.std())

def feats_of(P):
    m = gray(P['mask']); g = gray(P['gen']); rgb = P['gen']
    hole = erode(m) > 0.5
    prof, rr = radial(g, m)
    sel = (rr >= 32) & (rr <= 120)
    slope = float(np.polyfit(rr[sel], prof[sel], 1)[0])
    gp = np.pad(g, 1, mode='edge')
    lap = np.abs(4 * g - gp[:-2, 1:-1] - gp[2:, 1:-1] - gp[1:-1, :-2] - gp[1:-1, 2:])
    tex = float(lap[hole].mean())
    md3 = med(g, 3)
    ph = float(((np.abs(g - md3) > 0.055) & hole).sum()) / hole.sum() * 1000
    ring = (m > 0.5) & dilate(m <= 0.5, 4)
    inter = erode(m, 4)
    ph_r = float(((np.abs(g - md3) > 0.055) & ring).sum()) / max(ring.sum(), 1) * 1000
    ph_i = float(((np.abs(g - md3) > 0.055) & inter).sum()) / max(inter.sum(), 1) * 1000
    vis = erode(1.0 - m) > 0.5
    ph_v = float(((np.abs(g - md3) > 0.055) & vis).sum()) / max(vis.sum(), 1) * 1000
    a = rgb[..., 0] - rgb[..., 1]; b = rgb[..., 1] - rgb[..., 2]
    bl = float((((np.abs(a - med(a, 5)) + np.abs(b - med(b, 5))) > 0.045) & hole).sum()) / hole.sum() * 1000
    hflf = float(prof[rr >= 88].mean() - prof[rr <= 48].mean())
    coh, gmag = coherency(g, m)
    lsk, lstd = lskew_lstd(g, m)
    out_r = dilate(m <= 0.5, 4) & (m <= 0.5)
    seam = float(abs(lap[out_r].mean() - lap[ring].mean()))
    sat = (rgb.max(-1) - rgb.min(-1))[m > 0.5]
    d = dict(slope=slope, tex=tex, ph=ph, ph_ring=ph_r, ph_in=ph_i, ph_vis=ph_v,
             bl=bl, hflf=hflf, coh=coh, gmag=gmag, lskew=lsk, lstd=lstd, seam=seam,
             sat_m=float(sat.mean()), sat_s=float(sat.std()))
    for i, v in enumerate(prof):
        d[f'spec{i}'] = float(v)
    for i in range(3):
        d[f'cm{i}'] = float(rgb[..., i][hole].mean())
        d[f'cs{i}'] = float(rgb[..., i][hole].std())
    return d
