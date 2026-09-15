# Geometric coverage measurement (eval only, no training change).
#
# First-principles supervision question: the novel-view HOLE currently has only
# an EG3D anchor. A real-photo source exists — inv_warp(x_mirror -> c_novel),
# the same clean grid-sample transfer already used for the visible region.
# Measure: for each test identity x view, what fraction of the hole gets valid
# coverage from the mirror transfer? (validity = in-bounds & depth-consistent)

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append('.')
sys.path.append('..')

from datasets.dataset_inpainting_static import ImageFolderDataset
from utils.warp.Splatting import Warper
from utils.warp.splatting_ext import WarperExt

ROOT = '/data/xzy/warpgan20260803/20260803/warpgan_orig/WarpGAN-main'
ds = ImageFolderDataset(path=os.path.join(ROOT, 'data/celeba-hq_1000_static_rebalanced'),
                        resolution=None, use_labels=True, load_conf_map=False,
                        datast_json='dataset.json')
warper = Warper()
warper_ext = WarperExt()

ids = [ds._image_dirs[i] for i in range(3)]
print('identity   view  hole_frac  mirror_cov_in_hole  inv_cov_in_visible')
for ident in ids:
    # load raw items for all 3 views deterministically
    for view in (1, 2, 3):
        x, c, depth = None, None, None
        item_idx = ds._raw_idx[ids.index(ident)]
        # reload the raw files directly to force this view (dataset randomizes)
        base = os.path.join(ds._path, ident)
        a = np.asarray(Image.open(os.path.join(base, 'x.png'))).astype(np.float32) / 255.
        x = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).cuda()
        c = torch.load(os.path.join(base, 'c.pt'), map_location='cpu').float().reshape(1, -1).cuda()
        depth = torch.load(os.path.join(base, 'depth.pt'), map_location='cpu').float()
        if depth.dim() == 2:
            depth = depth[None, None]
        elif depth.dim() == 3:
            depth = depth[:, None]
        depth = depth.cuda()
        c_m = torch.load(os.path.join(base, 'c_mirror.pt'), map_location='cpu').float().reshape(1, -1).cuda()
        depth_m = torch.load(os.path.join(base, 'depth_mirror.pt'), map_location='cpu').float()
        if depth_m.dim() == 2:
            depth_m = depth_m[None, None]
        elif depth_m.dim() == 3:
            depth_m = depth_m[:, None]
        depth_m = depth_m.cuda()
        c_n = torch.load(os.path.join(base, f'c_novel_{view}.pt'), map_location='cpu').float().reshape(1, -1).cuda()
        depth_n = torch.load(os.path.join(base, f'depth_novel_{view}.pt'), map_location='cpu').float()
        if depth_n.dim() == 2:
            depth_n = depth_n[None, None]
        elif depth_n.dim() == 3:
            depth_n = depth_n[:, None]
        depth_n = depth_n.cuda()

        _, vis, _ = warper.forward_warp(img1=x, depth1=depth, c1=c, c2=c_n)
        hole = 1.0 - vis  # 1 = missing in novel view
        x_mirror = torch.flip(x, dims=[3])
        # real photo transferred into the novel view from the mirror camera
        _, cov_m = warper_ext.inverse_warp(img2=x_mirror, depth1=depth_n, depth2=depth_m,
                                           c1=c_n, c2=c_m)
        # existing visible anchor coverage (reference)
        _, cov_v = warper_ext.inverse_warp(img2=x, depth1=depth_n, depth2=depth,
                                           c1=c_n, c2=c)
        hole_frac = float(hole.mean())
        mirror_cov = float((cov_m * hole).sum() / (hole.sum() + 1e-6))
        inv_cov_vis = float((cov_v * (1 - hole)).sum() / ((1 - hole).sum() + 1e-6))
        blind = float(((1 - cov_m) * (1 - cov_v) * hole).sum() / (hole.sum() + 1e-6))
        print(f'{ident}   v{view}    {hole_frac:.3f}      {mirror_cov:.3f}            '
              f'{inv_cov_vis:.3f}   blind_after_both={blind:.3f}')
