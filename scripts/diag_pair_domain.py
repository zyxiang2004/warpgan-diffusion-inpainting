"""Diagnostic: what does the D's 'real' reference look like at each t?

The x0 distribution supervision compares (fake_t, real_t) where
real_t = add_noise(enc(photo), t). If decoding real_t no longer yields a
photographic image, the pair carries no appearance contrast — the loss
compares mush to mush. This panel makes the VALIDITY DOMAIN of the pair
directly visible: photo | dec(x_t) at t = 0,100,200,300,500,700,900.
Run: CUDA_VISIBLE_DEVICES=1 python scripts/diag_pair_domain.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(ROOT)
OUT = 'train_logs/diag_pair_domain.png'
SD = './pretrained_models/AI-ModelScope/stable-diffusion-v1-5'
CASE = './data/celeba-hq_1000_static_rebalanced/000004/x.png'
DEV = 'cuda:0'

vae = AutoencoderKL.from_pretrained(SD, subfolder='vae').to(DEV).eval()
sched = DDPMScheduler.from_pretrained(SD, subfolder='scheduler')

img = np.asarray(Image.open(CASE)).astype(np.float32) / 255.
x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(DEV)
with torch.no_grad():
    lat = vae.encode(x * 2 - 1).latent_dist.mode() * 0.18215

rows, logs = [], []
torch.manual_seed(0)
noise = torch.randn_like(lat)
row = [x[0].cpu()]
for t in (0, 100, 200, 300, 500, 700, 900):
    with torch.no_grad():
        if t == 0:
            xt = lat
        else:
            tv = torch.tensor([t], device=DEV)
            xt = sched.add_noise(lat, noise, tv)
        rec = (vae.decode(xt / 0.18215).sample / 2 + 0.5).clamp(0, 1)
    ab = float(sched.alphas_cumprod[t])
    l1 = float(F.l1_loss(rec, x))
    row.append(rec[0].cpu())
    logs.append(f't={t:4d}  alpha_bar={ab:.4f}  L1(dec(x_t), photo)={l1:.4f}')

grid = torch.cat([torch.stack(row, 0)], 0)
from torchvision.utils import save_image
save_image(grid, OUT, nrow=len(row), padding=2)
print('\n'.join(logs))
print('panel columns: photo | dec(x_t) t=0,100,200,300,500,700,900')
print(f'saved -> {OUT}')
