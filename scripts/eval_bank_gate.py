# Direction-B gate check (FROZEN weights, inference only):
#   variant R  : mirror reference features from the standalone RefNet (current)
#   variant B  : mirror reference features from the MAIN SD UNet's own attn1
#                activations (AnimateDiff "bank" — same weights, self-extract)
# Same consumption path, so this isolates ONE question: is the frozen RefNet's
# extraction meaningfully better than the main UNet reading itself? If B ~= R
# (or better), deleting the 860M RefNet is safe (direction B).

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append('.')
sys.path.append('..')

from omegaconf import OmegaConf
import torchvision.utils as vutils
from diffusers import DDIMScheduler

from training.coach_inpainting_diffusion import Coach

BASE = '/data/xzy/warpgan20260803/20260803/warpgan_orig/WarpGAN-main/experiments/train_inpainting_diffusion'
CKPT_DIR = os.path.join(BASE, 'step2v7a_mirror_kv_20k/checkpoints')
cands = sorted(f for f in os.listdir(CKPT_DIR) if f.endswith('.pt'))
CKPT = os.path.join(CKPT_DIR, cands[-1])
print(f'[gate] ckpt: {CKPT}')

exp = os.path.dirname(os.path.dirname(os.path.abspath(CKPT)))
cfg = os.path.join(exp, 'config.yaml')
if not os.path.exists(cfg):
    cfg = os.path.join(exp, 'config_resume.yaml')
opts = OmegaConf.load(cfg)
opts.checkpoint_path = os.path.abspath(CKPT)
opts.exp_dir = exp
opts.smoke.check = False
coach = Coach(opts)
coach.denoising_unet.eval()
dev = coach.device

ROOT = '/data/xzy/warpgan20260803/20260803/warpgan_orig/WarpGAN-main'
TEST = os.path.join(ROOT, 'data/celeba-hq_1000_static_rebalanced')
SAMPLES = [('000014', 1), ('000004', 2), ('000009', 3)]
OUT = os.path.join(exp, 'bank_gate_check')
os.makedirs(OUT, exist_ok=True)


def as_depth(t):
    t = t.float()
    if t.dim() == 2:
        t = t[None, None]
    elif t.dim() == 3:
        t = t[:, None]
    return t


def load_sample(ident, view):
    base = os.path.join(TEST, ident)

    def img(name):
        a = np.asarray(Image.open(os.path.join(TEST, ident, name))).astype(np.float32) / 255.
        return torch.from_numpy(a.transpose(2, 0, 1))

    x = img('x.png').unsqueeze(0).to(dev)
    c = torch.load(os.path.join(base, 'c.pt'), map_location='cpu').float().reshape(1, -1).to(dev)
    codes = torch.load(os.path.join(base, 'codes.pt'), map_location='cpu').float()
    codes = codes.reshape(1, *codes.shape[-2:]).to(dev)
    depth = as_depth(torch.load(os.path.join(base, 'depth.pt'), map_location='cpu')).to(dev)
    c_n = torch.load(os.path.join(base, f'c_novel_{view}.pt'), map_location='cpu').float().reshape(1, -1).to(dev)
    yh = img(f'y_hat_novel_{view}.png').unsqueeze(0).to(dev)
    dn = as_depth(torch.load(os.path.join(base, f'depth_novel_{view}.pt'), map_location='cpu')).to(dev)
    c_m = torch.load(os.path.join(base, 'c_mirror.pt'), map_location='cpu').float().reshape(1, -1).to(dev)
    dm = as_depth(torch.load(os.path.join(base, 'depth_mirror.pt'), map_location='cpu')).to(dev)
    x_mirror = torch.flip(x, dims=[3])
    return x, c, codes, depth, c_n, yh, dn, x_mirror, c_m, dm


@torch.no_grad()
def unet_bank_features(img_tensor):
    """AnimateDiff WRITE pass on the MAIN UNet (t=0, no grad): capture each
    attn1's INPUT (norm_hidden_states — exactly what mutual_self_attention
    banks at L224), reshaped to the {channels:[B,C,H,W]} dict the existing
    processor consumes. One representative per channel/resolution."""
    banks = {}
    hooks = []

    def make_hook():
        def hook(module, inputs):
            hs = inputs[0]                      # (b, l, c) post group-norm
            b, l, cdim = hs.shape
            side = int(l ** 0.5)
            fmap = hs.transpose(1, 2).reshape(b, cdim, side, side)
            bucket = banks.setdefault(cdim, [])
            if not any(t.shape[-2:] == fmap.shape[-2:] for t in bucket):
                bucket.append(fmap)
        return hook

    for name in coach.denoising_unet.attn_processors:
        if not name.endswith('attn1.processor'):
            continue
        module = coach.denoising_unet.get_submodule(name.removesuffix('.processor'))
        hooks.append(module.register_forward_pre_hook(make_hook()))


@torch.no_grad()
def sample_manual(cond, mask, codes, feat_primary, feat_extra):
    """Replay of _sample_novel with an injectable EXTRA-feature source."""
    cond_lat = coach.vae.encode(cond * 2.0 - 1.0).latent_dist.mode() * 0.18215
    mask_lat = F.interpolate(mask, size=cond_lat.shape[-2:], mode='nearest')
    conditioning = torch.cat([cond_lat, mask_lat], dim=1)
    bsz = cond_lat.shape[0]
    sched = DDIMScheduler.from_config(coach.noise_scheduler.config)
    sched.set_timesteps(50, device=dev)
    gen = torch.Generator(device=dev).manual_seed(42)
    latents = torch.randn(cond_lat.shape, generator=gen, device=dev) * sched.init_noise_sigma
    wp = coach._wplus_tokens(codes, False)
    cak = {'wplus_features': wp, 'reference_features': feat_primary,
           'reference_features_extra': feat_extra}
    prompt = coach.empty_prompt_embeds.expand(bsz, -1, -1)
    for t in sched.timesteps:
        d, m, u = coach.brushnet(latents, t, encoder_hidden_states=prompt,
                                 brushnet_cond=conditioning, return_dict=False)
        eps = coach.denoising_unet(sample=latents, timestep=t, encoder_hidden_states=prompt,
                                   down_block_add_samples=[s for s in d],
                                   mid_block_add_sample=m,
                                   up_block_add_samples=[s for s in u],
                                   cross_attention_kwargs=cak, return_dict=False)[0]
        latents = sched.step(eps, t, latents).prev_sample
    return (coach.vae.decode(latents / 0.18215).sample / 2 + 0.5).clamp(0, 1)


lines = []
for ident, view in SAMPLES:
    x, c, codes, depth, c_n, yh, dn, x_mirror, c_m, dm = load_sample(ident, view)
    nv = coach._build_novel_view(x, depth, c, c_n, yh, dn,
                                 x_mirror=x_mirror, c_mirror=c_m, depth_mirror=dm)
    mask, cond, anchor = nv['mask'], nv['cond'], nv['anchor']

    feat_primary, _ = coach._extract_reference_features([yh], ['y_hat_novel'])
    feat_refnet_mirror, _ = coach._extract_reference_features([x_mirror], ['x_mirror'])
    bank_mirror = unet_bank_features(x_mirror)

    # R: both sources via RefNet (current architecture, v7a semantics)
    gen_R = sample_manual(cond, mask, codes, feat_primary, feat_refnet_mirror)
    # B: mirror source via MAIN-UNET bank (direction B), primary unchanged
    gen_B = sample_manual(cond, mask, codes, feat_primary, bank_mirror)

    l1R = (gen_R - anchor).abs().mean(dim=1, keepdim=True)
    l1B = (gen_B - anchor).abs().mean(dim=1, keepdim=True)
    holeR = float((l1R * mask).sum() / (mask.sum() + 1e-6))
    holeB = float((l1B * mask).sum() / (mask.sum() + 1e-6))
    diffRB = float((gen_R - gen_B).abs().mean())
    lines.append(f'{ident}_v{view}: hole R(RefNet)={holeR:.4f}  B(UNet-bank)={holeB:.4f}  '
                 f'|R-B| mean pixel diff={diffRB:.4f}')
    print(lines[-1])
    panel = torch.cat([x[0].cpu(), yh[0].cpu(), cond[0].cpu(),
                       mask[0].expand(3, -1, -1).cpu(), anchor[0].cpu(),
                       gen_R[0].cpu(), gen_B[0].cpu()], dim=2)
    vutils.save_image(panel, os.path.join(OUT, f'bankgate_{ident}_v{view}.png'))

with open(os.path.join(OUT, 'bankgate_metrics.txt'), 'w') as f:
    f.write('panels: x | y_hat_novel | cond | mask | anchor | R_RefNet_mirror | B_UNet-bank_mirror\n\n')
    f.write('\n'.join(lines) + '\n')
print(f'\n[gate] saved -> {OUT}')

