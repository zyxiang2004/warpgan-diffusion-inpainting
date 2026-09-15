# -*- coding: utf-8 -*-
"""SDEdit / partial-noise ladder (InvSR 'Partial noise Prediction' applied to
the anchor). MOTIVATION (user feedback 2026-09-13): W3 outputs are 'vintage
oil-paint, grayish, kraft-paper background' — matching NEITHER the EG3D render
NOR the anchor: the style comes from the frozen SD prior dominating the
50-step trajectory from PURE NOISE (the high-t段 is prior territory; our
supervision lives at low t). TEST: start sampling from the NOISED ANCHOR at
t in {100, 300, 500} instead of pure noise. If the trajectory hypothesis is
right, lower t_start should shift style toward the photo domain with no
training at all. Precedent note: the 2026-09-01 SDEdit probe (§8.33) failed
under the v12-era recipe; THIS recipe's pass1 eps-training noises the ANCHOR
itself, so sde-from-anchor is in-distribution — worth the 15-min retest.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/eval_sde_ladder.py [ckpt_dir]
Output: train_logs/sde_ladder.png + per-column L1 vs anchor.
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from omegaconf import OmegaConf
import torchvision.utils as vutils
from diffusers import DDIMScheduler

DEV = 'cuda:0'
DATA_ROOT = './data/celeba-hq_1000_static_rebalanced'
OUT = 'train_logs/sde_ladder.png'
IDS = ['000004', '000550', '001418']     # fixed ids used across our evals
VIEW = 1
OUT = 'train_logs/sde_prompt_ladder.png'
SD_PATH = './pretrained_models/AI-ModelScope/stable-diffusion-v1-5'
# InvSR trainer.py L44-47 verbatim — note the negative prompt EXPLICITLY
# steers away from 'oil painting' (our observed failure mode).
POS_TEXT = ('Cinematic, high-contrast, photo-realistic, 8k, ultra HD, '
            'meticulous detailing, hyper sharpness, perfect without deformations')
NEG_TEXT = ('Low quality, blurring, jpeg artifacts, deformed, over-smooth, '
            'cartoon, noisy, painting, drawing, sketch, oil painting')


def encode_prompts(coach):
    """Load the CLIP text encoder (the coach keeps only the EMPTY embeds),
    encode the InvSR quality prompts, then free it."""
    from transformers import CLIPTokenizer, CLIPTextModel
    tok = CLIPTokenizer.from_pretrained(SD_PATH, subfolder='tokenizer')
    enc = CLIPTextModel.from_pretrained(SD_PATH, subfolder='text_encoder').to(DEV).eval()
    out = {}
    with torch.no_grad():
        for name, txt in (('pos', POS_TEXT), ('neg', NEG_TEXT)):
            ti = tok([txt], padding='max_length', max_length=tok.model_max_length,
                     truncation=True, return_tensors='pt')
            out[name] = enc(ti.input_ids.to(DEV))[0]
    out['empty'] = coach.empty_prompt_embeds
    del tok, enc
    torch.cuda.empty_cache()
    return out


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


def build_coach(ckpt):
    exp = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    cfg = os.path.join(exp, 'config.yaml')
    if not os.path.exists(cfg):
        cfg = os.path.join(exp, 'config_resume.yaml')
    opts = OmegaConf.load(cfg)
    opts.checkpoint_path = os.path.abspath(ckpt)
    opts.exp_dir = exp
    opts.smoke.check = False
    opts.device = DEV
    from training.coach_inpainting_diffusion import Coach
    return Coach(opts)
@torch.no_grad()
def sample_from(coach, nv, case, t_start, num_steps=50, seed=42,
                prompt_embeds=None, neg_embeds=None, cfg=1.0):
    """Coach._sample_novel mirrored + two InvSR levers:
    (a) SDEdit start from the noised ANCHOR for t_start>0;
    (b) prompt conditioning / CFG: cond pass uses prompt_embeds (pos or empty);
        cfg>1 adds a negative pass (InvSR _negative) with eps combined the
        standard way. BrushNet add-samples are computed ONCE and shared by
        both passes (same latents+condition inputs); W+/bank injections ride
        BOTH passes (identity conditioning is not part of the CFG contrast)."""
    cond_latents = coach.vae.encode(nv['cond'] * 2.0 - 1.0).latent_dist.mode() * 0.18215
    m_brush = nv['mask'] if nv.get('mask_cond') is None else nv['mask_cond']
    mask_latents = F.interpolate(m_brush, size=cond_latents.shape[-2:], mode='nearest')
    conditioning = torch.cat([cond_latents, mask_latents], dim=1)
    bsz = cond_latents.shape[0]

    sched = DDIMScheduler.from_config(coach.noise_scheduler.config)
    generator = torch.Generator(device=DEV).manual_seed(seed)
    noise = torch.randn(cond_latents.shape, generator=generator, device=DEV)
    if t_start > 0:
        anchor_lat = coach.vae.encode(nv['anchor'] * 2.0 - 1.0).latent_dist.mode() * 0.18215
        tt = torch.tensor([t_start], device=DEV)
        latents = sched.add_noise(anchor_lat, noise, tt)
        # manual descending integer timesteps t_start -> 0
        timesteps = sorted(set(np.linspace(t_start, 0, num_steps).round().astype(int).tolist()),
                           reverse=True)
    else:
        sched.set_timesteps(num_steps, device=DEV)
        latents = torch.randn(cond_latents.shape, generator=generator, device=DEV) \
            * sched.init_noise_sigma
        timesteps = None  # use the scheduler's own loop below

    wplus_features = coach._wplus_tokens(case['codes'], False)
    refs, tags = coach._ref_inputs_real(case['yh'], case['x_mirror'])
    ref_feat, ref_extra = coach._extract_reference_features(refs, tags)
    cak = {}
    if wplus_features is not None:
        cak['wplus_features'] = wplus_features
    if ref_feat is not None:
        cak['reference_features'] = ref_feat
        if ref_extra is not None:
            cak['reference_features_extra'] = ref_extra
    prompt = coach.empty_prompt_embeds.expand(bsz, -1, -1)

    if prompt_embeds is None:
        prompt_embeds = prompt
    if cfg > 1.0 and neg_embeds is None:
        cfg = 1.0

    def unet_eps(lats, t, cond_embeds):
        down_res, mid_res, up_res = coach.brushnet(
            lats, t, encoder_hidden_states=prompt,
            brushnet_cond=conditioning, return_dict=False)
        eps = coach.denoising_unet(
            sample=lats, timestep=t, encoder_hidden_states=cond_embeds,
            down_block_add_samples=[s for s in down_res],
            mid_block_add_sample=mid_res,
            up_block_add_samples=[s for s in up_res],
            cross_attention_kwargs=cak, return_dict=False)[0]
        return eps

    def step_eps(lats, t):
        eps_c = unet_eps(lats, t, prompt_embeds)
        if cfg > 1.0:
            eps_u = unet_eps(lats, t, neg_embeds)
            eps_c = eps_u + cfg * (eps_c - eps_u)
        return eps_c

    if timesteps is None:
        # exactly Coach._sample_novel's loop (pure-noise reference column)
        for t in sched.timesteps:
            eps = step_eps(latents, t)
            latents = sched.step(eps, t, latents).prev_sample
    else:
        # manual DDIM (eta=0) over OUR timestep list — same update the
        # scheduler applies, but with a custom step size so a t_start=300
        # trajectory really takes 50 even steps down to 0.
        ac = sched.alphas_cumprod
        for i, t in enumerate(timesteps):
            tt = torch.tensor(t, device=DEV)
            eps = step_eps(latents, tt)
            ab = ac[t]
            ab_prev = ac[timesteps[i + 1]] if i + 1 < len(timesteps) \
                else torch.ones_like(ab)
            x0 = (latents - (1.0 - ab).sqrt() * eps) / ab.sqrt()
            latents = ab_prev.sqrt() * x0 + (1.0 - ab_prev).sqrt() * eps

    return (coach.vae.decode(latents / 0.18215).sample / 2 + 0.5).clamp(0, 1)


def main():
    torch.manual_seed(0)
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(
        './experiments/train_inpainting_diffusion/*_v18w3_w3p_30k/checkpoints'))[-1]
    cands = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pt'))
    ckpt = os.path.join(ckpt_dir, cands[-1])
    print(f'[ladder] coach from {ckpt}', flush=True)
    coach = build_coach(ckpt)

    rows = []
    embeds = encode_prompts(coach)
    VARIANTS = [
        ('ddim50_empty', 0, 'empty', 1.0),
        ('ddim50_pos',   0, 'pos',   1.0),
        ('ddim50_cfg7',  0, 'pos',   7.0),
        ('sde300_empty', 300, 'empty', 1.0),
        ('sde300_pos',   300, 'pos',   1.0),
        ('sde300_cfg7',  300, 'pos',   7.0),
    ]
    for ident in IDS:
        case = load_case(ident, VIEW)
        nv = coach._build_novel_view(
            case['x'], case['depth'], case['c'], case['c_n'], case['yh'], case['dn'],
            x_mirror=case['x_mirror'], c_mirror=case['c_m'], depth_mirror=case['d_m'])
        cols = [case['x'][0].cpu(), nv['anchor'][0].cpu()]
        names = ['photo', 'anchor']
        for nm, t0, pk, cfg in VARIANTS:
            g = sample_from(coach, nv, case, t0,
                            prompt_embeds=embeds[pk],
                            neg_embeds=embeds['neg'], cfg=cfg)
            l1 = float(F.l1_loss(g, nv['anchor']))
            print(f'[ladder] {ident} {nm}: L1_vs_anchor={l1:.4f}', flush=True)
            cols.append(g[0].cpu())
            names.append(nm)
        rows.append(torch.stack(cols, 0))
    grid = torch.cat(rows, 0)
    vutils.save_image(grid, OUT, nrow=len(names), padding=2)
    print('columns: ' + ' | '.join(names))
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()
