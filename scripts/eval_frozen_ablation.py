# Frozen-weight causal ablation (PROJECT_HISTORY §5.3 methodology #2).
#
# Loads the Step-1 checkpoint (iteration_0009999.pt), freezes everything, and
# samples the SAME validation identity with the SAME seed/sampler while
# swapping ONE condition at a time. Locates which branch causes the degraded
# novel-view output (val hole L1 rose 0.05 -> 0.20 during training).
#
# Variants:
#   A as_trained      — everything on (reproduces val_step9999 panel 6)
#   B no_wplus        — drop W+ tokens from attn2
#   C no_refnet       — drop RefNet features from attn1
#   D no_wplus_no_ref — drop both
#   E brushnet_reset  — official pretrained BrushNet, trained processors kept
#   F pure_pretrained — BrushNet reset + no W+ + no Ref (pipeline floor)
#
# Usage (from repo root):
#   python scripts/eval_frozen_ablation.py <ckpt_path>

import os
import sys

import torch
import torch.nn.functional as F

sys.path.append('.')
sys.path.append('..')

from omegaconf import OmegaConf
from diffusers import DDIMScheduler, BrushNetModel

from training.coach_inpainting_diffusion import Coach

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    './experiments/train_inpainting_diffusion/[20260823-123706]_step1_baseline_nomirror/checkpoints/iteration_0009999.pt'
EXP = os.path.dirname(os.path.dirname(os.path.abspath(CKPT)))
CFG = os.path.join(EXP, 'config.yaml')

opts = OmegaConf.load(CFG)
opts.checkpoint_path = os.path.abspath(CKPT)
opts.exp_dir = EXP
opts.smoke.check = False

coach = Coach(opts)  # __init__ -> resume() from the checkpoint
coach.denoising_unet.eval()

# ---- same identity as validate(): first batch of the unshuffled test loader
batch = next(iter(coach.test_dataloader))
b = coach._parse_real_batch(batch)
x, c, codes, depth = b['x'], b['c'], b['codes'], b['depth']
c_novel, y_hat_novel, depth_novel = b['c_novel'], b['y_hat_novel'], b['depth_novel']

warp_img, vis_mask, _ = coach.warper.forward_warp(img1=x, depth1=depth, c1=c, c2=c_novel)
mask = 1.0 - vis_mask
cond = warp_img * (1.0 - mask) + y_hat_novel * mask
inv_warp, inv_valid = coach.warper_ext.inverse_warp(
    img2=x, depth1=depth_novel, depth2=depth, c1=c_novel, c2=c)
vis_eff = inv_valid.clamp(0, 1) * (1.0 - mask)
anchor = inv_warp * vis_eff + y_hat_novel * (1.0 - vis_eff)

pretrained_brushnet = BrushNetModel.from_pretrained(
    opts.paths.brushnet, torch_dtype=torch.float32).to(coach.device).eval()
trained_brushnet_sd = {k: v.clone() for k, v in coach.brushnet.state_dict().items()}


@torch.no_grad()
def sample(use_wplus, use_ref, seed=42, num_steps=50):
    cond_latents = coach.vae.encode(cond * 2.0 - 1.0).latent_dist.mode() * 0.18215
    mask_latents = F.interpolate(mask, size=cond_latents.shape[-2:], mode='nearest')
    conditioning = torch.cat([cond_latents, mask_latents], dim=1)
    bsz = cond_latents.shape[0]

    scheduler = DDIMScheduler.from_config(coach.noise_scheduler.config)
    scheduler.set_timesteps(num_steps, device=coach.device)
    generator = torch.Generator(device=coach.device).manual_seed(seed)
    latents = torch.randn(cond_latents.shape, generator=generator,
                          device=coach.device) * scheduler.init_noise_sigma

    cross_attention_kwargs = {}
    if use_wplus:
        cross_attention_kwargs['wplus_features'] = coach.w_mapper(codes)
    if use_ref:
        ref_feat, _ = coach._extract_reference_features([y_hat_novel], ['y_hat_novel'])
        cross_attention_kwargs['reference_features'] = ref_feat
    prompt = coach.empty_prompt_embeds.expand(bsz, -1, -1)

    for t in scheduler.timesteps:
        down_res, mid_res, up_res = coach.brushnet(
            latents, t, encoder_hidden_states=prompt,
            brushnet_cond=conditioning, return_dict=False)
        eps = coach.denoising_unet(
            sample=latents, timestep=t, encoder_hidden_states=prompt,
            down_block_add_samples=[s for s in down_res],
            mid_block_add_sample=mid_res,
            up_block_add_samples=[s for s in up_res],
            cross_attention_kwargs=cross_attention_kwargs or None,
            return_dict=False)[0]
        latents = scheduler.step(eps, t, latents).prev_sample
    return (coach.vae.decode(latents / 0.18215).sample / 2 + 0.5).clamp(0, 1)


def reset_brushnet(pretrained: bool):
    if pretrained:
        coach.brushnet.load_state_dict(pretrained_brushnet.state_dict(), strict=True)
    else:
        coach.brushnet.load_state_dict(trained_brushnet_sd, strict=True)


VARIANTS = [
    ('A_as_trained',      True,  True,  False),
    ('B_no_wplus',        False, True,  False),
    ('C_no_refnet',       True,  False, False),
    ('D_no_wplus_no_ref', False, False, False),
    ('E_brushnet_reset',  True,  True,  True),
    ('F_pure_pretrained', False, False, True),
]

rows, stats = [], []
for name, use_w, use_r, reset in VARIANTS:
    if reset:
        reset_brushnet(True)
    gen = sample(use_w, use_r)
    if reset:
        reset_brushnet(False)
    l1 = (gen - anchor).abs().mean(dim=1, keepdim=True)
    hole = float((l1 * mask).sum() / mask.sum())
    full = float(l1.mean())
    stats.append((name, full, hole))
    print(f'[variant {name:18s}] full={full:.4f}  hole={hole:.4f}')
    rows.append(gen[0].cpu())

# ---- grid: header row (inputs) + one row per variant
import torchvision.utils as vutils
from PIL import Image, ImageDraw

header = torch.cat([x[0].cpu(), y_hat_novel[0].cpu(), cond[0].cpu(),
                    mask[0].expand(3, -1, -1).cpu(), anchor[0].cpu(), rows[0]], dim=2)
grid = header.unsqueeze(0)
for r in rows[1:]:
    # 6-tile row: five blanks + the generated panel (matches header width)
    blank6 = torch.zeros(3, r.shape[-2], header.shape[-1] - r.shape[-1])
    grid = torch.cat([grid, torch.cat([blank6, r], dim=2).unsqueeze(0)], dim=0)

out = os.path.join(EXP, f'frozen_ablation_step{coach.global_step:07d}.png')
vutils.save_image(grid, out, nrow=1)

img = Image.open(out)
draw = ImageDraw.Draw(img)
labels = ['x | yhat_novel | cond | mask | anchor | A_as_trained'] + \
         [f'{n}  (full={f:.3f} hole={h:.3f})' for n, f, h in stats[1:]]
for i, lab in enumerate(labels):
    draw.text((8, 8 + i * 24), lab, fill=(255, 255, 0))
img.save(out)
print(f'\nsaved grid -> {out}')
print('\n==== summary (lower hole = better novel generation) ====')
for n, f, h in stats:
    print(f'  {n:18s} full={f:.4f}  hole={h:.4f}')

