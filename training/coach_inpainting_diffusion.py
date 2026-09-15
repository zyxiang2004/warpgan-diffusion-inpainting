# -*- coding: utf-8 -*-
"""Diffusion port of the orig WarpGAN SVINet trainer (coach_inpainting_static.py).

Faithful translation of the FFC-ResNet inpaintor to BrushNet + SD1.5 following
docs/ORIG_FAITHFUL_PORT_SPEC.md (teacher-approved final spec, 2026-08-23):

Batch A (synthetic, 1:1 alternation like orig coach L489-560):
    condition = hybrid[warp(src) visible + target_hat hole] + mask  -> BrushNet 5ch
    RefNet input = target_hat (clean complete inversion render) [| x_mirror]
    W+ tokens = mapper(codes) -> attn2 (fixed tanh(3) gate)
    losses: eps-MSE (full frame) + low-t x0 pixel loss vs target_img
            (L1*10 + ResNetPL*30 + ID*0.5) + latent closure (*0.1)

Batch B (real FFHQ, dual pass like orig coach L207-305):
    pass 1 (novel): condition = hybrid[warp(x) + y_hat_novel hole];
        RefNet input = y_hat_novel; weak eps anchor (spec §5.3):
        visible target = Warper.inverse_warp(x) @1.0, hole target = y_hat_novel @0.1.
    pass 2 (source, the ONLY real-photo supervision, orig warp_pred=False):
        warp_warp_img = forward_warp(pass1 hybrid condition, c_novel->c);
        condition = hybrid[warp_warp_img + y_hat hole]; RefNet input = y_hat;
        losses: eps-MSE (full frame, with_mask=False) + t<200 x0 single-step
        decode pixel loss vs real x + latent closure (*0.1).

Everything else keeps orig semantics: full-frame generation (no pasting),
raw splatting masks (no erosion), full timestep range [0,1000), GAN/FM = 0.
The orig file is untouched — this coach is added alongside for diffing.

==============================================================================
ARCHITECTURE MAP — v16 current state (2026-09-08). The historical spec above
is kept verbatim for port-diffing; the loss weights it lists are NOT the
running recipe (see the "v16 running recipe" note at the bottom).
==============================================================================

MODULE ASSEMBLY — Coach.__init__, in build order:
  frozen     vae (SD1.5, pixel losses backprop THROUGH the decoder, ckpt'd)
             denoising_unet (SD1.5 UNet, gradient-checkpointed, .train() only
               so checkpointing engages — numerically inert, no BN/dropout)
             empty CLIP prompt embeds (text encoder deleted after encoding '')
             gan = WplusNet (GOAE: codes, y_hat/y_hat_novel renders, depths)
             warper (forward splat, utils/warp/Splatting.py)
             warper_ext (inverse warp, utils/warp/splatting_ext.py)
  trainable  brushnet (official inpainting ckpt — the ONLY big module)
             attention processors on EVERY SD attn layer (see below)
             w_mapper = WProjModel (W+ [14,512] -> [18,768] tokens)
             [default off] discriminator (make_discriminator) — v17 FM/adv

FORWARD GRAPH — _diffusion_forward() is the single choke point:
  condition_img -> VAE latents; + downsampled mask -> 5ch BrushNet cond
  x -> VAE latents -> add_noise(t ~ U[0,1000)) -> noisy_latents
  ref images -> no-grad t=0 pass on the SAME frozen UNet -> per-layer attn1
    banks {channels: [B,C,H,W]} (_write_bank / _extract_reference_features)
  brushnet(noisy, cond) -> down/mid/up add-samples
  denoising_unet(noisy, add_samples, cak{wplus_features, reference_features,
    reference_features_extra}) -> eps_pred   (training target = the noise)

TRAIN LOOP — train(), real/synth 1:1 alternation (orig L489-560 shape):
  _forward_real : pass1 novel dual-band anchor (photo evidence full band /
                  blind low band vs render) | pass2 source: eps-MSE + t<200
                  x0-patch L1 photo reflow (v16)
  _forward_synth: eps-MSE structure band ONLY (texture band weight = 0 —
                  "paint-oil" red line: render texture is never ground truth)
  W+ red line: w_mapper / attn2-QKV receive gradients on SYNTH updates only
                  (_zero_wplus_grads clears them on real updates)

ATTENTION PROTOCOL — ReferenceAttentionProcessor (models/referencenet/):
  attn1 (self)  reference texture branch: K/V/O adapters over banked features;
                primary = x_mirror photo (real pass1, real_primary=mirror),
                extra = y_hat_novel render; source-separating mirror K/V (v7a);
                local-window texture band (v11)
  attn2 (cross) W+ identity branch: tokens concat with empty prompt, FIXED
                tanh(3) gate; wplus_mode = off / frozen / s1_mapper / s2_qkv

v16 running recipe (CLI deltas in scripts/v16_resume.sh): mirror_anchor=1.0 /
blind_struct=1.0 / synth texture=zero / pixel L1 x2 real-pass2-only patches /
all GAN-FM-id-latent-PL weights 0 / cond soften erode3+blur21.
"""

import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diffusers import DDPMScheduler, DDIMScheduler, AutoencoderKL, UNet2DConditionModel, BrushNetModel
from transformers import CLIPTextModel, CLIPTokenizer

# v8: the standalone 860M RefNet is DELETED — reference features come from the
# main UNet's own attn1 activations banked during a no-grad write pass
# (AnimateDiff mutual-self-attention paradigm; gate-checked in STEP0 §8.21).
from models.mapper.w_proj import WProjModel
from models.referencenet.attention_processor import ReferenceAttentionProcessor
from models.saicinpainting.utils import set_requires_grad
from models.saicinpainting.training.losses.perceptual import ResNetPL
from models.saicinpainting.training.modules import make_discriminator
from models.saicinpainting.training.losses.adversarial import make_discrim_loss
from models.saicinpainting.training.losses.feature_matching import feature_matching_loss
from models.wplusnet import WplusNet
from criteria import id_loss

from datasets.dataset_inpainting_static import ImageFolderDataset
from datasets.dataset_inpainting_synth_static import SynthImageFolderDataset
from utils.warp.Splatting import Warper
from utils.warp.splatting_ext import WarperExt


def patch_goae_device_compat():
    """torch>=2.6 device-strictness shim (runtime only; orig files untouched).

    models/goae/swin_transformer.py L161 clamps a CUDA parameter with a CPU
    scalar tensor: torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1./0.01))).
    Older torch silently accepted the mixed devices; torch 2.8 raises. The latent
    closure (spec §5.2, first time ported) is the only caller of this path.
    Replacing the tensor bound with a python float is numerically identical.
    """
    import models.goae.swin_transformer as st

    if getattr(st.WindowAttention, '_warpgan_device_patched', False):
        return
    _orig_forward = st.WindowAttention.forward

    def _forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias,
                                  torch.zeros_like(self.v_bias, requires_grad=False),
                                  self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        # float bound instead of a CPU tensor (the one-line compat fix)
        logit_scale = torch.clamp(self.logit_scale, max=4.605170185988092).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + 16 * torch.sigmoid(relative_position_bias).unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) \
                + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        else:
            attn = attn.reshape(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        return x

    st.WindowAttention.forward = _forward
    st.WindowAttention._warpgan_device_patched = True
    print('[compat] patched models/goae WindowAttention.forward '
          '(torch>=2.6 CPU-bound clamp fix; numerically identical)')


patch_goae_device_compat()



class Coach:
    def __init__(self, opts):
        self.opts = opts
        self.device = opts.device
        self.global_step = 0

        # ---------------- effective flags ----------------
        self.use_synth = bool(opts.data.synth.able)
        self.use_mirror = bool(opts.reference.use_mirror)
        self.wplus_mode = str(opts.wplus.mode)          # off | frozen | s1_mapper | s2_qkv
        assert self.wplus_mode in ('off', 'frozen', 's1_mapper', 's2_qkv'), self.wplus_mode
        self.use_wplus = self.wplus_mode != 'off'
        self.brushnet_init = str(getattr(opts.brushnet, 'init', 'pretrained'))
        assert self.brushnet_init in ('pretrained', 'from_unet', 'random'), self.brushnet_init
        if self.use_mirror and not bool(opts.reference.use):
            raise ValueError('reference.use_mirror=True requires reference.use=True')
        self.smoke_check = bool(opts.smoke.check)
        self.fix_timestep = opts.smoke.fix_timestep
        self.pixel_max_t = int(opts.losses.pixel.max_timestep)
        self.x0_clip = float(opts.losses.pixel.x0_clip)
        self.pixel_res = int(opts.losses.pixel.resolution)
        # v16: random latent-patch decode size (0 = legacy full-frame decode)
        self.pixel_patch_lat = int(getattr(opts.losses.pixel, 'patch_latent', 0)
                                   or 0)
        # v16: the photo-ref pixel backflow is REAL-pass2-only (synth targets
        # are EG3D renders — the paint-oil red line)
        self.pixel_apply_synth = bool(
            getattr(opts.losses.pixel, 'apply_to_synth', False))
        self.ref_gate_init = float(opts.reference.gate_init)
        self.mirror_anchor = bool(getattr(opts.losses.real_pass1, 'mirror_anchor', False))
        self.synth_texture = str(getattr(opts.losses, 'synth_texture', 'full'))
        assert self.synth_texture in ('full', 'zero', 'lowpass'), self.synth_texture
        self.synth_lp_sigma = float(getattr(opts.losses, 'synth_lowpass_sigma', 2.0))
        self.real_primary_mirror = str(getattr(
            getattr(opts, 'reference', None), 'real_primary', 'render')) == 'mirror'
        # v11: local texture-band retrieval (activates the formerly DEAD
        # reference_local_scale pathway — code audit 2026-08-27)
        self.local_window_k = int(getattr(opts.reference, 'local_window_k', 0))
        self.local_gate_init = float(getattr(opts.reference, 'local_gate_init', 0.25))
        # v12: blind-region structure-band anchor (user's three-partition plan:
        # photo evidence -> full band; blind -> LOW band only vs the FULL SHARP
        # render anchor — geometry anchored, EG3D texture band zero gradient)
        self.blind_struct_weight = float(getattr(
            getattr(opts.losses, 'real_pass1', None), 'blind_struct_weight', 0.0))
        # v18 STRUCT arm: when False, the real batch runs pass2 ONLY — no
        # pass1 forward and no pass1 supervision (the orig coach NEVER
        # supervised pred_novel on real data; the anchor's blind region is
        # fabricated evidence — renders — and teaching it is the suspected
        # oil-paint/sandpaper saboteur). cond1 is still constructed because
        # pass2's condition derives from it. Inference (validate/_sample_novel)
        # is UNCHANGED — the novel-view pipeline still runs at eval time.
        self.real_pass1_forward = bool(getattr(
            getattr(opts.losses, 'real_pass1', None), 'forward', True))
        # v18.6: pass1 timestep domain = the DEPLOYED inference domain. With
        # SDEdit-300 inference the trajectory lives in t∈[0,300]; pass1
        # training beyond that teaches prior-dominated generation that is
        # never deployed (the vintage-oil-paint source). 0 = full range.
        self.real_pass1_t_max = int(getattr(
            getattr(opts.losses, 'real_pass1', None), 't_max', 0) or 0)
        # v18.7: median cleanup of the supervision anchor (impulse noise from
        # inverse-warp coverage gaps / sub-pixel misalignment). 0 = off.
        self.anchor_median = int(getattr(
            getattr(opts.losses, 'real_pass1', None), 'anchor_median', 0) or 0)
        # v18.8 (W5): AnyDoor-style high-frequency conditioning channel.
        # Sobel edge-weighted image of the texture reference (mirror photo),
        # warped to the target view, concatenated into BrushNet conditioning
        # (5ch → 8ch). Zero-initialized new channels = safe from-scratch.
        self.hf_enable = bool(getattr(
            getattr(opts, 'hf', None), 'enable', False))
        self.hf_thresh = float(getattr(
            getattr(opts, 'hf', None), 'thresh', 50.0)) / 255.0  # AnyDoor: 50 on 0-255
        # v18.8 (W5): DreamBooth-style prior preservation. Every N steps,
        # run one extra forward where condition=target=real photo (identity
        # mapping), teaching BrushNet 'photo condition → photo output'.
        self.prior_preserve_freq = int(getattr(
            getattr(opts, 'prior_preserve', None), 'freq', 0) or 0)
        # v18.6: inference starts from the noised ANCHOR at t_start (InvSR
        # 'Partial noise Prediction'; user-verified domain fix 2026-09-13).
        # 0 = legacy pure-noise 50-step chain.
        _inf = getattr(opts, 'infer', None)
        self.infer_t_start = int(getattr(_inf, 't_start', 0) or 0) if _inf is not None else 0
        # v10 condition-side softening (orig process_mask port; §8.26)
        _w = getattr(opts, 'warp', None)
        self.cond_erode_kernel = int(getattr(_w, 'cond_erode_kernel', 0))
        self.cond_blur_kernel = int(getattr(_w, 'cond_gaussian_blur_kernel', 0))
        self.cond_soften = (self.cond_erode_kernel > 0) or (self.cond_blur_kernel > 0)

        self._print_effective_config()

        # ---------------- frozen SD base ----------------
        sd_path = opts.paths.sd
        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder='vae').to(self.device).eval()
        set_requires_grad(self.vae, False)
        # pixel losses backprop THROUGH the VAE decoder -> checkpoint it.
        # v16 fix: diffusers only checkpoints in TRAIN mode (same argument as
        # the UNet below); with .eval() the decoder activations stayed live
        # and the v16 pixel path peaked 22.4-22.6 GiB > 22 GiB budget. The VAE
        # has no BN/dropout, so train() is numerically inert.
        self.vae.enable_gradient_checkpointing()
        self.vae.train()
        self.noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder='scheduler')
        assert int(self.noise_scheduler.config.num_train_timesteps) == 1000, \
            'training timestep range must be [0,1000) (v3.1 lesson)'
        self.val_scheduler = DDIMScheduler.from_pretrained(sd_path, subfolder='scheduler')
        self.denoising_unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder='unet').to(self.device).eval()
        set_requires_grad(self.denoising_unet, False)
        # Gradients flow THROUGH the frozen UNet into BrushNet; checkpointing it
        # trades ~30% step time for several GiB of activation memory (needed to
        # stay under the 22GiB Step-0 budget). SD UNet has no BN/dropout, so
        # train() is numerically inert — diffusers only checkpoints in train mode.
        self.denoising_unet.enable_gradient_checkpointing()
        self.denoising_unet.train()

        # trainable-parameter routing (populated by processor install)
        self.wplus_qkv_params = []      # s2 only; grads cleared on real updates
        self.wplus_mapper_params = []   # s1/s2; grads cleared on real updates
        self.refnet_adapter_params = []
        self.mirror_attention_procs = []  # v7: procs carrying the attn-share gauge
        self.mirror_kv_params = []        # v7a: source-separating mirror K/V
        self._install_attention_processors()

        # ---------------- empty prompt (W+ rides attn2 in parallel with it) ----------------
        tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder='tokenizer')
        text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder='text_encoder').to(self.device).eval()
        text_inputs = tokenizer('', padding='max_length', max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors='pt')
        with torch.no_grad():
            self.empty_prompt_embeds = text_encoder(text_inputs.input_ids.to(self.device))[0]
        # v18.5: optional CONSTANT prompt (InvSR trains with a positive QUALITY
        # prompt, not the empty string — trainer.py L44-47, L862/875; teacher ②
        # also intended W+ tokens to align WITH text features). Default preset
        # '' reproduces the historical empty-prompt behavior EXACTLY.
        _pc = getattr(opts, 'prompt', None)
        _preset = str(getattr(_pc, 'preset', '') or '') if _pc is not None else ''
        _ptext = str(getattr(_pc, 'text', '') or '') if _pc is not None else ''
        if _preset == 'invsr':
            _ptext = ('Cinematic, high-contrast, photo-realistic, 8k, ultra HD, '
                      'meticulous detailing, hyper sharpness, perfect without deformations')
        if _ptext:
            ti = tokenizer(_ptext, padding='max_length',
                           max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors='pt')
            with torch.no_grad():
                self.empty_prompt_embeds = text_encoder(ti.input_ids.to(self.device))[0]
            print(f'[v18.5] CONSTANT prompt ACTIVE (preset={_preset}): "{_ptext[:70]}..."')
        del tokenizer, text_encoder
        torch.cuda.empty_cache()

        # ---------------- v8: reference features via MAIN-UNET bank ----------------
        # Direction-B refactor (gate-checked 2026-08-25, STEP0 §8.21): the frozen
        # 860M RefNet duplicates the SD UNet's own attn1 activations — pixel-level
        # equivalence verified (B slightly better on all 3 probes). DELETED.
        # Reference images now run through the MAIN (frozen) UNet once (no_grad,
        # t=0) and each attn1 layer banks its norm_hidden_states, which the same
        # layer retrieves during denoising — the AnimateDiff mutual-self-attention
        # paradigm in its original form (models/referencenet/mutual_self_attention.py).
        self._bank_hooks = []


        # ---------------- BrushNet (the ONLY big trainable module) ----------------
        # init=pretrained  : official inpainting checkpoint (default, spec §4.1)
        # init=from_unet   : the OFFICIAL "from scratch" — BrushNet's published
        #                    training itself starts from SD UNet encoder weights
        #                    (diffusers BrushNetModel.from_unet: copies down/mid/up
        #                    + time emb; conv_in_condition = unet.conv_in into both
        #                    the sample & condition halves, mask channel zeroed).
        #                    Uses ONLY SD weights — satisfies "除了SD不用其他预训练".
        # init=random      : fully random weights (no official precedent; expected
        #                    to need far more steps to match).
        if self.brushnet_init == 'pretrained':
            self.brushnet = BrushNetModel.from_pretrained(
                self.opts.paths.brushnet, torch_dtype=torch.float32).to(self.device)
            print('[BrushNet] init: official pretrained inpainting checkpoint')
        elif self.brushnet_init == 'from_unet':
            self.brushnet = BrushNetModel.from_unet(
                self.denoising_unet, conditioning_channels=5,
                brushnet_conditioning_channel_order='rgb').to(self.device)
            print('[BrushNet] init: from_unet (SD UNet encoder copy — official '
                  '"train from scratch" recipe; no non-SD pretrained weights)')
        else:
            import json as _json
            with open(os.path.join(self.opts.paths.brushnet, 'config.json')) as f:
                cfg = _json.load(f)
            for drop in ('_class_name', '_diffusers_version', '_name_or_path', '_attn_implementation'):
                cfg.pop(drop, None)
            self.brushnet = BrushNetModel.from_config(cfg).to(self.device)
            print('[BrushNet] init: RANDOM (no pretrained weights at all)')
        self.brushnet.train()
        set_requires_grad(self.brushnet, True)
        if bool(self.opts.brushnet.gradient_checkpointing):
            self.brushnet.enable_gradient_checkpointing()
        # v18.8: expand BrushNet conditioning 5ch → 8ch (HF channel)
        if self.hf_enable:
            self._expand_brushnet_conditioning(extra_ch=3)

        # ---------------- W+ mapper ----------------
        if self.use_wplus:
            self.w_mapper = WProjModel(cross_attention_dim=768, embeddings_dim=512).to(self.device)
            if self.wplus_mode == 'frozen' or bool(self.opts.wplus.init_from_stage_c):
                ckpt = torch.load(self.opts.paths.wplus_stage_c, map_location='cpu', weights_only=False)
                state = ckpt['w_mapper_state_dict'] if 'w_mapper_state_dict' in ckpt else ckpt
                self.w_mapper.load_state_dict(state, strict=True)
                print(f"[W+] loaded Stage-C mapper weights: {self.opts.paths.wplus_stage_c}")
            if self.wplus_mode == 'frozen':
                self.w_mapper.eval()
                set_requires_grad(self.w_mapper, False)
            else:  # s1_mapper / s2_qkv: mapper trains, but ONLY on synth updates
                self.w_mapper.train()
                self.wplus_mapper_params = [p for p in self.w_mapper.parameters()]
        else:
            self.w_mapper = None
            print('[W+] mode=off — no W+ tokens injected (pure SD + conditions)')

        # ---------------- frozen aux nets ----------------
        if float(self.opts.losses.latent.weight) > 1e-5:
            self.gan = WplusNet(self.opts.gan).to(self.device).eval()
            set_requires_grad(self.gan, False)
            # encoder_forward (latent closure) never touches the EG3D synthesis
            # decoder — park it on the CPU to reclaim GPU memory
            self.gan.decoder.to('cpu')
            torch.cuda.empty_cache()
        if float(self.opts.losses.pixel.resnet_pl_weight) > 1e-5:
            self.loss_resnet_pl = ResNetPL(
                weight=float(self.opts.losses.pixel.resnet_pl_weight),
                weights_path=self.opts.losses.pixel.weights_path,
            ).to(self.device).eval()
        if float(self.opts.losses.pixel.id_weight) > 1e-5:
            self.loss_id = id_loss.IDLoss().to(self.device).eval()
        # ---------------- v14: discriminator + FM/adv (orig third layer) -----
        # Faithful port of orig coach L407-448 / L1169-1198 with the two
        # DIFFUSION-MANDATED adjustments documented in STEP0 §8.39:
        #   (a) fake = the x0 single-step decode (t<200 gate) — the FFC
        #       one-shot generator's closest diffusion counterpart (spec §5.2),
        #       at 256px; v13 proved this path 50K-stable with id/latent;
        #   (b) G-side adversarial defaults 0 (FM-first policy). Everything
        #       else is verbatim orig: NLayer PatchGAN, NonSaturatingWithR1
        #       gp_coef=0.001 inline (non-lazy), full-frame D (with_mask=False),
        #       full-frame FM (mask_for_fm=None), D Adam lr=1e-4.
        # SYNTH batches never enter D (deliberate deviation, §8.39): D's real
        # distribution must stay REAL photos — feeding EG3D renders as "real"
        # would teach D that plastic is genuine.
        self.fm_weight = float(self.opts.losses.feature_matching.weight)
        self.adv_weight = float(self.opts.losses.adversarial.weight)
        self.use_gan_fm = (self.fm_weight > 1e-5) or (self.adv_weight > 1e-5)
        # v15: blind high-frequency score preservation (prior takeover)
        _pw = self.opts.losses.get('preserve', None)
        self.preserve_weight = float(_pw.weight) if _pw is not None and 'weight' in _pw else 0.0
        self.teacher_wplus = bool(_pw.teacher_wplus) if _pw is not None and 'teacher_wplus' in _pw else True
        # v17-prime: gate-free x0 distribution supervision (see _x0_losses)
        _x0 = self.opts.losses.get('x0', None)
        self.x0_enable = bool(_x0.enable) if _x0 is not None and 'enable' in _x0 else False
        self.x0_l1_weight = float(_x0.l1_weight) if _x0 is not None and 'l1_weight' in _x0 else 1.0
        self.x0_clip = float(_x0.clip) if _x0 is not None and 'clip' in _x0 else 8.0
        self.x0_res_lat = int(_x0.res_latent) if _x0 is not None and 'res_latent' in _x0 else 32
        # v18.3: InvSR-calibrated mode (CVPR'5 2412.09013). 'renoise' = the
        # v18 wave-1 legacy path (kept verbatim for reproducibility);
        # 'invsr' = latent-space low-t supervision: MSE + FM + small
        # adversarial on the single-step x0 estimate, timestep-conditioned
        # UNet discriminator, hinge loss, D warmup — see _x0_losses_invsr.
        self.x0_mode = str(_x0.mode) if _x0 is not None and 'mode' in _x0 else 'renoise'
        assert self.x0_mode in ('renoise', 'invsr'), self.x0_mode
        self.x0_t_max = int(_x0.t_max) if _x0 is not None and 't_max' in _x0 else 300
        self.x0_w_ldif = float(_x0.w_ldif) if _x0 is not None and 'w_ldif' in _x0 else 1.0
        self.x0_w_lfm = float(_x0.w_lfm) if _x0 is not None and 'w_lfm' in _x0 else 2.0
        self.x0_w_ldis = float(_x0.w_ldis) if _x0 is not None and 'w_ldis' in _x0 else 0.1
        self.x0_dis_warmup = int(_x0.dis_warmup) if _x0 is not None and 'dis_warmup' in _x0 else 3000
        self.x0_latent_bound = float(_x0.latent_bound) if _x0 is not None and 'latent_bound' in _x0 else 10.0
        # v18.4: InvSR's finetuned latent-LPIPS (their MAIN loss term). Replaces
        # the FM substitute once the weight file is downloaded. Built lazily in
        # __init__ (see the discriminator block below) — frozen metric net.
        self.x0_llpips_enable = bool(_x0.llpips_enable) if _x0 is not None and 'llpips_enable' in _x0 else False
        self.x0_llpips_ckpt = str(_x0.llpips_ckpt) if _x0 is not None and 'llpips_ckpt' in _x0 \
            else './weights/vgg16_sdturbo_lpips.pth'
        self.llpips_loss = None
        if self.x0_enable:
            if self.x0_mode == 'invsr':
                print('[v18.3] InvSR x0 SUPERVISION ACTIVE: latent-space MSE+FM+adv '
                      f'on the single-step x0, t<={self.x0_t_max}, '
                      f'w=ldif:{self.x0_w_ldif}/lfm:{self.x0_w_lfm}/ldis:{self.x0_w_ldis}')
            else:
                need_fm = float(self.opts.losses.feature_matching.weight) > 1e-5 \
                    or float(self.opts.losses.adversarial.weight) > 1e-5
                assert need_fm, 'losses.x0.enable=True needs feature_matching (or adversarial) weight > 0'
                print('[v17] GATE-FREE x0 SUPERVISION ACTIVE: re-noised symmetric pairs '
                      '(fake_t vs real_t at the same noise level); analytic ab_t '
                      'weighting; NO timestep gate; D real = photos on BOTH batches.')
        if self.preserve_weight > 0:
            print(f'[v15] SCORE-PRESERVATION ACTIVE: weight={self.preserve_weight} | '
                  f'teacher = frozen UNet, zero control injection, '
                  f'W+ tokens in teacher = {self.teacher_wplus} | '
                  f'band = HIGH (offset minus low-pass, sigma={self.synth_lp_sigma} latent px)')
        if self.use_gan_fm:
            from omegaconf import OmegaConf as _OC
            disc_cfg = _OC.to_container(self.opts.losses.discriminator)
            disc_cfg.pop('kind', None)
            self.discriminator = make_discriminator(
                self.opts.losses.discriminator.kind, **disc_cfg).to(self.device)
            self.discriminator.train()
            adv_cfg = _OC.to_container(self.opts.losses.adversarial)
            adv_cfg.pop('kind', None)
            self.adversarial_loss = make_discrim_loss(
                self.opts.losses.adversarial.kind, **adv_cfg)
            self.optimizer_d = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=float(self.opts.losses.optimizers.discriminator.lr))
            self._disc_queue = []   # detached (fake, real) pairs queued at G time
            n_d = sum(p.numel() for p in self.discriminator.parameters())
            print(f'[v14] GAN/FM ACTIVE: NLayer D params={n_d:,} | '
                  f'fm_weight={self.fm_weight} (orig 100) | '
                  f'g_adv_weight={self.adv_weight} (orig 10; 0=FM-first) | '
                  f'r1 gp_coef={self.opts.losses.adversarial.gp_coef}')
        else:
            self.discriminator = None

        # v18.3 InvSR mode: latent UNet discriminator (timestep-conditioned,
        # projected-input, multi-scale heads — unet_discriminator.py, a
        # faithful port). Optimizer per InvSR config: Adam lr_dis=5e-5,
        # weight_decay_dis=1e-3 (sd-turbo-sr-ldis.yaml train section).
        if self.x0_enable and self.x0_mode == 'invsr':
            from models.saicinpainting.training.modules.unet_discriminator import \
                UNetLatentDiscriminator
            self.discriminator = UNetLatentDiscriminator(
                in_channels=4, cross_attention_dim=768).to(self.device)
            self.discriminator.train()
            self.optimizer_d = torch.optim.Adam(
                self.discriminator.parameters(), lr=5e-5, weight_decay=1e-3)
            self._disc_queue_invsr = []
            n_d = sum(p.numel() for p in self.discriminator.parameters())
            print(f'[v18.3 InvSR] latent UNet-D {n_d:,} params | t_max={self.x0_t_max} | '
                  f'w ldif:lfm:ldis = {self.x0_w_ldif}:{self.x0_w_lfm}:{self.x0_w_ldis} | '
                  f'latent clamp ±{self.x0_latent_bound} | dis_warmup={self.x0_dis_warmup} '
                  f'(their 10K/100K iters scaled to our 30K budget)')
        # v18.4: finetuned latent-LPIPS (InvSR llpips slot, their MAIN term).
        # Instantiated per their config: net=vgg16, latent=True, in_chans=4,
        # lpips=True, pnet_tune=True, use_dropout=True, eval, FROZEN.
        if self._invsr() and self.x0_llpips_enable:
            from models.invsr_latent_lpips.lpips import LPIPS as _LatentLPIPS
            self.llpips_loss = _LatentLPIPS(
                pretrained=False, net='vgg16', lpips=True, spatial=False,
                pnet_rand=False, pnet_tune=True, use_dropout=True,
                eval_mode=True, latent=True, in_chans=4, verbose=False,
                model_path=os.path.abspath(self.x0_llpips_ckpt)).to(self.device)
            self.llpips_loss.eval()
            set_requires_grad(self.llpips_loss, False)
            print(f'[v18.4] InvSR finetuned latent-LPIPS ACTIVE '
                  f'(w={self.x0_w_lfm}) <- {self.x0_llpips_ckpt}')

        # ---------------- warper ----------------
        self.warper = Warper()          # orig forward_warp (splatting), untouched
        self.warper_ext = WarperExt()   # + inverse_warp (clean visible anchor)

        # ---------------- optimizer (spec §5.2 routing) ----------------
        self._route_trainable_parameters()
        self.optimizer = self._build_optimizer()

        # ---------------- data ----------------
        self.dataset_json = 'dataset.json'
        self.configure_datasets()
        self.configure_dataloaders()

        # ---------------- logging / checkpoints ----------------
        log_dir = os.path.join(self.opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = os.path.join(self.opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_train_dir = os.path.join(log_dir, 'images/train')
        self.save_val_dir = os.path.join(log_dir, 'images/val')
        os.makedirs(self.save_train_dir, exist_ok=True)
        os.makedirs(self.save_val_dir, exist_ok=True)

        if self.opts.checkpoint_path is not None:
            self.resume()


    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------

    def _install_attention_processors(self):
        """W+ tokens -> attn2 (independent branch, fixed tanh(3) gate);
        RefNet features -> attn1 (K/V/O adapters, gate 0->0.5, warm-start)."""
        attn_procs = {}
        for name in self.denoising_unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith('attn1.processor')
                else self.denoising_unet.config.cross_attention_dim
            )
            if name.startswith('mid_block'):
                hidden_size = self.denoising_unet.config.block_out_channels[-1]
            elif name.startswith('up_blocks'):
                block_id = int(name[len('up_blocks.')])
                hidden_size = list(reversed(self.denoising_unet.config.block_out_channels))[block_id]
            elif name.startswith('down_blocks'):
                block_id = int(name[len('down_blocks.')])
                hidden_size = self.denoising_unet.config.block_out_channels[block_id]
            else:
                hidden_size = self.denoising_unet.config.block_out_channels[0]

            proc = ReferenceAttentionProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                local_window=self.local_window_k,
            ).to(self.device)
            attn_module = self.denoising_unet.get_submodule(name.removesuffix('.processor'))
            proc.initialize_from_attention(attn_module)  # K/V/O warm-start from SD attn
            if cross_attention_dim is not None:
                # W+ gate: fixed tanh(3) ~= 0.995 constant injection (spec §4.2)
                proc.wplus_scale.data.fill_(float(self.opts.wplus.gate_value))
                proc.wplus_scale.requires_grad_(False)
            else:
                # RefNet gate: 0 -> 0.5 init (先验 16: zero-init gates never grow)
                proc.reference_scale.data.fill_(math.atanh(self.ref_gate_init))
                # v11 local texture gate: warm-start (same 先验 16). Conservative
                # 0.25 — new pathway, let training raise it if the real-photo
                # texture band actually helps the loss.
                proc.reference_local_scale.data.fill_(
                    math.atanh(self.local_gate_init))
            attn_procs[name] = proc
        self.denoising_unet.set_attn_processor(attn_procs)

    def _route_trainable_parameters(self):
        """spec §5.2: BrushNet full / RefNet adapters / W+ ladder."""
        for name, proc in self.denoising_unet.attn_processors.items():
            if not isinstance(proc, ReferenceAttentionProcessor):
                continue
            if proc.wplus_scale is not None:  # attn2 -> W+ branch
                set_requires_grad(proc, False)
                if self.wplus_mode == 's2_qkv':
                    # unlock Q/K/V only (O frozen); grads cleared on real updates
                    for proj in (proc.to_q_wplus, proc.to_k_wplus, proc.to_v_wplus):
                        for p in proj.parameters():
                            p.requires_grad_(True)
                            self.wplus_qkv_params.append(p)
            else:  # attn1 -> bank-retrieval branch (v8: features from MAIN UNet)
                set_requires_grad(proc, False)
                for proj in (proc.to_k_reference, proc.to_v_reference, proc.to_out_reference):
                    for p in proj.parameters():
                        p.requires_grad_(True)
                        self.refnet_adapter_params.append(p)
                proc.reference_scale.requires_grad_(True)
                self.refnet_adapter_params.append(proc.reference_scale)
                # v11: local texture gate joins the same group (adapters lr)
                if self.local_window_k:
                    proc.reference_local_scale.requires_grad_(True)
                    self.refnet_adapter_params.append(proc.reference_local_scale)
                # v7a/v8: mirror gets its OWN K/V (source-separating channel,
                # lr = brushnet's — the v2 lesson). Softmax competition over the
                # banked MAIN-UNET features; shared O stays unchanged.
                if self.use_mirror:
                    for proj in (proc.to_k_reference_mirror, proc.to_v_reference_mirror):
                        for p in proj.parameters():
                            p.requires_grad_(True)
                            self.mirror_kv_params.append(p)
                    self.mirror_attention_procs.append(proc)

        n_ref = sum(p.numel() for p in self.refnet_adapter_params)
        n_qkv = sum(p.numel() for p in self.wplus_qkv_params)
        n_map = sum(p.numel() for p in self.wplus_mapper_params)
        print(f"[params] bank-retrieval adapters trainable: {n_ref:,} | "
              f"W+ QKV (s2) trainable: {n_qkv:,} | W+ mapper trainable: {n_map:,} | "
              f"mirror: own K/V {sum(p.numel() for p in self.mirror_kv_params):,} "
              f"+ softmax competition ({len(self.mirror_attention_procs)} gauged layers) | "
              f"RefNet DELETED (860M saved, bank=main-UNet self-read)")


    def _build_optimizer(self):
        groups = [{
            'params': [p for p in self.brushnet.parameters() if p.requires_grad],
            'lr': float(self.opts.brushnet.lr), 'name': 'brushnet',
        }]
        if self.refnet_adapter_params:
            groups.append({'params': self.refnet_adapter_params,
                           'lr': float(self.opts.reference.lr), 'name': 'refnet_adapters'})
        if self.wplus_mapper_params:
            groups.append({'params': self.wplus_mapper_params,
                           'lr': float(self.opts.brushnet.lr), 'name': 'wplus_mapper'})
        if self.wplus_qkv_params:
            groups.append({'params': self.wplus_qkv_params,
                           'lr': float(self.opts.brushnet.lr), 'name': 'wplus_qkv'})
        if self.mirror_kv_params:
            groups.append({'params': self.mirror_kv_params,
                           'lr': float(self.opts.brushnet.lr), 'name': 'mirror_kv'})
        if bool(self.opts.brushnet.use_8bit_adam):
            import bitsandbytes as bnb
            print('[optim] 8-bit AdamW (bitsandbytes)')
            return bnb.optim.AdamW8bit(groups)
        print('[optim] 32-bit AdamW')
        return torch.optim.AdamW(groups)

    def configure_datasets(self):
        paths = self.opts.paths.dataset
        self.train_dataset = ImageFolderDataset(
            path=paths.train, resolution=None, use_labels=True,
            load_conf_map=False, datast_json=self.dataset_json)
        self.test_dataset = ImageFolderDataset(
            path=paths.test, resolution=None, use_labels=True,
            load_conf_map=False, max_size=1000)
        if self.use_synth:
            self.train_synth_dataset = SynthImageFolderDataset(
                path_synth=paths.synth, total_imgs_num=100000, resolution=None)
            print(f"Synth dataset: {paths.synth} -> {len(self.train_synth_dataset)} samples")
        print(f"Real train dataset: {paths.train} -> {len(self.train_dataset)} samples")
        print(f"Test dataset: {paths.test} -> {len(self.test_dataset)} samples")

    def configure_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=int(self.opts.data.batch_size), shuffle=True,
            num_workers=int(self.opts.data.workers), drop_last=True)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=int(self.opts.data.test_batch_size), shuffle=False,
            num_workers=int(self.opts.data.test_workers), drop_last=True)
        if self.use_synth:
            self.train_synth_dataloader = DataLoader(
                self.train_synth_dataset, batch_size=int(self.opts.data.synth.batch_size),
                shuffle=True, num_workers=int(self.opts.data.workers), drop_last=True)

    def _print_effective_config(self):
        """事故录 #10/#11: 配置写了 != 生效。逐项打印 + 路径存在性检查。"""
        o = self.opts
        print('=' * 78)
        print('[EFFECTIVE CONFIG — verify line by line before trusting any run]')
        print(f"  exp_dir               : {o.exp_dir}")
        print(f"  max_steps             : {o.max_steps}")
        print(f"  device                : {o.device}")
        print(f"  synth/real 1:1        : use_synth={self.use_synth}")
        print(f"  batch_size (real/synth): {o.data.batch_size} / {o.data.synth.batch_size}")
        print(f"  warp: 512={bool(o.warp.warp_ori_train)} hybrid={bool(o.warp.hybrid)} "
              f"warp_pred={bool(o.warp.warp_pred)} mask=raw_splatting (NO erode/blur)")
        print(f"  warp cond soften     : erode={self.cond_erode_kernel}px blur_k={self.cond_blur_kernel} "
              f"(v10: CONDITION-side only, supervision keeps raw mask; orig port)")
        print(f"  wplus: mode={self.wplus_mode} gate=fixed tanh({o.wplus.gate_value}) "
              f"stage_c_init={bool(o.wplus.init_from_stage_c)}")
        print(f"  reference: use={bool(o.reference.use)} use_mirror={self.use_mirror} "
              f"gate_init={self.ref_gate_init} lr={o.reference.lr}")
        print(f"  reference real_primary : {self.real_primary_mirror and 'mirror' or 'render'} "
              f"(user plan: real photo = PRIMARY bank source, render -> extra)")
        print(f"  reference LOCAL       : window={self.local_window_k} "
              f"gate_init={self.local_gate_init} "
              f"(v11 texture-band retrieval from PRIMARY (real-photo) source; "
              f"activates the formerly-dead reference_local_scale)")
        print(f"  real pass1 dual-band  : visible/mirror FULL-band x{float(self.opts.losses.real_pass1.visible_weight)} "
              f"+ blind LOW-band x{self.blind_struct_weight} "
              f"(v12: target=FULL SHARP render, only low-freq error penalized; "
              f"hole_weight={float(self.opts.losses.real_pass1.hole_weight)} ignored "
              f"when dual-band active; 0=off -> single-band v11 path)")
        print(f"  loss synth_texture     : {self.synth_texture} "
              f"(sigma={self.synth_lp_sigma} latent px; user plan: structure-only "
              f"supervision, texture band = 0)")
        print(f"  brushnet: init={self.brushnet_init} lr={o.brushnet.lr} 8bit={bool(o.brushnet.use_8bit_adam)} "
              f"grad_ckpt={bool(o.brushnet.gradient_checkpointing)}")
        print(f"  loss eps_weight       : {o.losses.eps_weight}")
        print(f"  real pass1 FORWARD    : {self.real_pass1_forward} "
              "(False = STRUCT arm: real batch trains pass2 only; inference unchanged)")
        print(f"  real pass1 t-domain   : "
              f"{'FULL [0,1000)' if not self.real_pass1_t_max else f'[0, {self.real_pass1_t_max}]'}"
              f" | infer.t_start={self.infer_t_start} "
              f"(0=pure-noise chain; >0=SDEdit from the noised anchor)")
        if self._invsr():
            print(f"  x0 InvSR mode         : t_max={self.x0_t_max} "
                  f"w=ldif:{self.x0_w_ldif}/lfm:{self.x0_w_lfm}/ldis:{self.x0_w_ldis} "
                  f"warmup={self.x0_dis_warmup} clamp=±{self.x0_latent_bound} (LATENT space)")
        print(f"  loss real_p1 anchor   : visible={o.losses.real_pass1.visible_weight} "
              f"(target=inv_warp) hole={o.losses.real_pass1.hole_weight} (target=y_hat_novel) "
              f"mirror_anchor={self.mirror_anchor} (hole real-photo coverage via inv_warp(x_mirror))")
        print(f"  loss pixel (t<{self.pixel_max_t})   : L1*{o.losses.pixel.l1_weight} "
              f"PL*{o.losses.pixel.resnet_pl_weight} ID*{o.losses.pixel.id_weight} "
              f"x0_clip={self.x0_clip} res={self.pixel_res}")
        print(f"  loss latent closure   : *{o.losses.latent.weight} (orig L1195-1200)")
        print(f"  loss lpips/gan/fm     : {o.losses.lpips.weight} / {o.losses.adversarial.weight} "
              f"/ {o.losses.feature_matching.weight}  (orig GAN/FM 10/100 -> default 0)")
        print(f"  timestep range        : [0, 1000) FULL RANGE (no low-t clamp; v3.1 lesson)")
        print(f"  smoke: check={self.smoke_check} fix_timestep={self.fix_timestep}")
        for label, p in [('sd', o.paths.sd), ('brushnet', o.paths.brushnet),
                         # v18: check wplus_stage_c ONLY when it is actually
                         # loaded — s1_mapper/s2_qkv with init_from_stage_c=False
                         # train the mapper from scratch (teacher ladder rung 1)
                         # and must not depend on the remote-era file.
                         ('wplus_stage_c', o.paths.wplus_stage_c
                          if (self.use_wplus and (self.wplus_mode == 'frozen'
                                                  or bool(o.wplus.init_from_stage_c)))
                          else None),
                         ('train', o.paths.dataset.train), ('test', o.paths.dataset.test),
                         ('synth', o.paths.dataset.synth),
                         ('LaMa_PL', o.losses.pixel.weights_path),
                         ('gan_encoder', o.gan.checkpoint_path)]:
            if p is None:
                continue
            exists = os.path.exists(str(p))
            flag = 'OK ' if exists else 'MISSING!'
            print(f"  path [{flag}] {label:14s}: {p}")
            if not exists:
                raise FileNotFoundError(f"Required path missing: {label}={p}")
        print('=' * 78)


    # ------------------------------------------------------------------
    # geometry / batch parsing
    # ------------------------------------------------------------------

    @staticmethod
    def flip_yaw(pose_matrix):
        flipped = pose_matrix.clone()
        flipped[:, 0, 1] *= -1
        flipped[:, 0, 2] *= -1
        flipped[:, 1, 0] *= -1
        flipped[:, 2, 0] *= -1
        flipped[:, 0, 3] *= -1
        return flipped

    def get_mirror_c(self, c):
        pose, intrinsics = c[:, :16].reshape(-1, 4, 4), c[:, 16:].reshape(-1, 3, 3)
        flipped_pose = self.flip_yaw(pose)
        c_mirror = torch.cat(
            [flipped_pose.reshape(-1, 4 * 4), intrinsics.reshape(-1, 3 * 3)], dim=1
        ).reshape(-1, 25)
        return c_mirror

    def _parse_real_batch(self, t):
        """Real dataset tuple (16 items, orig coach L330-332 ordering)."""
        assert len(t) >= 15, f'unexpected real batch length {len(t)}'
        return {
            'x_256': t[0].to(self.device).float(), 'x': t[1].to(self.device).float(),
            'c': t[2].to(self.device).float(), 'codes': t[3].to(self.device).float(),
            'y_hat': t[4].to(self.device).float(), 'depth': t[5].to(self.device).float(),
            'x_mirror': t[7].to(self.device).float(),
            'c_mirror': t[8].to(self.device).float(),
            'depth_mirror': t[10].to(self.device).float(),
            'c_novel': t[12].to(self.device).float(),
            'y_hat_novel': t[13].to(self.device).float(),
            'depth_novel': t[14].to(self.device).float(),
        }

    def _parse_synth_batch(self, t):
        """Synth dataset tuple (12 items, orig coach L498-499 ordering)."""
        assert len(t) >= 11, f'unexpected synth batch length {len(t)}'
        batch = {
            'x_256': t[0].to(self.device).float(), 'x': t[1].to(self.device).float(),
            'c': t[2].to(self.device).float(), 'codes': t[3].to(self.device).float(),
            'y_hat': t[4].to(self.device).float(), 'depth': t[5].to(self.device).float(),
            'target_256': t[6].to(self.device).float(),
            'target': t[7].to(self.device).float(),
            'c_novel': t[8].to(self.device).float(),
            'y_hat_novel': t[9].to(self.device).float(),   # target_hat (inversion render)
            'depth_novel': t[10].to(self.device).float(),
        }
        if self.use_mirror:
            batch['x_mirror'] = torch.flip(batch['x'], dims=[3])
            batch['c_mirror'] = self.get_mirror_c(batch['c'])
            batch['depth_mirror'] = torch.flip(batch['depth'], dims=[3])
        return batch

    # ------------------------------------------------------------------
    # encoding / feature helpers
    # ------------------------------------------------------------------

    def _encode_image(self, img):
        """[0,1] RGB -> VAE latents (stochastic posterior, BrushNet recipe)."""
        return self.vae.encode(img * 2.0 - 1.0).latent_dist.sample() * 0.18215

    @torch.no_grad()
    def _write_bank(self, img):
        """AnimateDiff WRITE pass on the MAIN UNet (v8): encode the clean
        reference image, run one no-grad forward at t=0, and bank each attn1
        layer's INPUT (norm_hidden_states — the exact quantity mutual_self_
        attention.py L224 banks), reshaped to (B,C,H,W). Deduplicated per
        (channels, resolution) so the existing nearest-resolution retrieval
        in the processor picks a unique representative."""
        banks = {}

        def hook(module, inputs):
            hs = inputs[0]                       # (b, l, c), post group-norm
            b, l, cdim = hs.shape
            side = int(l ** 0.5)
            fmap = hs.transpose(1, 2).reshape(b, cdim, side, side)
            bucket = banks.setdefault(cdim, [])
            if not any(t.shape[-2:] == fmap.shape[-2:] for t in bucket):
                bucket.append(fmap)

        for name in self.denoising_unet.attn_processors:
            if not name.endswith('attn1.processor'):
                continue
            module = self.denoising_unet.get_submodule(name.removesuffix('.processor'))
            self._bank_hooks.append(module.register_forward_pre_hook(hook))
        try:
            latents = self.vae.encode(img * 2.0 - 1.0).latent_dist.mode() * 0.18215
            t0 = torch.zeros(img.shape[0], device=self.device, dtype=torch.long)
            prompt = self.empty_prompt_embeds.expand(img.shape[0], -1, -1)
            self.denoising_unet(sample=latents, timestep=t0, encoder_hidden_states=prompt)
        finally:
            for h in self._bank_hooks:
                h.remove()
            self._bank_hooks = []
        return banks

    def _extract_reference_features(self, images, source_tags):
        """v8 bank extraction: each clean reference image runs ONCE through the
        MAIN frozen UNet (no_grad, t=0); every attn1 layer banks its input
        (norm_hidden_states). Banked per-block features live in the SAME
        {channels: [B,C,H,W]} dict format the attention processor already
        consumes, so the retrieval side is unchanged. Interface-compatible
        replacement for the deleted 860M RefNet (gate-checked, §8.21)."""
        primary, extra = None, None
        for idx, (img, tag) in enumerate(zip(images, source_tags)):
            with torch.no_grad():
                feat = self._write_bank(img)
            if self.smoke_check and self.global_step <= 2:
                print(f"[bank input check] source={tag} shape={tuple(img.shape)} "
                      f"mean={img.mean().item():.4f} std={img.std().item():.4f} "
                      f"finite={bool(torch.isfinite(img).all())}")
            if idx == 0:
                primary = feat
            else:
                extra = feat
        return primary, extra

    def _wplus_tokens(self, codes, train_this_update):
        """W+ [B,14,512] -> mapper -> [B,18,768] tokens (spec §4.2).
        Frozen档 / real-batch updates: no grad (§17.8 red line: the W+ branch
        only ever trains on SYNTH updates). mode=off: no tokens at all."""
        if not self.use_wplus:
            return None
        grad_on = (self.wplus_mode != 'frozen') and train_this_update
        ctx = torch.enable_grad() if grad_on else torch.no_grad()
        with ctx:
            return self.w_mapper(codes)


    # ------------------------------------------------------------------
    # diffusion forward (teacher ③: input noised, target = the added noise)
    # ------------------------------------------------------------------

    def _diffusion_forward(self, target_img, condition_img, mask, codes,
                           ref_images, ref_tags, t_override=None, train_wplus=False,
                           mask_cond=None, t_range=None, hf_img=None):
        """One BrushNet+UNet epsilon forward. Returns everything the loss needs.
        v10: mask_cond (softened) drives the BrushNet condition channel; the
        returned mask_latents stays RAW for supervision weights.
        v18.8: hf_img (B,3,H,W in [0,1], pixel space) is the AnyDoor-style
        high-frequency texture map. When hf_enable, it is downsampled to
        latent resolution and concatenated into BrushNet conditioning
        (5ch → 8ch). Pass None for legacy 5ch behavior."""
        with torch.no_grad():
            target_latents = self._encode_image(target_img)
            cond_latents = self._encode_image(condition_img)
            m_brush = mask if mask_cond is None else mask_cond
            mask_latents = F.interpolate(m_brush, size=target_latents.shape[-2:], mode='nearest')
            if self.hf_enable and hf_img is not None:
                # HF in pixel space → downsample to latent resolution
                hf_lat = F.interpolate(hf_img, size=target_latents.shape[-2:],
                                       mode='bilinear', align_corners=False)
                conditioning_latents = torch.cat(
                    [cond_latents, mask_latents, hf_lat], dim=1)  # 4+1+3 = 8ch
            else:
                conditioning_latents = torch.cat([cond_latents, mask_latents], dim=1)  # 4+1 = 5ch
        mask_latents = F.interpolate(mask, size=target_latents.shape[-2:], mode='nearest')

        bsz = target_latents.shape[0]
        noise = torch.randn_like(target_latents)
        if t_override is not None:
            timesteps = torch.full((bsz,), int(t_override), device=self.device, dtype=torch.long)
        else:  # FULL range [0,1000) — low-t-only training drifts to haze (v3.1).
            # v18.6: t_range optionally restricts the domain (pass1 now trains
            # ONLY in the deployed SDEdit domain t∈[0,real_pass1.t_max] — the
            # InvSR pattern: their training timesteps [100..250] = their
            # inference start domain; high-t pass1 training taught prior-
            # dominated generation that inference never uses).
            _hi = int(t_range) if t_range else \
                int(self.noise_scheduler.config.num_train_timesteps)
            timesteps = torch.randint(0, _hi, (bsz,), device=self.device).long()
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)

        wplus_features = self._wplus_tokens(codes, train_wplus)
        ref_feat, ref_feat_extra = self._extract_reference_features(ref_images, ref_tags)

        prompt_embeds = self.empty_prompt_embeds.expand(bsz, -1, -1)

        # v15 teacher: the SAME frozen UNet with ZERO control injection on the
        # SAME noisy anchor, run BEFORE the student graph is built (lowest
        # memory window; output is a small no-grad tensor). The anchor's
        # geometry/color ride in x_t, so this is NOT an unconditional prior —
        # it is "the photo-prior denoising instinct looking at our data".
        # Structurally clean domain (v14 lesson): no BrushNet add-samples, no
        # bank retrieval — those are exactly the control being trained.
        eps_teacher = None
        if self.preserve_weight > 0:
            with torch.no_grad():
                _cak = {}
                if wplus_features is not None and self.teacher_wplus:
                    _cak['wplus_features'] = wplus_features
                eps_teacher = self.denoising_unet(
                    sample=noisy_latents, timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    down_block_add_samples=None,
                    mid_block_add_sample=None,
                    up_block_add_samples=None,
                    cross_attention_kwargs=_cak or None,
                    return_dict=False)[0]

        down_res, mid_res, up_res = self.brushnet(
            noisy_latents, timesteps, encoder_hidden_states=prompt_embeds,
            brushnet_cond=conditioning_latents, return_dict=False)

        cross_attention_kwargs = {}
        if wplus_features is not None:
            cross_attention_kwargs['wplus_features'] = wplus_features
        if ref_feat is not None:
            cross_attention_kwargs['reference_features'] = ref_feat
            if ref_feat_extra is not None:
                cross_attention_kwargs['reference_features_extra'] = ref_feat_extra

        eps_pred = self.denoising_unet(
            sample=noisy_latents, timestep=timesteps, encoder_hidden_states=prompt_embeds,
            down_block_add_samples=[s for s in down_res],
            mid_block_add_sample=mid_res,
            up_block_add_samples=[s for s in up_res],
            cross_attention_kwargs=cross_attention_kwargs or None,
            return_dict=False)[0]

        return {
            'eps_pred': eps_pred, 'noise': noise, 'timesteps': timesteps,
            'target_latents': target_latents, 'noisy_latents': noisy_latents,
            'mask_latents': mask_latents, 'eps_teacher': eps_teacher,
        }


    def _eps_mse(self, eps_pred, noise, weight_map=None):
        """epsilon-MSE. weight_map=None -> full frame (orig with_mask=False)."""
        sq = (eps_pred.float() - noise.float()).pow(2)
        if weight_map is None:
            return sq.mean()
        w = weight_map.float()
        return (sq * w).sum() / (w.sum() * sq.shape[1] + 1e-6)

    def _texture_zero_weight(self, shape, device):
        """User's plan (2026-08-26): structure-only synth supervision.

        A Gaussian low-pass WEIGHT FIELD (separable, latent resolution): each
        latent pixel's error is averaged with its neighborhood at the
        'structure scale' (sigma = synth_lowpass_sigma latent px). High-freq
        (texture) error components cancel in this local average -> the texture
        band receives ZERO gradient; structure (contours/organs, ~1-2 latent
        px scale) is preserved at full strength. Implemented as an explicitly
        normalized conv on the SQUARED-ERROR MAP, i.e. eps-MSE evaluated on
        the low-passed error field — mathematically equivalent to filtering
        the loss, without ever touching the target latents."""
        c, h, w = shape[1], shape[2], shape[3]
        s = self.synth_lp_sigma
        k = max(3, int(2 * round(2 * s) + 1))
        r = k // 2
        x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
        g1d = torch.exp(-0.5 * (x / s) ** 2)
        g1d = g1d / g1d.sum()
        return g1d.view(1, 1, 1, k), g1d.view(1, 1, k, 1), (h, w)

    def _eps_mse_structure_only(self, eps_pred, noise):
        """Structure-band eps-MSE (texture band weight = 0). Math: the eps loss
        is the energy of the SIGNED error e = eps_pred - noise; frequency
        decomposition ||e||^2 = sum_f |E(f)|^2. Low-passing e FIRST and then
        squaring keeps only the structure band: loss = mean((G_sigma * e)^2).
        (Squaring first would destroy signs and fail to cancel texture.)
        Checkerboard error at sigma=2 latent px -> ~2e-4 of full loss."""
        e = eps_pred.float() - noise.float()                    # (b,4,64,64)
        return self._lowpass_eps(e).pow(2).mean()

    def _lowpass_eps(self, e):
        """Separable Gaussian low-pass ON THE ERROR FIELD (never on images or
        targets — v3.2 lesson: no lowpassed IMAGE may ever enter the pipeline;
        this kernel only selects WHICH FREQUENCY BAND of the error is
        penalized). replicate padding keeps border supervision intact."""
        kx, ky, _ = self._texture_zero_weight(e.shape, e.device)
        r = (ky.shape[-2] // 2, kx.shape[-1] // 2)
        ch = e.shape[1]
        e = F.pad(e, (r[1], r[1], r[0], r[0]), mode='replicate')
        return F.conv2d(F.conv2d(e, ky.repeat(ch, 1, 1, 1), groups=ch),
                        kx.repeat(ch, 1, 1, 1), groups=ch)

    def _eps_mse_dual_band(self, eps_pred, noise, w_full, w_low):
        """v12 evidence-graded supervision (user's three-partition plan).

        Per latent position the loss follows the EVIDENCE CONTINUOUS FIELD:
          photo-evidence positions (vis_eff + mirror, continuous in [0,1])
            -> FULL-band eps-MSE   (target = clean photo re-projection)
          blind positions (1 - evidence, continuous)
            -> LOW-band eps-MSE ONLY (target = FULL, SHARP EG3D render anchor;
               only the low-frequency component of the error is penalized, so
               geometry/contour is anchored while the EG3D texture band gets
               ZERO gradient — no oil-paint teaching, no lowpassed target)
        Both fields are continuous (v6 lesson: no binarized boundaries).
        Normalization mirrors _eps_mse: sum(w*sq) / (sum(w) * ch).
        """
        e = eps_pred.float() - noise.float()
        sq_full = e.pow(2)
        sq_low = self._lowpass_eps(e).pow(2)
        w_full = w_full.float()
        w_low = w_low.float()
        num = (sq_full * w_full + sq_low * w_low).sum()
        den = (w_full + w_low).sum() * e.shape[1] + 1e-6
        return num / den

    def _predict_x0(self, eps_pred, noisy_latents, timesteps):
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        sqrt_ap = alphas_cumprod[timesteps] ** 0.5
        sqrt_1map = (1.0 - alphas_cumprod[timesteps]) ** 0.5
        return (noisy_latents - sqrt_1map.view(-1, 1, 1, 1) * eps_pred) / sqrt_ap.view(-1, 1, 1, 1)

    def _decode_x0_to_pixel(self, x0_latents, output_size=256):
        """Resize in LATENT space, then decode (4x decoder-memory saving)."""
        latent_size = output_size // 8
        if x0_latents.shape[-2:] != (latent_size, latent_size):
            x0_latents = F.interpolate(
                x0_latents, size=(latent_size, latent_size),
                mode='bilinear', align_corners=False)
        return (self.vae.decode(x0_latents / 0.18215).sample / 2 + 0.5).clamp(0, 1)

    def _low_t_losses(self, out, target_img, codes, allow_pixel=True):
        """t<max_t self-gated x0 single-step decode (FFC 1-step translation):
        pixel L1*10 + ResNetPL*30 + ID*0.5 vs target + latent closure *0.1.
        v16: allow_pixel=False on the SYNTH pass keeps the paint-oil red line
        (synth targets are EG3D renders; the photo-ref backflow is REAL-only)."""
        metrics = {'pixel_l1': 0.0, 'pixel_pl': 0.0, 'pixel_id': 0.0,
                   'latent': 0.0, 'low_t_hits': 0.0}
        w = self.opts.losses.pixel
        _any_low = ((allow_pixel and float(w.l1_weight) > 1e-5)
                    or float(w.resnet_pl_weight) > 1e-5
                    or float(w.id_weight) > 1e-5
                    or float(self.opts.losses.latent.weight) > 1e-5)
        sel = out['timesteps'] < self.pixel_max_t
        metrics['low_t_hits'] = float(sel.float().mean())
        if not bool(sel.any()) or not _any_low:
            zero = out['eps_pred'].new_zeros(())
            return zero, metrics

        x0 = self._predict_x0(out['eps_pred'], out['noisy_latents'], out['timesteps'])
        x0_sel = x0[sel].clamp(-self.x0_clip, self.x0_clip)  # keep the graph
        # v16: random latent-patch decode (memory precedent, PROJECT_HISTORY
        # "32x32 latent patch (+4px ctx)"): ONE shared random crop per step —
        # 256px-class texture resolution at ~1/4 the decoder activations
        # (full-frame decode peaked 22.4GiB > 22GiB budget). Pred and target
        # crop the SAME place so the L1 stays strictly paired.
        ph = self.pixel_patch_lat
        L = int(x0_sel.shape[-1])
        patch_info = 'full'
        if ph > 0 and L > ph:
            ctx = 4
            cs = min(ph + 2 * ctx, L)
            top = int(torch.randint(0, L - cs + 1, (1,)).item())
            left = int(torch.randint(0, L - cs + 1, (1,)).item())
            crop = x0_sel[:, :, top:top + cs, left:left + cs]
            pred_img = (self.vae.decode(crop / 0.18215).sample / 2 + 0.5).clamp(0, 1)
            H = int(target_img.shape[-1])
            sc = H / float(L)
            tgt = target_img[sel][:, :, int(top * sc):int((top + cs) * sc),
                                  int(left * sc):int((left + cs) * sc)]
            target_sel = F.interpolate(tgt, size=pred_img.shape[-2:],
                                       mode='bilinear', align_corners=False)
            patch_info = f'patch(lat{top}:{top + cs},{left}:{left + cs})->px{pred_img.shape[-1]}'
        else:
            pred_img = self._decode_x0_to_pixel(x0_sel, self.pixel_res)
            target_sel = F.interpolate(
                target_img[sel], size=(self.pixel_res, self.pixel_res),
                mode='bilinear', align_corners=False)

        total = pred_img.new_zeros(())
        _pix_on = allow_pixel and float(w.l1_weight) > 1e-5
        if _pix_on:
            l1 = F.l1_loss(pred_img, target_sel)
            total = total + l1 * float(w.l1_weight)
            metrics['pixel_l1'] = float(l1)
            if self.smoke_check and not getattr(self, '_v16_fired', False):
                self._v16_fired = True
                t_now = int(out['timesteps'][sel][0]) if bool(sel.any()) else -1
                print(f'[v16 PHOTO-REF L1 FIRED, real pass2] l1={float(l1):.5f} '
                      f'x{float(w.l1_weight):.1f} -> contrib={float(l1) * float(w.l1_weight):.5f} '
                      f'| t={t_now} (<{self.pixel_max_t} gate) | {patch_info} '
                      f'| x0 range [{float(x0_sel.min()):.2f}, {float(x0_sel.max()):.2f}] '
                      f'(clip {self.x0_clip}) | target = REAL PHOTO x '
                      f'(orig pillar-2 pixel-domain backflow, coach L1269)')
        if float(w.resnet_pl_weight) > 1e-5:
            pl = self.loss_resnet_pl(pred_img, target_sel)
            total = total + pl  # weight applied inside ResNetPL (orig behavior)
            metrics['pixel_pl'] = float(pl)
        if float(w.id_weight) > 1e-5:
            id_value, _, _ = self.loss_id(pred_img, target_sel, target_sel)
            total = total + id_value * float(w.id_weight)
            metrics['pixel_id'] = float(id_value)
            if self.smoke_check and not getattr(self, '_id_fired_logged', False):
                self._id_fired_logged = True
                print(f'[v13 id loss FIRED, real data] id={float(id_value):.5f} '
                      f'pred{tuple(pred_img.shape)} ArcFace semantic-scale anchor ACTIVE')

        if float(self.opts.losses.latent.weight) > 1e-5:
            pred_256 = F.adaptive_avg_pool2d(pred_img, (256, 256))
            w_pred = self.gan.encoder_forward(pred_256)
            latent_value = F.mse_loss(codes[sel], w_pred)
            total = total + latent_value * float(self.opts.losses.latent.weight)
            metrics['latent'] = float(latent_value)
            if self.smoke_check and not getattr(self, '_latent_fired_logged', False):
                self._latent_fired_logged = True
                print(f'[v13 latent closure FIRED, real data] latent='
                      f'{float(latent_value):.5f} codes{tuple(codes[sel].shape)} '
                      f'w_pred{tuple(w_pred.shape)} — W+ identity anchor ACTIVE '
                      f'(orig pass1-only supervision restored)')

        out['pred_x0_img'] = pred_img.detach()  # for image logging
        out['pred_x0_img_g'] = pred_img         # v14: graph-carrying (G-side FM/adv)
        return total, metrics

    def _x0_losses(self, out, target_img, photo_ref, tag):
        """v17-prime: GATE-FREE x0 distribution supervision.

        First-principles construction (no timestep gate, no hand weights):
          x0_hat = (x_t - sqrt(1-ab_t)*eps_pred) / sqrt(ab_t)          (analytic)
          fake_t = sqrt(ab_t)*x0_hat + sqrt(1-ab_t)*eps_new            (re-noised
                   to the SAME noise level as x_t)
          real_t = x_t when the eps-target is a photo (real pass2), or
                   sqrt(ab_t)*enc(photo_ref) + sqrt(1-ab_t)*eps' (synth — the
                   D's real distribution stays PHOTOS on both batches).
        Properties (all analytic, none hand-set):
          * noise shortcut symmetrized away — D cannot tell fake from real by
            residual noise level (the v14 lesson generalised);
          * supervision decays to zero as t->T on its own (both sides
            degenerate to pure noise) — the t<200 gate replaced by physics;
          * the FM/L1 gradient through fake_t carries the sqrt(ab_t) factor
            (free SNR weighting); the x0-space L1 uses weight ab_t.
        Shared 32x32-latent (256px) decode for L1 and FM (memory budget).
        Returns (loss, metrics); queues the detached pair for the D step.
        """
        metrics = {}
        ab = self.noise_scheduler.alphas_cumprod[out['timesteps']].view(-1, 1, 1, 1)
        x0_hat = (out['noisy_latents'] - (1.0 - ab).sqrt() * out['eps_pred']) / ab.sqrt()
        x0_hat = x0_hat.clamp(-self.x0_clip, self.x0_clip)   # numeric guard only
        eps_new = torch.randn_like(x0_hat)
        fake_lat = ab.sqrt() * x0_hat + (1.0 - ab).sqrt() * eps_new
        real_lat = out['noisy_latents']                       # photo-based eps target
        if photo_ref is not None:
            with torch.no_grad():
                ref_px = F.interpolate(photo_ref, size=(256, 256), mode='bilinear',
                                       align_corners=False)
                ref_lat = self.vae.encode(ref_px * 2.0 - 1.0).latent_dist.mode() * 0.18215
                ref_lat = F.interpolate(ref_lat, size=x0_hat.shape[-2:], mode='bilinear',
                                        align_corners=False)
                real_lat = ab.sqrt() * ref_lat + (1.0 - ab).sqrt() * torch.randn_like(ref_lat)

        # v18: NATIVE-RESOLUTION patch decode (replaces the 64->32 latent
        # downscale). First principles: interpolating latents down halves
        # EVERY texture frequency before the VAE decode AND feeds the decoder
        # an off-manifold latent; the pathology under test (blind-region
        # high-freq content) lives at NATIVE latent spacing. A shared random
        # latent crop (res_latent + 2*ctx, v16 memory precedent) keeps decoder
        # activations at the same budget while supervising native-frequency
        # pixels; both sides crop the SAME location (strictly paired);
        # coverage is uniform in expectation over steps.
        ctx = 4
        L_ = int(x0_hat.shape[-1])
        cs = min(self.x0_res_lat + 2 * ctx, L_)
        top = int(torch.randint(0, L_ - cs + 1, (1,)).item())
        left = int(torch.randint(0, L_ - cs + 1, (1,)).item())

        def dec(lat):
            crop = lat[:, :, top:top + cs, left:left + cs]
            return (self.vae.decode(crop / 0.18215).sample / 2 + 0.5).clamp(0, 1)

        fake_px = dec(fake_lat)
        with torch.no_grad():
            real_px = dec(real_lat)

        # Pixel L1 on the SYMMETRIC pair (one gradient-carrying decode — the
        # v16 patch-decode memory class). Low-t: == clean L1(x0, target);
        # high-t: noise-dominated but the analytic ab_t weight -> 0 suppresses
        # it. No gate, no patch sampling, no hand weight.
        if self.x0_l1_weight > 1e-5:
            l1 = F.l1_loss(fake_px, real_px)
            w = float(ab.mean())
            total = l1 * w * self.x0_l1_weight
            metrics[f'{tag}_x0_l1'] = float(l1)
            metrics[f'{tag}_x0_abw'] = w
        else:
            total = fake_px.new_zeros(())

        out['pred_x0_img'] = fake_px.detach()
        out['pred_x0_img_g'] = fake_px            # graph-carrying for G-side FM
        # v18 SNR alignment: d(fake_lat)/d(eps_pred) = -sqrt(1-ab) — STRONGEST
        # at high t where the pair carries no content signal. Multiplying the
        # whole distribution family by ab makes the net eps_pred-gradient
        # factor ab*sqrt(1-ab) (peaked mid-SNR, ->0 at both ends) — the
        # min-SNR principle applied to auxiliary losses. L1 already carries ab.
        gm = self._gan_fm_g_loss(out, real_px, tag, weight=float(ab.mean()))
        if gm is not None:
            total = total + gm[0]
            metrics.update(gm[1])
        return total, metrics

    def _invsr(self):
        """v18.3 mode predicate: InvSR-calibrated latent distribution supervision.
        (getattr-guarded: _print_effective_config runs BEFORE the x0 flags are
        parsed in __init__ — the print only needs the safe default 'off'.)"""
        return bool(getattr(self, 'x0_enable', False)) \
            and getattr(self, 'x0_mode', '') == 'invsr'

    def _hinge_d_loss_invsr(self, logits_real, logits_fake):
        """InvSR hinge (trainer L1612+): 0.5*(relu(1-real)+relu(1+fake)) per
        head, averaged over the multi-scale heads."""
        loss = 0.0
        for lr_, lf_ in zip(logits_real, logits_fake):
            loss = loss + (0.5 * (F.relu(1.0 - lr_) + F.relu(1.0 + lf_))).mean()
        return loss / len(logits_real)

    def _x0_losses_invsr(self, out, tag):
        """v18.3 InvSR-calibrated distribution supervision (CVPR'5 2412.09013,
        trainer.py backward_step L1239-1312 adapted to this coach):

          domain : t <= t_max — their timesteps [100,250]; our MEASURED
                   validity boundary 300 (train_logs/diag_pair_domain.png:
                   beyond it the pair degenerates to noise-vs-noise)
          object : z0_hat = single-step x0 estimate (their z0_pred L1506-09),
                   clamped to ±latent_bound (their _Latent_bound ±10)
          losses : ldif = MSE(z0_hat, target_latents) x w_ldif      (L1263-71)
                   lfm  = feature matching over D head features x w_lfm
                          (stands in for their finetuned latent-LPIPS which
                          is not downloadable offline; note orig WarpGAN's
                          own third layer was FM x100 + adv x10 too)
                   ldis = -mean(logits) over heads x w_ldis, applied only
                          AFTER dis_warmup steps (their L1286 + L995-997)
          target : out['target_latents'] — pass2: the real PHOTO latents;
                   synth: the paired render latents (pixel-paired, same view)
          D pairs: queued (tag, fake.detach, target.detach, t) -> hinge D step.

        All in LATENT space — no VAE decode in the loss (their design; also
        frees the decode memory that pushed wave-1 to 22.7 GiB).
        """
        metrics = {}
        t = out['timesteps']
        sel = t <= self.x0_t_max
        zero = out['eps_pred'].new_zeros(())
        if not bool(sel.any()):
            metrics[f'{tag}_hit'] = 0.0
            return zero, metrics
        metrics[f'{tag}_hit'] = 1.0
        metrics[f'{tag}_t'] = float(t[sel].float().mean())
        ab = self.noise_scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)
        x0_hat = (out['noisy_latents'][sel] - (1.0 - ab[sel]).sqrt() * out['eps_pred'][sel]) \
            / ab[sel].sqrt()
        x0_hat = x0_hat.clamp(-self.x0_latent_bound, self.x0_latent_bound)
        tgt = out['target_latents'][sel]
        t_sel = t[sel]
        prompt = self.empty_prompt_embeds.expand(x0_hat.shape[0], -1, -1)

        ldif = F.mse_loss(x0_hat, tgt)
        total = ldif * self.x0_w_ldif
        metrics[f'{tag}_ldif'] = float(ldif)

        set_requires_grad(self.discriminator, False)      # InvSR: D frozen at G step
        fake_logits, fake_feats = self.discriminator(x0_hat, t_sel, prompt,
                                                     return_features=True)
        if self.global_step > self.x0_dis_warmup and self.x0_w_ldis > 0:
            adv_g = sum(-lg.mean() for lg in fake_logits) / len(fake_logits)
            total = total + adv_g * self.x0_w_ldis
            metrics[f'{tag}_gen_adv'] = float(adv_g)
        if self.x0_w_lfm > 0:
            if self.llpips_loss is not None:
                # v18.4: InvSR's finetuned latent-LPIPS in the llpips slot —
                # a real perceptual-domain metric (the FM substitute measured
                # 0.001-0.009, i.e. numerically dead, at 50K steps).
                with torch.no_grad():
                    pass  # metric is frozen; graph flows through z0_hat input
                lfm = self.llpips_loss(x0_hat, tgt).view(-1).mean()
            else:
                with torch.no_grad():
                    _, real_feats = self.discriminator(tgt, t_sel, prompt,
                                                       return_features=True)
                lfm = feature_matching_loss(fake_feats, real_feats)
            total = total + lfm * self.x0_w_lfm
            metrics[f'{tag}_lfm'] = float(lfm)

        self._disc_queue_invsr.append(
            (tag, x0_hat.detach().clamp(-self.x0_latent_bound, self.x0_latent_bound),
             tgt.detach(), t_sel.detach()))
        return total, metrics

    def _gan_fm_g_loss(self, out, real_img, tag, weight=1.0):
        """G-side discriminator losses (orig coach L1169-1198 faithful port).

        fake = the SAME x0 single-step decode the identity layer rides on
        (exists only when t<200 fired); real = source photo downscaled to
        fake's resolution (orig real=x; orig itself has a 256 branch via
        warp_ori_train=False). FM is FULL-FRAME (orig mask_for_fm=None),
        G-side adversarial optional (orig x10; v14 default 0, FM-first).
        The pair is queued DETACHED for the D step — orig: batch['pred'].detach().
        """
        if not self.use_gan_fm or 'pred_x0_img_g' not in out:
            return None
        fake = out['pred_x0_img_g']
        real = F.interpolate(real_img.detach(), size=fake.shape[-2:],
                             mode='bilinear', align_corners=False)
        set_requires_grad(self.discriminator, False)   # orig: D frozen at G step
        discr_real_pred, discr_real_features = self.discriminator(real)
        discr_fake_pred, discr_fake_features = self.discriminator(fake)
        metrics, total = {}, fake.new_zeros(())
        if self.adv_weight > 1e-5:
            adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(
                real_batch=real, fake_batch=fake,
                discr_real_pred=discr_real_pred,
                discr_fake_pred=discr_fake_pred, mask=None)
            total = total + adv_gen_loss
            metrics[f'{tag}_gen_adv'] = float(adv_gen_loss)
        if self.fm_weight > 1e-5:
            fm_value = feature_matching_loss(
                discr_fake_features, discr_real_features, mask=None)
            total = total + fm_value * self.fm_weight
            metrics[f'{tag}_gen_fm'] = float(fm_value)
            if self.smoke_check and not getattr(self, '_fm_fired_logged', False):
                self._fm_fired_logged = True
                print(f'[v14 G-side FM FIRED, real data] {tag}_gen_fm='
                      f'{float(fm_value):.5f} (x{self.fm_weight}) | '
                      f'D preds real {float(discr_real_pred.mean()):.3f} / '
                      f'fake {float(discr_fake_pred.mean()):.3f} | '
                      f'feats {len(discr_fake_features)} layers @ '
                      f'{tuple(discr_fake_features[0].shape[-2:])}..'
                      f'{tuple(discr_fake_features[-1].shape[-2:])}')
        # v18: queue carries the TAG so the D step logs per-pair health
        # (the pass2 pair and the synth pair were previously
        # indistinguishable in metrics); the caller's SNR weight (ab) is
        # applied HERE to the returned total.
        self._disc_queue.append((tag, fake.detach(), real.detach()))
        return total * float(weight), metrics

    def _discriminator_step(self):
        """D update on the queued detached pairs (orig coach L1215-1240:
        pred.detach(), pre_discriminator_step -> R1 needs real.requires_grad,
        inline gp_coef=0.001, external backward, then optimizer step)."""
        if not self.use_gan_fm or not self._disc_queue:
            return None
        set_requires_grad(self.discriminator, True)
        self.optimizer_d.zero_grad(set_to_none=True)
        total, m = 0.0, {}
        for dtag, fake, real in self._disc_queue:
            real = real.clone().requires_grad_(True)   # R1 gradient penalty input
            self.adversarial_loss.pre_discriminator_step(
                real_batch=real, fake_batch=fake,
                generator=None, discriminator=self.discriminator)
            discr_real_pred, _ = self.discriminator(real)
            discr_fake_pred, _ = self.discriminator(fake)
            adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(
                real_batch=real, fake_batch=fake,
                discr_real_pred=discr_real_pred,
                discr_fake_pred=discr_fake_pred, mask=None)
            adv_discr_loss.backward()
            total += float(adv_discr_loss)
            # v18: PER-PAIR (per-batch-type) health — the v14 shortcut
            # diagnostics (viewpoint / noise shortcuts) live exactly here: if
            # the synth pair separates much faster than the pass2 pair, D is
            # again keying on a nuisance axis instead of content/texture.
            m.update({f'{dtag}_{k}': float(v) for k, v in adv_metrics.items()})
        self.optimizer_d.step()
        if self.smoke_check and not getattr(self, '_dstep_logged', False):
            self._dstep_logged = True
            print(f'[v18 D-step FIRED] pairs={len(self._disc_queue)} '
                  f'discr_adv={total:.4f} | ' +
                  ' | '.join(f'{k}={v:.4f}' for k, v in m.items()))
        self._disc_queue = []
        return total, m

    def _discriminator_step_invsr(self):
        """InvSR D update (trainer L994-1025): hinge over every head on the
        queued latent pairs; updates EVERY iteration (their
        dis_update_freq=1; G sees ldis only after dis_warmup)."""
        if not self._invsr() or not self._disc_queue_invsr:
            return None
        set_requires_grad(self.discriminator, True)
        self.optimizer_d.zero_grad(set_to_none=True)
        total, m = 0.0, {}
        for dtag, fake, real, tt in self._disc_queue_invsr:
            prompt = self.empty_prompt_embeds.expand(fake.shape[0], -1, -1)
            logits_real, _ = self.discriminator(real, tt, prompt, return_features=True)
            logits_fake, _ = self.discriminator(fake, tt, prompt, return_features=True)
            loss = self._hinge_d_loss_invsr(logits_real, logits_fake)
            loss.backward()
            total += float(loss)
            m[f'{dtag}_d_real'] = float(sum(l.mean() for l in logits_real) / len(logits_real))
            m[f'{dtag}_d_fake'] = float(sum(l.mean() for l in logits_fake) / len(logits_fake))
        self.optimizer_d.step()
        self._disc_queue_invsr = []
        return total, m

    def _gan_fm_grad_norms(self):
        """Stage-B stop-loss gauge: per-block |grad| sums for G and D."""
        if not (self.use_gan_fm or self._invsr()):
            return {}
        gn = sum(float(p.grad.abs().sum()) for p in self.brushnet.parameters()
                 if p.grad is not None)
        dn = sum(float(p.grad.abs().sum()) for p in self.discriminator.parameters()
                 if p.grad is not None)
        return {'g_grad_abs': gn, 'd_grad_abs': dn}



    # ------------------------------------------------------------------
    # batch forwards (mirror the orig coach's dual-pass structure)
    # ------------------------------------------------------------------

    def _soften_cond_mask(self, mask):
        """v10: CONDITION-side mask softening, faithful port of orig
        process_mask (coach_inpainting_static L186-198) — BINARIZE, erode the
        VISIBLE region by k pixels, then Gaussian-blur, in PIXEL space at the
        mask's native resolution (512).
        Input-fidelity note (2026-08-27 real-data audit): the orig splatting
        yields a BINARY vis ((warped==0).all -> {0,1}); our real-batch masks
        measured binary as well, but continuous confidences are possible
        (pass2/synth paths, future warper changes) — grayscale min-pool erosion
        on a continuous field would eat every local confidence dip. So we
        threshold at 0.5 FIRST to pin the orig's effective input distribution,
        then run the orig op sequence. Supervision keeps the raw mask.
        Op fidelity (verified numerically vs /data/xzy/warpgan_orig):
          * erosion: kornia km.erosion(ones(3,3)) == -max_pool2d(-vis) EXACTLY
            (max_diff 0.00e+00, incl. borders — kornia 'geodesic' border ==
            -inf maxpool padding).
          * blur: torchvision GaussianBlur(21) samples sigma ~ U(0.1, 2.0)
            per call (tv 0.23 source, default sigma=(0.1, 2.0)); we fix sigma
            at the midpoint 1.05 for determinism (band ~4px, max_jump 0.380),
            and pad with mode='reflect' exactly like tv
            (_functional_tensor.py L760) — replicate differed by up to 0.29
            where the hole reaches the image frame.
        Used ONLY for the hybrid condition mixing and the BrushNet mask
        channel. The supervision side keeps the raw mask (orig also feeds
        the SOFT mask into its l1/FM loss weights — deliberate single-
        variable omission here, see STEP0_SMOKE_REPORT §8.27)."""
        if not self.cond_soften:
            return mask
        vis = (1.0 - mask) >= 0.5
        vis = vis.to(mask.dtype)                       # orig input: binary vis
        if self.cond_erode_kernel > 0:
            k = self.cond_erode_kernel
            pad = k // 2
            vis = -F.max_pool2d(-vis, k, stride=1, padding=pad)  # local min = erosion
        if self.cond_blur_kernel > 0:
            assert mask.dim() == 4, f"cond mask must be (B,1,H,W), got {tuple(mask.shape)}"
            k = self.cond_blur_kernel
            x = torch.arange(k, device=mask.device, dtype=torch.float32) - (k - 1) / 2.
            g = torch.exp(-0.5 * (x / 1.05) ** 2)   # orig: sigma~U(0.1,2.0), fixed midpoint
            g = (g / g.sum())
            ch = vis.shape[1]
            p = k // 2
            # torchvision gaussian_blur pads with mode="reflect" (tv 0.23
            # _functional_tensor.py L760) — replicate was wrong by up to 0.29
            # mask units where the hole reaches the image frame (real-data
            # audit 2026-08-27)
            vis = F.conv2d(F.conv2d(F.pad(vis, (p, p, p, p), mode='reflect'),
                                    g.view(1, 1, 1, k).repeat(ch, 1, 1, 1), groups=ch),
                          g.view(1, 1, k, 1).repeat(ch, 1, 1, 1), groups=ch)
        return (1.0 - vis).clamp(0.0, 1.0)

    def _ref_inputs(self, primary_img, mirror_img, primary_tag):
        images, tags = [primary_img], [primary_tag]
        if self.use_mirror:
            if mirror_img is None:
                print('[ref] use_mirror=True but no mirror image in batch; '
                      'falling back to primary-only bank')
            else:
                images.append(mirror_img)
                tags.append('x_mirror')
        return images, tags

    def _ref_inputs_real(self, y_hat_novel, mirror_img):
        """REAL pass1/validate reference arrangement (user's plan 2026-08-26).

        real_primary=mirror: the REAL PHOTO is the primary bank source (its
        texture is retrieved by the MAIN attn1 K/V/O adapter channel, in
        parallel with W+ identity tokens); the render demotes to the extra
        (mirror-KV) channel. render: old order."""
        if self.real_primary_mirror and mirror_img is not None:
            return [mirror_img, y_hat_novel], ['x_mirror', 'y_hat_novel']
        return self._ref_inputs(y_hat_novel, mirror_img, 'y_hat_novel')

    def _build_novel_view(self, x, depth, c, c_novel, y_hat_novel, depth_novel,
                          x_mirror=None, c_mirror=None, depth_mirror=None):
        """Shared novel-view construction (v7).

        CONDITION (back to the v4/orig contract): hybrid — visible = splat warp,
        hole = EG3D render. The mirror contributes ZERO condition pixels
        (v6 lesson: the copy-prior amplifies projection error as fragmentation).

        SUPERVISION (C'): continuous confidence — mirror transfer where the
        geometry supports it, blended by w = clamp(1 - depth_mismatch/eps, 0, 1).
        The depth-consistency threshold is demoted from a binary verdict to an
        attenuation scale: no cliff, no salt-and-pepper, no seams. The anchor
        mixes mirror-real and EG3D by the SAME w; the loss weight map follows
        the same w. mirror off -> exact v4 behavior everywhere.
        """
        warp_img, vis_mask, _ = self.warper.forward_warp(img1=x, depth1=depth, c1=c, c2=c_novel)
        mask = 1.0 - vis_mask
        with torch.no_grad():
            inv_warp, inv_valid = self.warper_ext.inverse_warp(
                img2=x, depth1=depth_novel, depth2=depth, c1=c_novel, c2=c)
            vis_eff = inv_valid.clamp(0, 1) * (1.0 - mask)
            # v7 confidence field: continuous in the depth mismatch
            w_mirror = torch.zeros_like(vis_eff)
            inv_warp_m = None
            if self.mirror_anchor and x_mirror is not None:
                inv_warp_m, _, conf_m = self.warper_ext.inverse_warp(
                    img2=x_mirror, depth1=depth_novel, depth2=depth_mirror,
                    c1=c_novel, c2=c_mirror, return_mismatch=True)
                w_mirror = conf_m.clamp(0, 1) * mask  # continuous in (0,1)
            hole_real = w_mirror                     # soft coverage, no binarization
            blind = (1.0 - vis_eff - hole_real).clamp(0, 1)

            # supervision anchor: real (source) | real (mirror, weighted by w) | EG3D
            anchor = inv_warp * vis_eff \
                + (inv_warp_m * hole_real if inv_warp_m is not None else 0) \
                + y_hat_novel * blind
            # v18.7 (user feedback 2026-09-14): SDE-path defects = ANCHOR
            # defects (pinholes/沙沙, skin->hair bleed, near-boundary
            # background noise). Single-pixel coverage gaps and sub-pixel
            # misalignment in the inverse-warped evidence are IMPULSE noise —
            # the textbook response is a median filter, applied to the
            # supervision target ONLY (condition untouched). 0 = off.
            if self.anchor_median > 0:
                from kornia.filters import median_blur
                anchor = median_blur(anchor, (self.anchor_median,) * 2)
            # condition: orig hybrid contract (EG3D fills the hole). v10: the
            # MIXING uses the softened mask (orig process_mask semantics) —
            # the seam the BrushNet sees is a ~4px smooth blend (erode3+blur21
            # σ=1.05, verified vs orig), not a knife edge. Supervision below
            # keeps the raw mask untouched.
            mask_cond = self._soften_cond_mask(mask)
            cond = warp_img * (1.0 - mask_cond) + y_hat_novel * mask_cond
        return dict(mask=mask, cond=cond, anchor=anchor, warp_img=warp_img,
                    vis_eff=vis_eff, hole_eff=hole_real, blind=blind,
                    inv_warp_m=inv_warp_m, mask_cond=mask_cond)

    def _compute_hf_map(self, img):
        """AnyDoor high-frequency map (datasets/data_utils.py sobel(), PyTorch
        port). Input: [0,1] RGB (B,3,H,W). Output: [0,1] edge-weighted image
        (B,3,H,W) — only edge-rich areas (skin pores, hair strands) retain
        their content; smooth areas are zeroed. This is the 'detail texture'
        conditioning stream, SEPARATE from identity (W+/DINOv2)."""
        gray = img.mean(dim=1, keepdim=True)  # (B,1,H,W)
        # Sobel k=3, OpenCV-equivalent kernels
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
        gx = F.conv2d(gray, kx, padding=1).abs()
        gy = F.conv2d(gray, ky, padding=1).abs()
        # addWeighted(|Gx|, 0.5, |Gy|, 0.5) → then max across channels
        # (input is already gray, so max is identity)
        edge = 0.5 * gx + 0.5 * gy  # (B,1,H,W)
        # Threshold (AnyDoor: thresh=50 on 0-255 → 50/255 on [0,1])
        edge = torch.where(edge < self.hf_thresh,
                           torch.zeros_like(edge), edge)
        # Multiply with original: only edge-rich areas keep content
        return img * edge  # (B,3,H,W)

    def _expand_brushnet_conditioning(self, extra_ch=3):
        """Surgically expand BrushNet's conv_in_condition from 5ch to (5+extra_ch)
        conditioning input. Pretrained 5ch weights are PRESERVED; the new
        channels are ZERO-initialized (ControlNet convention: new pathways
        start inert, training learns to use them). This is done ONCE at
        model-build time when hf_enable=True."""
        old = self.brushnet.conv_in_condition  # Conv2d(4+5, C, 3, padding=1)
        old_in = old.in_channels  # 9 (= 4 latent + 5 cond)
        new_in = old_in + extra_ch  # 12 (= 4 latent + 8 cond)
        new = torch.nn.Conv2d(
            new_in, old.out_channels, old.kernel_size, old.stride,
            old.padding, old.dilation, old.groups,
            bias=old.bias is not None, padding_mode=old.padding_mode,
            device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight[:, :old_in] = old.weight  # preserve pretrained
            new.weight[:, old_in:] = 0.0          # zero-init new channels
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.brushnet.conv_in_condition = new
        print(f'[v18.8] BrushNet conv_in_condition expanded: '
              f'{old_in} → {new_in} input channels '
              f'(+{extra_ch} HF, zero-initialized)')

    def _forward_real(self, b, t_override=None):
        """Real batch, orig coach L207-305 semantics: pass1 novel + pass2 source."""
        x, c, codes = b['x'], b['c'], b['codes']
        depth, y_hat = b['depth'], b['y_hat']
        c_novel, y_hat_novel, depth_novel = b['c_novel'], b['y_hat_novel'], b['depth_novel']
        metrics, panels = {}, {'x': x.detach(), 'y_hat': y_hat.detach(),
                               'y_hat_novel': y_hat_novel.detach()}

        # ---- pass 1 (novel): v6 — condition AND supervision share geometry ----
        nv = self._build_novel_view(
            x, depth, c, c_novel, y_hat_novel, depth_novel,
            x_mirror=b.get('x_mirror'), c_mirror=b.get('c_mirror'),
            depth_mirror=b.get('depth_mirror'))
        mask, cond1, anchor = nv['mask'], nv['cond'], nv['anchor']
        vis_eff, hole_eff = nv['vis_eff'], nv['hole_eff']
        if self.mirror_anchor:
            metrics['p1_mirror_hole_cov'] = float(
                hole_eff.sum() / (mask.sum() + 1e-6))
            if nv['inv_warp_m'] is not None:
                panels['inv_warp_mirror'] = nv['inv_warp_m'].detach()

        # v18 STRUCT: pass1 forward + supervision switchable. loss_p1=None
        # tells the train loop to skip its backward (metrics['total'] guards).
        loss_p1 = None
        if self.real_pass1_forward:
            ref_images, ref_tags = self._ref_inputs_real(y_hat_novel, b.get('x_mirror'))
            # v18.8: HF texture conditioning (AnyDoor). HF of the MIRROR photo
            # (the same-person texture source), forward-warped to the novel
            # view — same warp as the mirror condition. Holes = zero (no info).
            hf_img1 = None
            if self.hf_enable and b.get('x_mirror') is not None:
                with torch.no_grad():
                    hf_mirror = self._compute_hf_map(b['x_mirror'])
                    hf_warp, _, _ = self.warper.forward_warp(
                        img1=hf_mirror, depth1=b['depth_mirror'],
                        c1=b['c_mirror'], c2=c_novel)
                    hf_img1 = hf_warp
            out1 = self._diffusion_forward(
                target_img=anchor, condition_img=cond1, mask=mask, codes=codes,
                ref_images=ref_images, ref_tags=ref_tags,
                t_override=t_override, train_wplus=False,   # real update: W+ frozen
                mask_cond=nv.get('mask_cond'),
                t_range=(self.real_pass1_t_max or None),     # v18.6 deployed domain
                hf_img=hf_img1)
            if self.smoke_check and self.global_step <= 2:
                self._v10_audit(mask, nv['mask_cond'], out1['mask_latents'])
                if float(self.opts.losses.latent.weight) > 1e-5:
                    self._v13_identity_audit(x, codes)
            p1 = self.opts.losses.real_pass1
            if self.mirror_anchor:
                # evidence CONTINUOUS field (v6): photo pixels (source OR mirror
                # transfer) carry photo evidence; the blind residual has none.
                eff_img = (vis_eff + hole_eff).clamp(0, 1)
                # v12 audit, 3rd iteration: vis_eff contains the BINARY (1-mask)
                # factor -> a 1px STEP at the hole boundary. Any 8x8 block
                # average across a 1px step still yields adjacent latent weights
                # differing by up to ~0.9 (frequency-band seam). Fix: widen the
                # step into a ~1.5-latent transition band FIRST (13px box), then
                # area-downsample (=block average).
                k = 13
                box = torch.ones(1, 1, k, k, device=eff_img.device,
                                 dtype=eff_img.dtype) / float(k * k)
                eff_img = F.conv2d(F.pad(eff_img, (k // 2,) * 4, mode='replicate'), box)
                eff_latents = F.interpolate(eff_img, size=out1['mask_latents'].shape[-2:],
                                            mode='area')
            else:
                # full-review finding (2026-08-28): WITHOUT mirror_anchor there is
                # no continuous evidence field — this branch derives weights from
                # the BINARY mask, whose latent step would enter the dual-band
                # weights unsnoothed. The combination is untested; forbid it.
                assert self.blind_struct_weight <= 0, \
                    ('dual-band supervision (blind_struct_weight>0) requires '
                     'losses.real_pass1.mirror_anchor=True: the non-mirror branch '
                     'has no continuous evidence field (binary-mask weights would '
                     'reintroduce the latent-domain seam v12 just fixed)')
                eff_latents = 1.0 - out1['mask_latents']
            if self.x0_enable:
                # v17-prime: SINGLE full-band eps-MSE against the anchor (the
                # anchor itself keeps its continuous evidence construction — that
                # is DATA, not a loss gate). dual-band / sigma / 13px transition
                # band are RETIRED: texture content is now the distribution
                # supervision's job (D), geometry is the eps target's job.
                loss_p1 = self._eps_mse(out1['eps_pred'], out1['noise']) \
                    * float(self.opts.losses.eps_weight)
            elif self.blind_struct_weight > 0:
                # v12: evidence-graded dual-band supervision. Photo evidence ->
                # full band; blind -> LOW band only (target stays the FULL SHARP
                # render anchor; only the error's low-frequency component is
                # penalized -> geometry anchored, texture band zero gradient).
                w_full = eff_latents * float(p1.visible_weight)
                w_low = (1.0 - eff_latents) * self.blind_struct_weight
                loss_p1 = self._eps_mse_dual_band(
                    out1['eps_pred'], out1['noise'], w_full, w_low) \
                    * float(self.opts.losses.eps_weight)
                if self.smoke_check and self.global_step <= 2:
                    self._v12_audit(out1, w_full, w_low)
            else:
                # v11-and-earlier path: single-band weighted eps-MSE
                w_map = eff_latents * float(p1.visible_weight) \
                    + (1.0 - eff_latents) * float(p1.hole_weight)
                loss_p1 = self._eps_mse(out1['eps_pred'], out1['noise'], w_map) \
                    * float(self.opts.losses.eps_weight)
            # v15: blind high-frequency SCORE PRESERVATION — the control's HIGH
            # band in the blind region must not deviate from the photo-prior
            # teacher on the SAME noisy input. Low band is untouched: BrushNet
            # keeps its geometry mandate (the v12 anchor stays in force).
            if self.preserve_weight > 0 and out1.get('eps_teacher') is not None:
                offset = out1['eps_pred'].float() - out1['eps_teacher'].float()
                hp = offset - self._lowpass_eps(offset)
                w_blind_p = (1.0 - eff_latents).clamp(0.0, 1.0)  # same continuous field (v12)
                loss_pres = (hp.pow(2) * w_blind_p).sum() \
                    / (w_blind_p.sum() * hp.shape[1] + 1e-6)
                loss_p1 = loss_p1 + loss_pres * self.preserve_weight
                metrics['p1_preserve'] = float(loss_pres)
                if self.smoke_check and self.global_step <= 2:
                    self._v15_audit(out1, offset, hp, w_blind_p, loss_pres)

            # v14: G-side FM(/adv) on the x0 decode vs source photo (orig third
            # layer; rides the SAME t<200 self-gated path as id/latent).
            # pass1 has NO _low_t_losses (spec §5.3: eps-anchor only) — the orig
            # D judged pred_novel directly, so the diffusion port must decode
            # pass1's x0 for the SAME purpose. Decode ONLY (no pixel/id/latent
            # added to pass1 — single-mandate, faithful to both).
            if self.use_gan_fm and bool((out1['timesteps'] < self.pixel_max_t).any()):
                x0_1 = self._predict_x0(out1['eps_pred'], out1['noisy_latents'],
                                        out1['timesteps'])
                sel1 = out1['timesteps'] < self.pixel_max_t
                out1['pred_x0_img_g'] = self._decode_x0_to_pixel(
                    x0_1[sel1].clamp(-self.x0_clip, self.x0_clip), self.pixel_res)
                out1['pred_x0_img'] = out1['pred_x0_img_g'].detach()
                if 'pred_x0_img' not in panels:
                    panels['p1_pred_x0'] = out1['pred_x0_img']
            # v14 G-side FM on the pass1 x0 decode — LEGACY path only. Under
            # x0_enable the pass1 output gets NO distribution supervision by
            # design: the novel view has no photo-distribution reference (the v14
            # failure mode), quality transfers from pass2 through the shared net.
            if not self.x0_enable:
                gm1 = self._gan_fm_g_loss(out1, x, 'p1')
                if gm1 is not None:
                    loss_p1 = loss_p1 + gm1[0]
                    metrics.update(gm1[1])
            metrics['p1_eps'] = float(loss_p1)
        else:
            # v18 STRUCT: pass1 entirely unsupervised on real batches (orig
            # semantics for pred_novel); blind-region content transfers from
            # pass2/synth training through the shared network.
            metrics['p1_skipped'] = 1.0
        metrics['p1_hole_frac'] = float(mask.mean())
        panels.update({'warp_img': nv['warp_img'].detach(), 'cond1': cond1.detach(),
                       'mask': mask.detach().expand_as(x), 'anchor': anchor.detach()})

        # ---- pass 2 (source): the ONLY real-photo supervision (warp_pred=False) ----
        warp_warp_img, vis_inv, _ = self.warper.forward_warp(
            img1=cond1, depth1=depth_novel, c1=c_novel, c2=c)
        mask_inv = 1.0 - vis_inv
        # v10: pass2 condition softened the same way (orig routes pass2 through
        # get_inp/process_mask too)
        mask_inv_cond = self._soften_cond_mask(mask_inv)
        cond2 = warp_warp_img * (1.0 - mask_inv_cond) + y_hat * mask_inv_cond
        # v17-prime: teacher's rule ① — the pass2 reference bank is
        # warp_warp_img ("the twice-warped result") with y_hat as the extra
        # source. Photo pixels, aligned to the supervision viewpoint.
        # v18 STRUCT: with pass1 disabled, the mirror PHOTO rides pass2's bank
        # instead of y_hat — the orig kept its mirror condition channel in the
        # SUPERVISED pass (coach L255-266); without this the mirror-retrieval
        # pathway would only ever train on synth flipped RENDERS while
        # inference feeds the mirror PHOTO. warp_warp_img already carries the
        # inversion fill (its hole content IS y_hat_novel warped back), so the
        # extra slot goes to the strongest evidence: the real photo mirror.
        if self.x0_enable and not self.real_pass1_forward:
            ref_images2, ref_tags2 = [warp_warp_img, b['x_mirror']], \
                                     ['warp_warp_img', 'x_mirror']
        elif self.x0_enable:
            ref_images2, ref_tags2 = [warp_warp_img, y_hat], ['warp_warp_img', 'y_hat']
        else:
            ref_images2, ref_tags2 = self._ref_inputs(y_hat, b.get('x_mirror'), 'y_hat')
        # v18.8: HF for pass2 = HF of x (source view, already aligned; no warp)
        hf_img2 = None
        if self.hf_enable:
            with torch.no_grad():
                hf_img2 = self._compute_hf_map(x)
        out2 = self._diffusion_forward(
            target_img=x, condition_img=cond2, mask=mask_inv, codes=codes,
            ref_images=ref_images2, ref_tags=ref_tags2,
            t_override=t_override, train_wplus=False,
            mask_cond=mask_inv_cond,
            hf_img=hf_img2)
        loss_p2_eps = self._eps_mse(out2['eps_pred'], out2['noise']) \
            * float(self.opts.losses.eps_weight)
        if self.x0_enable:
            # v17-prime: gate-free x0 supervision replaces the t<200 pixel path
            # v18.3: invsr mode = InvSR-calibrated latent supervision
            if self.x0_mode == 'invsr':
                loss_p2_low, m_low = self._x0_losses_invsr(out2, 'p2')
            else:
                loss_p2_low, m_low = self._x0_losses(out2, x, None, 'p2')
        else:
            loss_p2_low, m_low = self._low_t_losses(out2, x, codes)
        metrics['p2_eps'] = float(loss_p2_eps)
        if self.x0_enable:
            metrics.update(m_low)      # keys already carry the p2_ prefix
        else:
            metrics.update({f'p2_{k}': v for k, v in m_low.items()})
        panels.update({'warp_warp_img': warp_warp_img.detach(),
                       'cond2': cond2.detach(),
                       'mask_inv': mask_inv.detach().expand_as(x)})
        if 'pred_x0_img' in out2:
            panels['p2_pred_x0'] = out2['pred_x0_img']

        loss_p2 = loss_p2_eps + loss_p2_low
        # v14 legacy G-side FM for pass2 — under x0_enable the FM already ran
        # INSIDE _x0_losses on the re-noised symmetric pair (real_t), so this
        # duplicate call is bypassed.
        if not self.x0_enable:
            gm2 = self._gan_fm_g_loss(out2, x, 'p2')
            if gm2 is not None:
                loss_p2 = loss_p2 + gm2[0]
                metrics.update(gm2[1])
        metrics['total'] = float((loss_p1 if loss_p1 is not None else 0.0) + loss_p2)
        metrics['p2_timestep'] = int(out2['timesteps'][0])
        # separate tensors: the train loop backwards them one at a time so the
        # two 512px graphs never coexist (peak-memory discipline)
        return loss_p1, loss_p2, metrics, panels

    def _forward_synth(self, b, t_override=None):
        """Synth batch (orig coach L489-560): the ONLY place an EG3D render is
        a supervision target. Full-frame eps + low-t pixel + latent closure."""
        x, c, codes = b['x'], b['c'], b['codes']
        depth, y_hat_novel, depth_novel = b['depth'], b['y_hat_novel'], b['depth_novel']
        c_novel, target = b['c_novel'], b['target']
        metrics, panels = {}, {'x': x.detach(), 'y_hat_novel': y_hat_novel.detach(),
                               'target': target.detach()}

        warp_img, vis_mask, _ = self.warper.forward_warp(img1=x, depth1=depth, c1=c, c2=c_novel)
        mask = 1.0 - vis_mask
        # v10: synth condition softened (synth batch goes through get_inp/
        # process_mask in the orig as well)
        mask_cond = self._soften_cond_mask(mask)
        cond = warp_img * (1.0 - mask_cond) + y_hat_novel * mask_cond
        ref_images, ref_tags = self._ref_inputs(y_hat_novel, b.get('x_mirror'), 'target_hat')
        # v18.8: HF for synth = HF of the source render, warped to target view
        hf_img_s = None
        if self.hf_enable:
            with torch.no_grad():
                hf_src = self._compute_hf_map(x)
                hf_warp_s, _, _ = self.warper.forward_warp(
                    img1=hf_src, depth1=depth, c1=c, c2=c_novel)
                hf_img_s = hf_warp_s
        out = self._diffusion_forward(
            target_img=target, condition_img=cond, mask=mask, codes=codes,
            ref_images=ref_images, ref_tags=ref_tags,
            t_override=t_override, train_wplus=True,    # synth update: W+ may train
            mask_cond=mask_cond,
            hf_img=hf_img_s)
        if self.synth_texture == 'full':
            loss_eps = self._eps_mse(out['eps_pred'], out['noise']) \
                * float(self.opts.losses.eps_weight)
        else:  # 'zero' (user plan: texture band gets no gradient) or 'lowpass'
            loss_eps = self._eps_mse_structure_only(out['eps_pred'], out['noise']) \
                * float(self.opts.losses.eps_weight)
        # v16: synth pass keeps the paint-oil red line — its target is an EG3D
        # render; the photo-ref pixel backflow is REAL-pass2-only by design.
        if self.x0_enable:
            # v18 P1 FIX: eps stays FULL-BAND vs the paired render; the x0
            # distribution pair is now VIEW-ALIGNED BY CONSTRUCTION —
            # photo_ref=None makes real_t = out['noisy_latents'] (the noised
            # TARGET latents): same view as fake (novel), pixel-paired, same
            # noise level. This exactly mirrors the orig synth D pair (orig
            # coach L530-536: real=target_img vs fake=pred_novel) and removes
            # the v14-class viewpoint shortcut the photo_ref=b['x']
            # construction carried (b['x'] is the SOURCE-view render while
            # fake decodes the NOVEL view).
            loss_low, m_low = self._x0_losses_invsr(out, 'x0') if self.x0_mode == 'invsr' \
                else self._x0_losses(out, target, None, 'x0')
        else:
            loss_low, m_low = self._low_t_losses(
                out, target, codes, allow_pixel=self.pixel_apply_synth)
        metrics['eps'] = float(loss_eps)
        metrics['eps_raw'] = float(self._eps_mse(out['eps_pred'], out['noise']))
        metrics.update(m_low)
        panels.update({'warp_img': warp_img.detach(), 'cond': cond.detach(),
                       'mask': mask.detach().expand_as(x)})
        if 'pred_x0_img' in out:
            panels['pred_x0'] = out['pred_x0_img']

        total = loss_eps + loss_low
        metrics['total'] = float(total)
        metrics['timestep'] = int(out['timesteps'][0])
        return total, metrics, panels


    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------

    def _zero_wplus_grads(self):
        """§4.2 red line: the W+ branch NEVER updates on real-batch steps."""
        for p in self.wplus_mapper_params + self.wplus_qkv_params:
            if p.grad is not None:
                p.grad = None

    def _wplus_params_for_audit(self):
        if not self.use_wplus:
            return []
        params = list(self.w_mapper.parameters())
        for proc in self.denoising_unet.attn_processors.values():
            if isinstance(proc, ReferenceAttentionProcessor) and proc.wplus_scale is not None:
                params += list(proc.parameters())
        return params

    def train(self):
        print('Start diffusion-port training (orig semantics, BrushNet+SD1.5).')
        torch.cuda.reset_peak_memory_stats(self.device)
        real_iterator = iter(self.train_dataloader)
        synth_iterator = iter(self.train_synth_dataloader) if self.use_synth else None

        while self.global_step < int(self.opts.max_steps):
            # ---------------- real update (pass1 + pass2, ONE optimizer step) --
            try:
                batch_real = next(real_iterator)
            except StopIteration:
                print('real_iterator exhausted -> reset')
                real_iterator = iter(self.train_dataloader)
                batch_real = next(real_iterator)
            b_real = self._parse_real_batch(batch_real)

            self.optimizer.zero_grad(set_to_none=True)
            loss_p1, loss_p2, m_real, panels_real = self._forward_real(
                b_real, t_override=self.fix_timestep)
            if loss_p1 is not None:   # v18 STRUCT: pass1 may be disabled
                loss_p1.backward()
            loss_p2.backward()
            self._zero_wplus_grads()
            self.optimizer.step()
            # v14: D update on the pairs queued during the G forward (orig
            # coach L405-448: every G step is followed by a D step; ours only
            # when the t<200 gate produced a clean x0 pair — G and D update at
            # the SAME gating frequency, so the game stays symmetric)
            d_res = self._discriminator_step()
            metrics = {f'real_{k}': v for k, v in m_real.items()}
            if d_res is not None:
                d_total, d_m = d_res
                metrics['discr_adv'] = d_total
                metrics.update(d_m)      # v18: per-batch-type D health
                metrics.update(self._gan_fm_grad_norms())
            # v18.3: InvSR-mode D update on the latent pairs
            if self._invsr():
                d2 = self._discriminator_step_invsr()
                if d2 is not None:
                    metrics['discr_adv'] = d2[0]
                    metrics.update(d2[1])
                    metrics.update(self._gan_fm_grad_norms())

            # ---------------- synth update (1:1, orig coach L489-560) ----------
            if self.use_synth:
                try:
                    batch_synth = next(synth_iterator)
                except StopIteration:
                    print('synth_iterator exhausted -> reset')
                    synth_iterator = iter(self.train_synth_dataloader)
                    batch_synth = next(synth_iterator)
                b_synth = self._parse_synth_batch(batch_synth)

                self.optimizer.zero_grad(set_to_none=True)
                loss_synth, m_synth, panels_synth = self._forward_synth(
                    b_synth, t_override=self.fix_timestep)
                loss_synth.backward()
                self.optimizer.step()
                metrics.update({f'synth_{k}': v for k, v in m_synth.items()})

            # ---------------- v18.8: prior preservation (DreamBooth) ---------
            # Every N steps: one extra forward where condition=target=real
            # photo (identity mapping). Teaches BrushNet 'photo condition →
            # photo output', preventing the render-heavy fine-tuning from
            # eroding SD's photographic prior.
            if (self.prior_preserve_freq > 0
                    and self.global_step % self.prior_preserve_freq == 0):
                self.optimizer.zero_grad(set_to_none=True)
                _x = b_real['x']
                with torch.no_grad():
                    hf_pp = self._compute_hf_map(_x) if self.hf_enable else None
                out_pp = self._diffusion_forward(
                    target_img=_x, condition_img=_x,
                    mask=torch.zeros(1, 1, _x.shape[-2], _x.shape[-1],
                                     device=_x.device),
                    codes=b_real['codes'],
                    ref_images=[], ref_tags=[],
                    train_wplus=False,
                    hf_img=hf_pp)
                loss_pp = self._eps_mse(out_pp['eps_pred'], out_pp['noise'])
                loss_pp.backward()
                self.optimizer.step()
                metrics['prior_preserve'] = float(loss_pp)

            # ---------------- logging ----------------
            if self.global_step % int(self.opts.log.board_interval) == 0:
                for k, v in metrics.items():
                    self.logger.add_scalar(k, v, self.global_step)
                if self.mirror_attention_procs:
                    # v7 soft-retrieval gauge: mean softmax mass on the EXTRA
                    # tokens (measured, not parameterized). ~0.5 at init when
                    # extra tokens equal primary tokens in count; the TRAINED
                    # trend is the signal. v9 NOTE: with real_primary=mirror
                    # the extra channel carries the RENDER -> the curve name
                    # changes accordingly; read it as "residual render trust".
                    shares = [p.mirror_attn_share for p in self.mirror_attention_procs
                              if p.mirror_attn_share is not None]
                    if shares:
                        sm = sum(shares) / len(shares)
                        gauge = 'extra_render_attn_share' if self.real_primary_mirror \
                            else 'mirror_attn_share'
                        metrics[gauge] = sm
                        self.logger.add_scalar(gauge, sm, self.global_step)
                print(f"[step {self.global_step}] " + ' '.join(
                    f"{k}={v:.4f}" for k, v in metrics.items()
                    if isinstance(v, float)))

            if self.global_step % int(self.opts.log.image_interval) == 0:
                self._save_panels(panels_real, 'real')
                if self.use_synth:
                    self._save_panels(panels_synth, 'synth')

            if (self.global_step + 1) % int(self.opts.log.save_interval) == 0 \
                    or self.global_step + 1 == int(self.opts.max_steps):
                self.checkpoint_me(metrics)

            if (self.global_step + 1) % int(self.opts.log.val_interval) == 0:
                self.validate()

            self.global_step += 1

        peak_gb = torch.cuda.max_memory_allocated(self.device) / 2 ** 30
        print(f'[done] peak GPU memory: {peak_gb:.2f} GiB '
              f'(budget <22 GiB on 24GB card)')
        if self.smoke_check:
            self._smoke_audit(metrics, peak_gb)
        self.logger.close()

    def _v10_audit(self, mask_raw, mask_cond, mask_latents_sup):
        """v10 REAL-DATA audit (user request 2026-08-27): verify the theory AND
        the wiring on actual pipeline tensors — the splatting mask of a real
        batch, before any training matters. Fails loudly (AssertionError) so a
        broken v10 can never reach a long run."""
        print('\n' + '-' * 78)
        print('[v10 REAL-DATA AUDIT — condition-side softening]')
        fails = []
        # torch 2.8 compat fix (2026-09-09, env bring-up): Tensor.detach() now
        # ALWAYS returns a new object (older torch returned self for grad-free
        # tensors), so the off-state IDENTITY check must compare the original
        # tensors, not the detached copies. Intent unchanged: mask_cond must be
        # the very same tensor object as the raw mask when softening is off.
        raw, soft = mask_raw.detach(), mask_cond.detach()

        def chk(name, ok, extra=''):
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:44s} {extra}")
            if not ok:
                fails.append(name)

        if not self.cond_soften:
            chk('soften OFF: mask_cond IS raw mask (old behavior)',
                mask_cond is mask_raw)
            print('-' * 78)
            if fails:
                raise AssertionError(f'v10 AUDIT FAILED: {fails}')
            print('v10 AUDIT PASSED (off-state identity verified on real data).')
            return

        r, s = raw[:, 0], soft[:, 0]
        chk('output range [0,1]',
            bool((soft >= 0).all() and (soft <= 1).all()),
            f'min={float(soft.min()):.4f} max={float(soft.max()):.4f}')
        # growth is DESCRIPTIVE, not gating: on heavily fragmented real masks
        # the ORIG ITSELF grows the hole mean by >0.1 (fine visible specks die
        # under the 3x3 erosion; verified: our output is bitwise-equal to the
        # orig op chain on these very tensors). Correctness is the orig
        # re-implementation check below.
        bin_mask = ((1.0 - raw.float()) < 0.5).to(raw.dtype)
        growth = float((soft.mean() - bin_mask.mean()).abs())
        print(f"  [info] hole mean growth = {growth:.4f} "
              f"(raw {float(raw.mean()):.4f} bin {float(bin_mask.mean()):.4f} "
              f"soft {float(soft.mean()):.4f}; fragmentation-dependent, "
              f"orig-equivalent by the check below)")
        # THE fidelity check: our soften == orig ops (binarize; kornia
        # erosion; torchvision GaussianBlur(21, sigma=1.05)) on THIS tensor
        import kornia.morphology as _km
        from torchvision import transforms as _tvT
        vis_o = (1.0 - raw.float()) >= 0.5
        vis_o = vis_o.to(raw.dtype)
        vis_o = _km.erosion(vis_o, torch.ones(3, 3, device=raw.device, dtype=raw.dtype))
        vis_o = _tvT.GaussianBlur(21, sigma=1.05)(vis_o)
        orig_soft = 1.0 - vis_o
        diff = float((soft.float() - orig_soft).abs().max())
        chk('vs ORIG ops on real tensor (kornia+tv, max diff < 1e-4)',
            diff < 1e-4, f'max_diff={diff:.2e}')
        band = ((s > 0.01) & (s < 0.99))
        band_frac = float(band.float().mean())
        has_boundary = bool(((r < 0.01) & (r > -0.01)).any() and (r > 0.99).any())
        chk('transition band exists (soft, not re-binarized)',
            (not has_boundary) or band_frac > 0.0,
            f'band_frac={band_frac:.5f} band_px/img={float(band.sum()) / r.shape[0]:.0f}')
        jump = float((s[:, :, 1:] - s[:, :, :-1]).abs().max()) if s.shape[-1] > 1 else 0.0
        chk('no knife edge: max per-px jump <= 0.45 (orig ~0.38)',
            jump <= 0.45, f'max_jump={jump:.3f}')
        # supervision/condition split: BrushNet channel = SOFT, supervision = RAW
        sup_ref = F.interpolate(raw.float(), size=mask_latents_sup.shape[-2:],
                                mode='nearest')
        sup_ok = bool(torch.equal(mask_latents_sup, sup_ref.to(mask_latents_sup.dtype)))
        chk('supervision mask_latents == interp(RAW mask)', sup_ok)
        brush_ref = F.interpolate(soft.float(), size=mask_latents_sup.shape[-2:],
                                  mode='nearest')
        split_diff = float((brush_ref - mask_latents_sup.float()).abs().max())
        chk('BrushNet channel != supervision at boundary (split real)',
            (not has_boundary) or split_diff > 0.01,
            f'max|soft-raw|@latent={split_diff:.3f}')
        # off-state identity on the SAME real tensor: flipping the switches
        # off must return the very same object (zero-risk resume of v9-style)
        e_k, b_k, sw = self.cond_erode_kernel, self.cond_blur_kernel, self.cond_soften
        try:
            self.cond_soften = False
            chk('off-state identity on real tensor',
                self._soften_cond_mask(raw) is raw)
        finally:
            self.cond_erode_kernel, self.cond_blur_kernel, self.cond_soften = e_k, b_k, sw
        print('-' * 78)
        if fails:
            raise AssertionError(f'v10 AUDIT FAILED: {fails}')
        print('v10 AUDIT PASSED — theory verified on real pipeline tensors.')


    # ------------------------------------------------------------------
    # Step-0 smoke audit (spec §8: verify the config actually took effect)
    # ------------------------------------------------------------------

    def _smoke_audit(self, metrics, peak_gb):
        print('\n' + '=' * 78)
        print('[SMOKE AUDIT — Step 0 checks (spec §8)]')
        failures = []

        def chk_nonzero(name, value):
            ok = value is not None and value > 0.0
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:26s} = {value}")
            if not ok:
                failures.append(name)

        for k in ('real_p1_eps', 'real_p2_eps', 'synth_eps'):
            if f'real_{k}' in metrics or k in metrics:
                v = metrics.get(f'real_{k}', metrics.get(k, 0.0))
                chk_nonzero(k, v)

        low_t_terms = ['real_p2_pixel_l1', 'real_p2_pixel_pl', 'real_p2_pixel_id',
                       'real_p2_latent', 'synth_pixel_l1', 'synth_pixel_pl',
                       'synth_pixel_id', 'synth_latent']
        # v13: config-aware gating — a term whose WEIGHT the recipe
        # deliberately zeroes (L1/PL red line) must NOT be demanded nonzero.
        _w = self.opts.losses
        _term_weight = {
            'pixel_l1': float(_w.pixel.l1_weight),
            'pixel_pl': float(_w.pixel.resnet_pl_weight),
            'pixel_id': float(_w.pixel.id_weight),
            'latent': float(_w.latent.weight),
        }
        forced_low_t = (self.fix_timestep is not None
                        and int(self.fix_timestep) < self.pixel_max_t)
        for k in low_t_terms:
            v = metrics.get(k, 0.0)
            parts = k.split('_')
            if parts and parts[0] in ('real', 'synth'):
                parts = parts[1:]
            if parts and parts[0] in ('p1', 'p2'):
                parts = parts[1:]
            wtk = '_'.join(parts[:2]) if parts[:1] == ['pixel'] else 'latent'
            enabled = _term_weight.get(wtk, 1.0) > 1e-5
            # v16: photo-ref L1 is REAL-pass2-only — synth pixel terms stay
            # disabled by the paint-oil red line unless apply_to_synth=true
            if (enabled and wtk == 'pixel_l1' and k.startswith('synth')
                    and not self.pixel_apply_synth):
                enabled = False
            if forced_low_t and enabled:
                chk_nonzero(k, v)
            elif not enabled:
                print(f"  [info] {k:26s} = {v:.6f} (weight=0 by recipe — red line, "
                      f"correctly inactive)")
            else:
                print(f"  [info] {k:26s} = {v:.6f} "
                      f"(stochastic gate; rerun with smoke.fix_timestep=100 to force)")

        if self.wplus_mode == 'frozen':
            grads = [p.grad for p in self._wplus_params_for_audit()
                     if p.grad is not None]
            grad_norm = sum(float(g.abs().sum()) for g in grads)
            ok = len(grads) == 0 or grad_norm == 0.0
            print(f"  [{'PASS' if ok else 'FAIL'}] W+ frozen: grad tensors={len(grads)}, "
                  f"total |grad|={grad_norm:.3e} (must be 0/None)")
            if not ok:
                failures.append('wplus_frozen_grads')

        if self.local_window_k:
            procs = [p for p in self.denoising_unet.attn_processors.values()
                     if isinstance(p, ReferenceAttentionProcessor)
                     and p.reference_local_scale is not None]
            n_active = sum(1 for p in procs if p.local_window == self.local_window_k)
            gates = [float(p.reference_local_scale) for p in procs]
            ok_routing = all(p.reference_local_scale.requires_grad for p in procs)
            ok_warm = all(abs(g - math.atanh(self.local_gate_init)) < 1e-4 for g in gates)
            print(f"  [{'PASS' if n_active == len(procs) else 'FAIL'}] v11 local pathway "
                  f"installed on {n_active}/{len(procs)} attn1 procs "
                  f"(window={self.local_window_k})")
            print(f"  [{'PASS' if ok_routing else 'FAIL'}] v11 local gate requires_grad")
            print(f"  [{'PASS' if ok_warm else 'FAIL'}] v11 local gate warm-start "
                  f"atanh({self.local_gate_init}) -> {gates[0]:.4f} (no longer the "
                  f"dead 0.000 param)")
            if n_active != len(procs):
                failures.append('v11_local_install')
            if not ok_routing:
                failures.append('v11_local_grad')
            if not ok_warm:
                failures.append('v11_local_warmstart')

        # v13 (2026-08-29): the restored ORIG recipe actually took effect —
        # identity nets really loaded, low-pass anchor really off.
        idt_on = (float(self.opts.losses.latent.weight) > 1e-5
                  or float(self.opts.losses.pixel.id_weight) > 1e-5)
        if idt_on:
            loaded = getattr(self, 'gan', None) is not None
            print(f"  [{'PASS' if loaded else 'FAIL'}] v13 identity layer: "
                  f"WplusNet loaded={loaded} "
                  f"(latent={float(self.opts.losses.latent.weight)}, "
                  f"id={float(self.opts.losses.pixel.id_weight)})")
            if not loaded:
                failures.append('v13_identity_net')
        else:
            print('  [info] v13 identity layer OFF (latent=0, id=0)')
        lowanchor = float(self.blind_struct_weight)
        print(f"  [{'PASS' if lowanchor <= 0 else 'INFO'}] v13: dual-band low-anchor "
              f"OFF (blind_struct_weight={lowanchor}) -> weighted eps path "
              f"visible@{float(self.opts.losses.real_pass1.visible_weight)} + "
              f"hole@{float(self.opts.losses.real_pass1.hole_weight)} (orig spec §5.3)")

        # v14: GAN/FM third layer really active (config-aware)
        if self._invsr():
            for k in ('real_p2_ldif', 'synth_x0_ldif'):
                v = metrics.get(k, 0.0)
                if forced_low_t:
                    chk_nonzero(k, v)
                else:
                    print(f'  [info] {k:26s} = {v:.6f} '
                          f'(stochastic t<={self.x0_t_max} domain; fix_timestep=100 forces it)')
            if forced_low_t:
                chk_nonzero('discr_adv (invsr D step)', metrics.get('discr_adv', 0.0))
            loaded_d = self.discriminator is not None
            n_d = sum(p.numel() for p in self.discriminator.parameters()) \
                if loaded_d else 0
            print(f"  [{'PASS' if loaded_d else 'FAIL'}] v18.3 InvSR latent UNet-D "
                  f"loaded ({n_d:,} params), t_max={self.x0_t_max}, "
                  f"warmup={self.x0_dis_warmup}")
            if not loaded_d:
                failures.append('invsr_discriminator')
        elif self.use_gan_fm:
            # v17-prime: under x0_enable the pass1 output has NO distribution
            # supervision BY DESIGN (novel view has no photo reference);
            # audit p2 + synth only.
            _fm_keys = ('real_p2_gen_fm',) if self.x0_enable \
                else ('real_p1_gen_fm', 'real_p2_gen_fm')
            for k in _fm_keys:
                v = metrics.get(k, 0.0)
                if forced_low_t:
                    chk_nonzero(k, v)
                else:
                    print(f"  [info] {k:26s} = {v:.6f} (t<200 gate; "
                          f"fix_timestep=100 forces it)")
            if forced_low_t:
                dv = metrics.get('discr_adv', 0.0)
                chk_nonzero('discr_adv (D step ran)', dv)
            loaded_d = self.discriminator is not None
            n_d = sum(p.numel() for p in self.discriminator.parameters()) \
                if loaded_d else 0
            print(f"  [{'PASS' if loaded_d else 'FAIL'}] v14 discriminator "
                  f"loaded ({n_d:,} params), fm_weight={self.fm_weight}, "
                  f"g_adv_weight={self.adv_weight}")
            if not loaded_d:
                failures.append('v14_discriminator')

        # v15: score preservation really active (fires at ALL timesteps —
        # no low-t gate; chk unconditionally when enabled)
        if self.preserve_weight > 0:
            chk_nonzero('real_p1_preserve', metrics.get('real_p1_preserve', 0.0))
            print(f"  [info] v15 preserve ACTIVE: weight={self.preserve_weight}, "
                  f"teacher_wplus={self.teacher_wplus}, all-t band=HIGH")

        # v16: pass2 photo-ref pixel L1 — magnitude guard against the Stage-B
        # divergence family (lesson #14: adversarial/perceptual on the x0
        # single-step decode diverged; plain L1 at low weight under x0_clip=3.0
        # must stay an order below the eps terms).
        if float(self.opts.losses.pixel.l1_weight) > 1e-5:
            l1v = float(metrics.get('real_p2_pixel_l1', 0.0))
            hit = float(metrics.get('real_p2_low_t_hits', 0.0))
            if hit > 0 and l1v > 0:
                ok_mag = l1v < 1.0
                print(f"  [{'PASS' if ok_mag else 'FAIL'}] v16 photo-ref L1 "
                      f"magnitude < 1.0 (divergence guard) = {l1v:.4f}")
                if not ok_mag:
                    failures.append('v16_pixel_l1_magnitude')
                print(f"  [info] v16 photo-ref ACTIVE: l1_weight="
                      f"{float(self.opts.losses.pixel.l1_weight)}, pl=0, id=0, "
                      f"latent=0 (single variable vs v12)")
            elif forced_low_t:
                failures.append('v16_photo_ref_not_fired')

        # v18: budget line realigned to the LOCAL hardware (ta13, RTX 3090,
        # 24 GiB). The old <22 line was a remote-era constant; measured peaks
        # on this machine: v17-prime 40-step 22.67 GiB, v18w1 arms 22.65-22.69
        # GiB (patch-decode x0 path = same decode size as the old resize
        # path). 23.5 leaves ~1.8 GiB allocator headroom below the 24G card
        # limit — this is also the PREREG_V18W1 §3 R-c stop line.
        ok_mem = peak_gb < 23.5
        print(f"  [{'PASS' if ok_mem else 'FAIL'}] peak GPU memory = {peak_gb:.2f} GiB (<23.5 on 24G card)")
        if not ok_mem:
            failures.append('peak_memory')
        print('=' * 78)
        if failures:
            raise AssertionError(f'SMOKE AUDIT FAILED: {failures}')
        print('SMOKE AUDIT PASSED — all Step-0 criteria met.')

    @torch.no_grad()
    def _v13_identity_audit(self, x, codes):
        """v13 (2026-08-29): verify the restored identity layer's DATA FLOW is
        REAL, not stubbed. The WplusNet GOAE encoder must map this person's
        PHOTO back near HIS OWN inversion codes (encoder/codes domain
        coherence). If enc(photo) were no closer to the true codes than to a
        foreign code, the latent closure would supervise against a meaningless
        target and silently do nothing (or harm). Fails loudly."""
        print('\n' + '-' * 78)
        print('[v13 IDENTITY-LAYER AUDIT — latent closure data flow is REAL]')
        photo_256 = F.adaptive_avg_pool2d(x.detach(), (256, 256))
        w_self = self.gan.encoder_forward(photo_256)
        # a foreign identity: channel-flipped codes (same marginal distribution,
        # valid W+ point, encodes a DIFFERENT person) — works at bs=1
        codes_foreign = codes.detach().flip(-1)
        same = float(F.mse_loss(w_self, codes.detach()))
        foreign = float(F.mse_loss(w_self, codes_foreign))
        ratio = foreign / (same + 1e-12)
        dim_ok = tuple(w_self.shape) == tuple(codes.shape)
        ok = ratio > 1.5 and dim_ok
        print(f"  [{'PASS' if ok else 'FAIL'}] enc(photo)->own codes MSE {same:.4f} vs "
              f"foreign {foreign:.4f} (ratio {ratio:.2f} > 1.5); "
              f"codes{tuple(codes.shape)} w_pred{tuple(w_self.shape)}")
        # decoder really parked on CPU (memory contract, §5.2 port notes)
        dec_cpu = all(p.device.type == 'cpu'
                      for p in self.gan.decoder.parameters())
        print(f"  [{'PASS' if dec_cpu else 'FAIL'}] gan.decoder parked on CPU "
              f"(encoder-only path)")
        if not ok or not dec_cpu:
            raise AssertionError(
                'v13 IDENTITY AUDIT FAILED: encoder/codes domain mismatch '
                f'(ratio={ratio:.3f}, dim_ok={dim_ok}) or decoder on GPU')
        print('v13 IDENTITY AUDIT PASSED — the closure target is meaningful.')

    def _v15_audit(self, out1, offset, hp, w_blind, loss_pres):
        """v15 real-data audit: the score-preservation theory verified on
        actual pipeline tensors BEFORE any long run (v14 lesson: never trust
        wiring that a smoke has not SEEN fire)."""
        print('\n' + '-' * 78)
        print('[v15 SCORE-PRESERVATION AUDIT — prior takeover on real data]')
        fails = []

        def chk(name, ok, extra=''):
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:46s} {extra}")
            if not ok:
                fails.append(name)

        d = float(offset.abs().mean())
        chk('teacher/student offset exists (0.0005 < mean|off| < 10)',
            0.0005 < d < 10.0, f'{d:.4f}')
        # band separation: DC must vanish in HP; checkerboard must survive
        dc_test = (offset + 0.3) - self._lowpass_eps(offset + 0.3)
        chk('HP kills DC (low band untouched by the loss)',
            float((dc_test - hp).abs().max()) < 1e-3,
            f'maxdiff={float((dc_test - hp).abs().max()):.2e}')
        eb = offset.clone()
        eb[..., ::2, ::2] += .5; eb[..., 1::2, 1::2] += .5
        eb[..., ::2, 1::2] -= .5; eb[..., 1::2, ::2] -= .5
        hp_eb = eb - self._lowpass_eps(eb)
        chk('HP keeps checkerboard (texture band IS protected)',
            float((hp_eb - hp).abs().mean()) > 0.05,
            f'{float((hp_eb - hp).abs().mean()):.4f}')
        chk('preserve loss > 0', float(loss_pres) > 0, f'{float(loss_pres):.6f}')
        chk('blind field mass > 0', float(w_blind.sum()) > 0,
            f'{float(w_blind.sum()):.1f}')
        chk('teacher really is no-control (eps_teacher detached)',
            not out1['eps_teacher'].requires_grad)
        print('-' * 78)
        if fails:
            raise AssertionError(f'v15 AUDIT FAILED: {fails}')
        print('v15 AUDIT PASSED — prior-takeover theory verified on real tensors.')

    def _v12_audit(self, out1, w_full, w_low):
        """v12 REAL-DATA audit (user review gate 2026-08-28): verify the
        three-partition theory on actual pipeline tensors BEFORE any long run.
        Fails loudly so a broken v12 can never reach a 50K training."""
        print('\n' + '-' * 78)
        print('[v12 REAL-DATA AUDIT — evidence-graded dual-band supervision]')
        fails = []
        def chk(name, ok, extra=''):
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:46s} {extra}")
            if not ok:
                fails.append(name)
        e = (out1['eps_pred'].detach() - out1['noise'].detach()).float()
        sq_full, sq_low = e.pow(2), self._lowpass_eps(e).pow(2)
        w_full_d, w_low_d = w_full.detach().float(), w_low.detach().float()

        # 1. decomposition: both bands actually contribute
        part_full = float((sq_full * w_full_d).sum() / ((w_full_d.sum() * e.shape[1]) + 1e-6))
        part_low = float((sq_low * w_low_d).sum() / ((w_low_d.sum() * e.shape[1]) + 1e-6))
        chk('photo FULL-band contribution > 0', part_full > 0, f'{part_full:.5f}')
        chk('blind LOW-band contribution > 0', part_low > 0, f'{part_low:.5f}')
        chk('blind field non-empty (w_low mass > 0)',
            float(w_low_d.sum()) > 0,
            f'w_low mass={float(w_low_d.sum()):.1f} frac={float((w_low_d > 0.01).float().mean()):.3f}')

        # 2. band separation ON THE REAL ERROR FIELD: the blind term must be
        #    insensitive to a ZERO-MEAN checkerboard (pure high-freq) error
        base = float(sq_low.mul(w_low_d).sum())
        eb = e.clone()
        eb[..., ::2, ::2] += 0.5
        eb[..., 1::2, 1::2] += 0.5
        eb[..., ::2, 1::2] -= 0.5          # +/-0.5: zero mean, no DC component
        eb[..., 1::2, ::2] -= 0.5
        pert_low = float(self._lowpass_eps(eb).pow(2).mul(w_low_d).sum())
        rel = (pert_low - base) / (base + 1e-9)
        chk('blind term kills checkerboard (high-freq) error: delta < 15%',
            abs(rel) < 0.15, f'rel_delta={rel:+.4f}')
        eb2 = e.clone(); eb2 += 0.3                              # DC (pure low-freq)
        pert2 = float(self._lowpass_eps(eb2).pow(2).mul(w_low_d).sum())
        chk('blind term keeps DC (low-freq) error', pert2 > base,
            f'{base:.1f} -> {pert2:.1f}')

        # 3. continuity of the two weight fields (v6 lesson: no binarized seams)
        for nm, w in (('w_full', w_full_d), ('w_low', w_low_d)):
            j = float((w[:, :, :, 1:] - w[:, :, :, :-1]).abs().max())
            chk(f'{nm} field continuous (max jump <= 0.6)', j <= 0.6, f'max_jump={j:.3f}')

        # 4. the anchor keeps FULL SHARP content in the blind region (no
        #    lowpassed image anywhere in the target chain — v3.2 lesson).
        #    anchor is logged via panels; here we verify through mask_latents:
        #    high-freq energy of the TARGET latents in blind positions must be
        #    nonzero (a lowpassed target would have ~0 HF energy).
        tl = out1['target_latents'].detach().float()
        blind_pos = (w_low_d > 0.5)
        if blind_pos.any():
            hf = (tl - self._lowpass_eps(tl)).pow(2)
            hf_blind = float(hf[blind_pos.expand_as(hf)].mean())
            chk('target HF energy in blind region > 1e-4 (target is SHARP, not lowpassed)',
                hf_blind > 1e-4, f'HF_energy={hf_blind:.5f}')
        else:
            print('  [info] no strong-blind positions in this batch')
        print('-' * 78)
        if fails:
            raise AssertionError(f'v12 AUDIT FAILED: {fails}')
        print('v12 AUDIT PASSED — three-partition theory verified on real tensors.')

    # ------------------------------------------------------------------
    # logging / checkpointing
    # ------------------------------------------------------------------

    def _save_panels(self, panels, tag):
        import torchvision.utils as vutils
        rows = []
        for name in sorted(panels.keys()):
            img = panels[name].cpu()
            if img.shape[0] > 1:
                img = img[:1]
            rows.append((name, img[0] if img.dim() == 4 else img))
        if not rows:
            return
        n = max(r[1].shape[-2] for r in rows)
        canvas = torch.cat([
            torch.nn.functional.interpolate(
                r[1].unsqueeze(0), size=(n, n), mode='bilinear',
                align_corners=False)[0] for r in rows], dim=2)
        path = os.path.join(self.save_train_dir,
                            f'{tag}_step{self.global_step:07d}.png')
        vutils.save_image(canvas, path)
        print(f"[images] {tag}: {' | '.join(r[0] for r in rows)} -> {path}")

    def __get_save_dict(self):
        processors = {
            name: proc.state_dict()
            for name, proc in self.denoising_unet.attn_processors.items()
            if isinstance(proc, ReferenceAttentionProcessor)
        }
        return {
            'global_step': self.global_step,
            'brushnet': self.brushnet.state_dict(),
            'w_mapper': self.w_mapper.state_dict() if self.w_mapper is not None else None,
            'attention_processors': processors,
            'optimizer': self.optimizer.state_dict(),
            'discriminator': self.discriminator.state_dict()
                if (self.use_gan_fm or self._invsr()) else None,
            'optimizer_d': self.optimizer_d.state_dict()
                if (self.use_gan_fm or self._invsr()) else None,
        }

    def checkpoint_me(self, metrics):
        path = os.path.join(self.checkpoint_dir, f'iteration_{self.global_step:07d}.pt')
        torch.save(self.__get_save_dict(), path)
        print(f"[ckpt] saved {path}")

    def resume(self):
        ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu',
                          weights_only=False)
        self.global_step = ckpt['global_step']
        self.brushnet.load_state_dict(ckpt['brushnet'], strict=True)
        if self.use_gan_fm and ckpt.get('discriminator') is not None:
            self.discriminator.load_state_dict(ckpt['discriminator'], strict=True)
            self.optimizer_d.load_state_dict(ckpt['optimizer_d'])
            print('[resume] v14 discriminator + optimizer_d restored')
        if self.w_mapper is not None and ckpt.get('w_mapper') is not None:
            self.w_mapper.load_state_dict(ckpt['w_mapper'], strict=True)
        for name, proc in self.denoising_unet.attn_processors.items():
            if name in ckpt['attention_processors']:
                # strict=False: checkpoints saved BEFORE the independent mirror
                # branch lack the *_extra keys; they keep their warm-start init
                # (== SD attention weights) and the gate stays at its tanh(0)=0
                # zero-perturbation start. Anything else missing is reported.
                missing, unexpected = proc.load_state_dict(
                    ckpt['attention_processors'][name], strict=False)
                unexpected = [k for k in unexpected if 'reference_extra' not in k]
                missing = [k for k in missing
                           if 'reference_extra' not in k and 'reference_mirror' not in k]
                if unexpected or [k for k in missing
                                  if 'reference_extra' not in k]:
                    print(f'[resume][WARN] {name}: unexpected={unexpected} '
                          f'missing_non_extra={missing}')
        # Optimizer state migration. v2 lesson (§8.15): DISCARDING the saved
        # AdamW state on a group-count mismatch (v2 resume) shocked the
        # 25K-step-trained BrushNet (un-adapted effective step size) and
        # produced the global S-warp — the 25999 crash happened while the
        # mirror gate was still ~0.003, so the injection could not be the
        # cause. Correct handling: migrate state BY GROUP NAME — existing
        # groups (brushnet / refnet_adapters) keep their moments; genuinely
        # new groups (mirror_*) start empty and AdamW lazy-initializes them.
        saved_opt = ckpt['optimizer']
        if len(saved_opt['param_groups']) == len(self.optimizer.param_groups):
            self.optimizer.load_state_dict(saved_opt)
        else:
            saved_state = saved_opt['state']
            migrated, fresh = 0, 0
            for cur_group in self.optimizer.param_groups:
                name = cur_group.get('name')
                match = next((g for g in saved_opt['param_groups']
                              if g.get('name') == name
                              and len(g['params']) == len(cur_group['params'])), None)
                if match is None:
                    fresh += len(cur_group['params'])
                    continue
                for cur_p, saved_idx in zip(cur_group['params'], match['params']):
                    st = saved_state.get(saved_idx)
                    if st is None:
                        fresh += 1
                        continue
                    self.optimizer.state[cur_p] = {
                        k: (v.to(cur_p.device) if torch.is_tensor(v) else v)
                        for k, v in st.items()
                    }
                    migrated += 1
            print(f'[resume] optimizer state MIGRATED by group name: '
                  f'{migrated} params kept moments, {fresh} fresh (lazy init). '
                  f'Groups now: {[g.get("name") for g in self.optimizer.param_groups]}')
        print(f"[resume] step {self.global_step} from {self.opts.checkpoint_path}")


    # ------------------------------------------------------------------
    # validation (inference = pass 1 ONLY; training/inference mirror, spec §1)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_novel(self, condition_img, mask, codes, ref_images, ref_tags,
                      num_steps=50, seed=42, mask_cond=None, anchor_img=None,
                      t_start=None, hf_img=None):
        """v18.6: t_start>0 (default self.infer_t_start) switches to SDEdit /
        InvSR partial-noise inference — start from the NOISED ANCHOR at
        t_start and run a manual DDIM (eta=0) ladder down to 0. The anchor is
        exactly what pass1 eps-training noises (train/inference contract);
        the trajectory never enters the prior-dominated high-t regime
        (user-verified domain fix, 2026-09-13). t_start=0 = legacy chain.
        v18.8: hf_img adds the AnyDoor HF channel to BrushNet conditioning."""
        if t_start is None:
            t_start = self.infer_t_start
        cond_latents = self.vae.encode(condition_img * 2.0 - 1.0).latent_dist.mode() * 0.18215
        m_brush = mask if mask_cond is None else mask_cond   # v10: BrushNet channel softened
        mask_latents = F.interpolate(m_brush, size=cond_latents.shape[-2:], mode='nearest')
        if self.hf_enable and hf_img is not None:
            hf_lat = F.interpolate(hf_img, size=cond_latents.shape[-2:],
                                   mode='bilinear', align_corners=False)
            conditioning = torch.cat([cond_latents, mask_latents, hf_lat], dim=1)
        else:
            conditioning = torch.cat([cond_latents, mask_latents], dim=1)
        bsz = cond_latents.shape[0]

        scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        scheduler.set_timesteps(num_steps, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        # latents mirror the IMAGE latent shape (4ch): BrushNet internally
        # concats [sample, brushnet_cond] before conv_in_condition, so a 5ch
        # init would corrupt the channel count (found by the valcheck run).
        noise = torch.randn(cond_latents.shape, generator=generator,
                            device=self.device)
        if int(t_start) > 0 and anchor_img is not None:
            anchor_lat = self.vae.encode(
                anchor_img * 2.0 - 1.0).latent_dist.mode() * 0.18215
            latents = scheduler.add_noise(
                anchor_lat, noise, torch.tensor([int(t_start)], device=self.device))
            ts_list = sorted(set(np.linspace(int(t_start), 0, num_steps)
                                 .round().astype(int).tolist()), reverse=True)
        else:
            latents = noise * scheduler.init_noise_sigma
            ts_list = None

        wplus_features = self._wplus_tokens(codes, False)
        ref_feat, ref_feat_extra = self._extract_reference_features(ref_images, ref_tags)
        cross_attention_kwargs = {}
        if wplus_features is not None:
            cross_attention_kwargs['wplus_features'] = wplus_features
        if ref_feat is not None:
            cross_attention_kwargs['reference_features'] = ref_feat
            if ref_feat_extra is not None:
                cross_attention_kwargs['reference_features_extra'] = ref_feat_extra
        prompt = self.empty_prompt_embeds.expand(bsz, -1, -1)

        if ts_list is None:
            for t in scheduler.timesteps:
                down_res, mid_res, up_res = self.brushnet(
                    latents, t, encoder_hidden_states=prompt,
                    brushnet_cond=conditioning, return_dict=False)
                eps = self.denoising_unet(
                    sample=latents, timestep=t, encoder_hidden_states=prompt,
                    down_block_add_samples=[s for s in down_res],
                    mid_block_add_sample=mid_res,
                    up_block_add_samples=[s for s in up_res],
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False)[0]
                latents = scheduler.step(eps, t, latents).prev_sample
        else:
            # manual DDIM (eta=0) over the custom ladder — same update the
            # scheduler applies, with a step size matched to the t_start domain
            ac = scheduler.alphas_cumprod
            for i, t in enumerate(ts_list):
                tt = torch.tensor(t, device=self.device)
                down_res, mid_res, up_res = self.brushnet(
                    latents, tt, encoder_hidden_states=prompt,
                    brushnet_cond=conditioning, return_dict=False)
                eps = self.denoising_unet(
                    sample=latents, timestep=tt, encoder_hidden_states=prompt,
                    down_block_add_samples=[s for s in down_res],
                    mid_block_add_sample=mid_res,
                    up_block_add_samples=[s for s in up_res],
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False)[0]
                ab = ac[t]
                ab_prev = ac[ts_list[i + 1]] if i + 1 < len(ts_list) \
                    else torch.ones_like(ab)
                x0 = (latents - (1.0 - ab).sqrt() * eps) / ab.sqrt()
                latents = ab_prev.sqrt() * x0 + (1.0 - ab_prev).sqrt() * eps

        return (self.vae.decode(latents / 0.18215).sample / 2 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def validate(self):
        batch = next(iter(self.test_dataloader))
        b = self._parse_real_batch(batch)
        x, c, codes, depth = b['x'], b['c'], b['codes'], b['depth']
        c_novel, y_hat_novel, depth_novel = b['c_novel'], b['y_hat_novel'], b['depth_novel']

        # v6: SAME construction as training pass1 (train/inference contract
        # can never diverge — the Phase-2 haze lesson).
        nv = self._build_novel_view(
            x, depth, c, c_novel, y_hat_novel, depth_novel,
            x_mirror=b.get('x_mirror'), c_mirror=b.get('c_mirror'),
            depth_mirror=b.get('depth_mirror'))
        mask, cond, anchor = nv['mask'], nv['cond'], nv['anchor']

        ref_images, ref_tags = self._ref_inputs_real(y_hat_novel, b.get('x_mirror'))
        # v18.8: HF conditioning at inference = same as training pass1
        hf_val = None
        if self.hf_enable and b.get('x_mirror') is not None:
            with torch.no_grad():
                hf_mirror = self._compute_hf_map(b['x_mirror'])
                hf_warp, _, _ = self.warper.forward_warp(
                    img1=hf_mirror, depth1=b['depth_mirror'],
                    c1=b['c_mirror'], c2=c_novel)
                hf_val = hf_warp
        gen = self._sample_novel(cond, mask, codes, ref_images, ref_tags,
                                 mask_cond=nv.get('mask_cond'),
                                 anchor_img=anchor, hf_img=hf_val)

        l1 = (gen - anchor).abs().mean(dim=1, keepdim=True)
        metrics = {
            'val_novel_full': float(l1.mean()),
            'val_novel_hole': float((l1 * mask).sum() / (mask.sum() * 1.0 + 1e-6)),
            'val_novel_visible': float((l1 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-6)),
        }
        for k, v in metrics.items():
            self.logger.add_scalar(k, v, self.global_step)
        print(f"[val step {self.global_step}] " + ' '.join(f'{k}={v:.4f}' for k, v in metrics.items()))

        import torchvision.utils as vutils
        panels = [x, y_hat_novel, cond, mask.expand_as(x), anchor, gen]
        canvas = torch.cat([p[:1][0].cpu() if p.dim() == 4 else p[:1].cpu() for p in panels], dim=2)
        vutils.save_image(canvas, os.path.join(
            self.save_val_dir, f'val_step{self.global_step:07d}.png'))











