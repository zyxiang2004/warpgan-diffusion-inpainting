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

from models.referencenet.unet_2d_condition import UNet2DConditionModel as ReferenceNet
from models.mapper.w_proj import WProjModel
from models.referencenet.attention_processor import ReferenceAttentionProcessor
from models.saicinpainting.utils import set_requires_grad
from models.saicinpainting.training.losses.perceptual import ResNetPL
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
        self.ref_gate_init = float(opts.reference.gate_init)
        self.mirror_anchor = bool(getattr(opts.losses.real_pass1, 'mirror_anchor', False))

        self._print_effective_config()

        # ---------------- frozen SD base ----------------
        sd_path = opts.paths.sd
        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder='vae').to(self.device).eval()
        set_requires_grad(self.vae, False)
        # pixel losses backprop THROUGH the VAE decoder -> checkpoint it
        self.vae.enable_gradient_checkpointing()
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
        self.mirror_branch_params = []  # independent mirror K/V (lr same as brushnet)
        self.mirror_gate_params = []    # independent mirror gates (high lr: scalar)
        self._install_attention_processors()

        # ---------------- empty prompt (W+ rides attn2 in parallel with it) ----------------
        tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder='tokenizer')
        text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder='text_encoder').to(self.device).eval()
        text_inputs = tokenizer('', padding='max_length', max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors='pt')
        with torch.no_grad():
            self.empty_prompt_embeds = text_encoder(text_inputs.input_ids.to(self.device))[0]
        del tokenizer, text_encoder
        torch.cuda.empty_cache()

        # ---------------- ReferenceNet (frozen feature extractor) ----------------
        self.reference_net = ReferenceNet.from_pretrained(sd_path, subfolder='unet').to(self.device).eval()
        set_requires_grad(self.reference_net, False)


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
            else:  # attn1 -> ReferenceNet branch
                set_requires_grad(proc, False)
                for proj in (proc.to_k_reference, proc.to_v_reference, proc.to_out_reference):
                    for p in proj.parameters():
                        p.requires_grad_(True)
                        self.refnet_adapter_params.append(p)
                proc.reference_scale.requires_grad_(True)
                self.refnet_adapter_params.append(proc.reference_scale)
                if self.use_mirror:
                    # INDEPENDENT mirror branch: own K/V + own gate, zero-init.
                    # v1 lesson (§8.13): a shared fixed gate made the injection
                    # unlearnable and destroyed the trained BrushNet. The gate
                    # now rides its own high-lr group so "how much mirror" is
                    # decided by the gradient (rises = needed; stays 0 = clean
                    # negative result for teacher ④'s parallel-injection form).
                    for proj in (proc.to_k_reference_extra, proc.to_v_reference_extra):
                        for p in proj.parameters():
                            p.requires_grad_(True)
                            self.mirror_branch_params.append(p)
                    proc.reference_extra_scale.requires_grad_(True)
                    self.mirror_gate_params.append(proc.reference_extra_scale)

        n_ref = sum(p.numel() for p in self.refnet_adapter_params)
        n_qkv = sum(p.numel() for p in self.wplus_qkv_params)
        n_map = sum(p.numel() for p in self.wplus_mapper_params)
        n_mir = sum(p.numel() for p in self.mirror_branch_params + self.mirror_gate_params)
        print(f"[params] RefNet adapters trainable: {n_ref:,} | "
              f"W+ QKV (s2) trainable: {n_qkv:,} | W+ mapper trainable: {n_map:,} | "
              f"mirror branch trainable (gate=0 init): {n_mir:,}")


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
        if self.mirror_branch_params:
            groups.append({'params': self.mirror_branch_params,
                           'lr': float(self.opts.brushnet.lr), 'name': 'mirror_branch'})
        if self.mirror_gate_params:
            # scalar gates need a high lr to move meaningfully within ~10K steps
            # (v1 failure: lr 5e-6 left adapters frozen). tanh is linear near 0,
            # so lr 1e-4 * 10K steps can reach saturation if the gradient wants it.
            groups.append({'params': self.mirror_gate_params,
                           'lr': float(self.opts.reference.mirror_gate_lr),
                           'name': 'mirror_gates'})
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
        print(f"  wplus: mode={self.wplus_mode} gate=fixed tanh({o.wplus.gate_value}) "
              f"stage_c_init={bool(o.wplus.init_from_stage_c)}")
        print(f"  reference: use={bool(o.reference.use)} use_mirror={self.use_mirror} "
              f"gate_init={self.ref_gate_init} lr={o.reference.lr}")
        print(f"  brushnet: init={self.brushnet_init} lr={o.brushnet.lr} 8bit={bool(o.brushnet.use_8bit_adam)} "
              f"grad_ckpt={bool(o.brushnet.gradient_checkpointing)}")
        print(f"  loss eps_weight       : {o.losses.eps_weight}")
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
                         ('wplus_stage_c', o.paths.wplus_stage_c if self.use_wplus else None),
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

    def _extract_reference_features(self, images, source_tags):
        """RefNet input = CLEAN COMPLETE images only (spec §4.3): inversion
        renders (y_hat / y_hat_novel / target_hat), optionally x_mirror in
        parallel. Broken warp images never enter the RefNet."""
        primary, extra = None, None
        for idx, (img, tag) in enumerate(zip(images, source_tags)):
            with torch.no_grad():
                latents = self.vae.encode(img * 2.0 - 1.0).latent_dist.mode() * 0.18215
                t_zero = torch.zeros(img.shape[0], device=self.device, dtype=torch.long)
                feat = self.reference_net.extract_features(
                    sample=latents, timestep=t_zero,
                    encoder_hidden_states=self.empty_prompt_embeds.expand(img.shape[0], -1, -1),
                )
            if self.smoke_check and self.global_step <= 2:
                print(f"[RefNet input check] source={tag} shape={tuple(img.shape)} "
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
                           ref_images, ref_tags, t_override=None, train_wplus=False):
        """One BrushNet+UNet epsilon forward. Returns everything the loss needs."""
        with torch.no_grad():
            target_latents = self._encode_image(target_img)
            cond_latents = self._encode_image(condition_img)
            mask_latents = F.interpolate(mask, size=target_latents.shape[-2:], mode='nearest')
            conditioning_latents = torch.cat([cond_latents, mask_latents], dim=1)  # 4+1 ch

        bsz = target_latents.shape[0]
        noise = torch.randn_like(target_latents)
        if t_override is not None:
            timesteps = torch.full((bsz,), int(t_override), device=self.device, dtype=torch.long)
        else:  # FULL range [0,1000) — low-t-only training drifts to haze (v3.1)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
                device=self.device).long()
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)

        wplus_features = self._wplus_tokens(codes, train_wplus)
        ref_feat, ref_feat_extra = self._extract_reference_features(ref_images, ref_tags)

        prompt_embeds = self.empty_prompt_embeds.expand(bsz, -1, -1)
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
            'mask_latents': mask_latents,
        }


    def _eps_mse(self, eps_pred, noise, weight_map=None):
        """epsilon-MSE. weight_map=None -> full frame (orig with_mask=False)."""
        sq = (eps_pred.float() - noise.float()).pow(2)
        if weight_map is None:
            return sq.mean()
        w = weight_map.float()
        return (sq * w).sum() / (w.sum() * sq.shape[1] + 1e-6)

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

    def _low_t_losses(self, out, target_img, codes):
        """t<max_t self-gated x0 single-step decode (FFC 1-step translation):
        pixel L1*10 + ResNetPL*30 + ID*0.5 vs target + latent closure *0.1."""
        metrics = {'pixel_l1': 0.0, 'pixel_pl': 0.0, 'pixel_id': 0.0,
                   'latent': 0.0, 'low_t_hits': 0.0}
        sel = out['timesteps'] < self.pixel_max_t
        metrics['low_t_hits'] = float(sel.float().mean())
        if not bool(sel.any()):
            zero = out['eps_pred'].new_zeros(())
            return zero, metrics

        x0 = self._predict_x0(out['eps_pred'], out['noisy_latents'], out['timesteps'])
        x0_sel = x0[sel].clamp(-self.x0_clip, self.x0_clip)  # keep the graph
        pred_img = self._decode_x0_to_pixel(x0_sel, self.pixel_res)
        target_sel = F.interpolate(
            target_img[sel], size=(self.pixel_res, self.pixel_res),
            mode='bilinear', align_corners=False)

        total = pred_img.new_zeros(())
        w = self.opts.losses.pixel
        if float(w.l1_weight) > 1e-5:
            l1 = F.l1_loss(pred_img, target_sel)
            total = total + l1 * float(w.l1_weight)
            metrics['pixel_l1'] = float(l1)
        if float(w.resnet_pl_weight) > 1e-5:
            pl = self.loss_resnet_pl(pred_img, target_sel)
            total = total + pl  # weight applied inside ResNetPL (orig behavior)
            metrics['pixel_pl'] = float(pl)
        if float(w.id_weight) > 1e-5:
            id_value, _, _ = self.loss_id(pred_img, target_sel, target_sel)
            total = total + id_value * float(w.id_weight)
            metrics['pixel_id'] = float(id_value)

        if float(self.opts.losses.latent.weight) > 1e-5:
            pred_256 = F.adaptive_avg_pool2d(pred_img, (256, 256))
            w_pred = self.gan.encoder_forward(pred_256)
            latent_value = F.mse_loss(codes[sel], w_pred)
            total = total + latent_value * float(self.opts.losses.latent.weight)
            metrics['latent'] = float(latent_value)

        out['pred_x0_img'] = pred_img.detach()  # for image logging
        return total, metrics


    # ------------------------------------------------------------------
    # batch forwards (mirror the orig coach's dual-pass structure)
    # ------------------------------------------------------------------

    def _ref_inputs(self, primary_img, mirror_img, primary_tag):
        images, tags = [primary_img], [primary_tag]
        if self.use_mirror:
            images.append(mirror_img)
            tags.append('x_mirror')
        return images, tags

    def _build_novel_view(self, x, depth, c, c_novel, y_hat_novel, depth_novel,
                          x_mirror=None, c_mirror=None, depth_mirror=None):
        """Shared novel-view construction (v6): condition and supervision follow
        the SAME geometry — visible = splat warp; hole = mirror-transferred REAL
        photo where the mirror camera sees it, EG3D render only in the truly
        blind region. Training pass1, validate() and the A/B scripts all call
        this ONE method, so the train/inference contract can never diverge
        again (the Phase-2 haze lesson).

        mirror off  -> cond hole = EG3D everywhere in hole (exact v4 behavior)
        mirror on   -> cond hole & anchor hole both = real|blind segmentation
        """
        warp_img, vis_mask, _ = self.warper.forward_warp(img1=x, depth1=depth, c1=c, c2=c_novel)
        mask = 1.0 - vis_mask
        with torch.no_grad():
            inv_warp, inv_valid = self.warper_ext.inverse_warp(
                img2=x, depth1=depth_novel, depth2=depth, c1=c_novel, c2=c)
            vis_eff = inv_valid.clamp(0, 1) * (1.0 - mask)
            hole_eff = torch.zeros_like(vis_eff)
            inv_warp_m = None
            if self.mirror_anchor and x_mirror is not None:
                inv_warp_m, inv_valid_m = self.warper_ext.inverse_warp(
                    img2=x_mirror, depth1=depth_novel, depth2=depth_mirror,
                    c1=c_novel, c2=c_mirror)
                hole_eff = inv_valid_m.clamp(0, 1) * mask
            blind = (1.0 - vis_eff - hole_eff).clamp(0, 1)

            # supervision anchor: real (source) | real (mirror) | EG3D (blind)
            anchor = inv_warp * vis_eff \
                + (inv_warp_m * hole_eff if inv_warp_m is not None else 0) \
                + y_hat_novel * blind
            # condition (v6): the SAME segmentation — the hole carries the
            # mirror-transferred REAL texture where available, EG3D only in the
            # blind region. The BrushNet copy-prior finally sees what the
            # target asks for. mask stays the raw splatting hole (full-frame
            # generation contract unchanged, spec principle 2).
            if inv_warp_m is not None:
                hole_fill = inv_warp_m * hole_eff + y_hat_novel * (mask - hole_eff).clamp(0, 1)
            else:
                hole_fill = y_hat_novel * mask
            cond = warp_img * (1.0 - mask) + hole_fill
        return dict(mask=mask, cond=cond, anchor=anchor, warp_img=warp_img,
                    vis_eff=vis_eff, hole_eff=hole_eff, blind=blind,
                    inv_warp_m=inv_warp_m)

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

        ref_images, ref_tags = self._ref_inputs(y_hat_novel, b.get('x_mirror'), 'y_hat_novel')
        out1 = self._diffusion_forward(
            target_img=anchor, condition_img=cond1, mask=mask, codes=codes,
            ref_images=ref_images, ref_tags=ref_tags,
            t_override=t_override, train_wplus=False)   # real update: W+ frozen
        p1 = self.opts.losses.real_pass1
        if self.mirror_anchor:
            # weight map follows REAL supervision coverage (validity mask):
            # real-photo pixels (source OR mirror transfer) -> full weight;
            # only the truly blind residual keeps the 0.1 EG3D anchor.
            eff_img = (vis_eff + hole_eff).clamp(0, 1)
            eff_latents = F.interpolate(eff_img, size=out1['mask_latents'].shape[-2:],
                                        mode='nearest')
            w_map = eff_latents * float(p1.visible_weight) \
                + (1.0 - eff_latents) * float(p1.hole_weight)
        else:
            w_map = (1.0 - out1['mask_latents']) * float(p1.visible_weight) \
                + out1['mask_latents'] * float(p1.hole_weight)
        loss_p1 = self._eps_mse(out1['eps_pred'], out1['noise'], w_map) \
            * float(self.opts.losses.eps_weight)
        metrics['p1_eps'] = float(loss_p1)
        metrics['p1_hole_frac'] = float(mask.mean())
        panels.update({'warp_img': nv['warp_img'].detach(), 'cond1': cond1.detach(),
                       'mask': mask.detach().expand_as(x), 'anchor': anchor.detach()})

        # ---- pass 2 (source): the ONLY real-photo supervision (warp_pred=False) ----
        warp_warp_img, vis_inv, _ = self.warper.forward_warp(
            img1=cond1, depth1=depth_novel, c1=c_novel, c2=c)
        mask_inv = 1.0 - vis_inv
        cond2 = warp_warp_img * (1.0 - mask_inv) + y_hat * mask_inv
        ref_images2, ref_tags2 = self._ref_inputs(y_hat, b.get('x_mirror'), 'y_hat')
        out2 = self._diffusion_forward(
            target_img=x, condition_img=cond2, mask=mask_inv, codes=codes,
            ref_images=ref_images2, ref_tags=ref_tags2,
            t_override=t_override, train_wplus=False)
        loss_p2_eps = self._eps_mse(out2['eps_pred'], out2['noise']) \
            * float(self.opts.losses.eps_weight)
        loss_p2_low, m_low = self._low_t_losses(out2, x, codes)
        metrics['p2_eps'] = float(loss_p2_eps)
        metrics.update({f'p2_{k}': v for k, v in m_low.items()})
        panels.update({'warp_warp_img': warp_warp_img.detach(),
                       'cond2': cond2.detach(),
                       'mask_inv': mask_inv.detach().expand_as(x)})
        if 'pred_x0_img' in out2:
            panels['p2_pred_x0'] = out2['pred_x0_img']

        loss_p2 = loss_p2_eps + loss_p2_low
        metrics['total'] = float(loss_p1 + loss_p2)
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
        cond = warp_img * (1.0 - mask) + y_hat_novel * mask
        ref_images, ref_tags = self._ref_inputs(y_hat_novel, b.get('x_mirror'), 'target_hat')
        out = self._diffusion_forward(
            target_img=target, condition_img=cond, mask=mask, codes=codes,
            ref_images=ref_images, ref_tags=ref_tags,
            t_override=t_override, train_wplus=True)    # synth update: W+ may train
        loss_eps = self._eps_mse(out['eps_pred'], out['noise']) \
            * float(self.opts.losses.eps_weight)
        loss_low, m_low = self._low_t_losses(out, target, codes)
        metrics['eps'] = float(loss_eps)
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
            loss_p1.backward()
            loss_p2.backward()
            self._zero_wplus_grads()
            if self.smoke_check and self.global_step <= 2 and self.mirror_gate_params:
                gvals = [float(p.grad.abs().sum()) for p in self.mirror_gate_params
                         if p.grad is not None]
                print(f'[smoke] mirror-gate |grad| per layer: nonzero={sum(1 for g in gvals if g > 0)}/'
                      f'{len(self.mirror_gate_params)}, max={max(gvals) if gvals else 0:.3e}')
            self.optimizer.step()

            metrics = {f'real_{k}': v for k, v in m_real.items()}

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

            # ---------------- logging ----------------
            if self.global_step % int(self.opts.log.board_interval) == 0:
                for k, v in metrics.items():
                    self.logger.add_scalar(k, v, self.global_step)
                if self.mirror_gate_params:
                    gates = [float(torch.tanh(p.detach()).mean())
                             for p in self.mirror_gate_params]
                    gm = sum(gates) / len(gates)
                    metrics['mirror_gate_mean'] = gm
                    self.logger.add_scalar('mirror_gate_mean', gm, self.global_step)
                    self.logger.add_scalar('mirror_gate_max', max(gates), self.global_step)
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
        forced_low_t = (self.fix_timestep is not None
                        and int(self.fix_timestep) < self.pixel_max_t)
        for k in low_t_terms:
            v = metrics.get(k, 0.0)
            if forced_low_t:
                chk_nonzero(k, v)
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

        ok_mem = peak_gb < 22.0
        print(f"  [{'PASS' if ok_mem else 'FAIL'}] peak GPU memory = {peak_gb:.2f} GiB (<22)")
        if not ok_mem:
            failures.append('peak_memory')
        print('=' * 78)
        if failures:
            raise AssertionError(f'SMOKE AUDIT FAILED: {failures}')
        print('SMOKE AUDIT PASSED — all Step-0 criteria met.')

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
                      num_steps=50, seed=42):
        cond_latents = self.vae.encode(condition_img * 2.0 - 1.0).latent_dist.mode() * 0.18215
        mask_latents = F.interpolate(mask, size=cond_latents.shape[-2:], mode='nearest')
        conditioning = torch.cat([cond_latents, mask_latents], dim=1)
        bsz = cond_latents.shape[0]

        scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        scheduler.set_timesteps(num_steps, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        # latents mirror the IMAGE latent shape (4ch): BrushNet internally
        # concats [sample, brushnet_cond] before conv_in_condition, so a 5ch
        # init would corrupt the channel count (found by the valcheck run).
        latents = torch.randn(cond_latents.shape, generator=generator,
                              device=self.device) * scheduler.init_noise_sigma

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

        ref_images, ref_tags = self._ref_inputs(y_hat_novel, b.get('x_mirror'), 'y_hat_novel')
        gen = self._sample_novel(cond, mask, codes, ref_images, ref_tags)

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











