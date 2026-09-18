#!/bin/bash
# =============================================================================
# v19 W7 LOCAL (single RTX 4090D, this machine): pass1 hole photo-statistics D.
# Local adaptation of the server's scripts/v19_w7_holed.sh — recipe flags are
# IDENTICAL; only machine plumbing differs:
#   - ROOT -> this machine's repo path
#   - GPU default 0 (single GPU box)
#   - log name v19_w7_holed_LOCAL.log (the server's v19_w7_holed.log is a
#     tracked project record and must not be mixed with this arm's output)
#   - dropped zz_cuda_home.sh (absent locally; torch 2.8.0+cu128 ships runtime)
#
# Lineage note (differs from server by necessity): server W7 resumes from
# W6-B@80K (its GPU1 arm); this machine's only local base is W6-A@150K
# (iteration_0149999.pt, stopped 2026-09-18 at step 150800). Both arms are
# W6-family (user calibration: arms equivalent); each is its own control.
#
# Recipe (identical to server): W6-B recipe + hole_d overrides:
#   fake = pass1 x0_hat masked to the hole (soft 13px latent mask, v12 recipe)
#   real = the SAME mask on clean PHOTO latents (.mode() encode)
#   G side adv-only (weight 0.1), D-only warmup 2000 steps from first pair.
#   w_ldis=0.1 (InvSR dose), t_max=300, anchor_median=5, real LPIPS on.
# Usage: bash scripts/v19_w7_holed_local.sh          (fresh from copied ckpt)
#        bash scripts/v19_w7_holed_local.sh resume   (resume from latest ckpt)
# =============================================================================
ARM=w7_holed
GPU=${GPU:-0}
ROOT=/data/xzy/warpgan20260803/20260803/warpgan_orig/WarpGAN-main
EXP=$ROOT/experiments/train_inpainting_diffusion/v19_${ARM}_300k
LOG=$ROOT/train_logs/v19_${ARM}_LOCAL.log
mkdir -p $ROOT/train_logs
cd $ROOT || exit 1
export CONDA_PREFIX=/home/xzy/miniconda3/envs/warpgan
export PATH=$CONDA_PREFIX/bin:$PATH
EXTRA=""
if [ -n "$1" ] && [ "$1" = "resume" ]; then
  if [ -n "$RESUME_CKPT" ]; then
    CKPT="$RESUME_CKPT"
  else
    CKPT=$(ls -1 $ROOT/experiments/train_inpainting_diffusion/*_v19_${ARM}_300k/checkpoints/*.pt 2>/dev/null | sort | tail -1)
  fi
  if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
    EXTRA="checkpoint_path='$CKPT'"
    echo "[v19-$ARM-local] resume from $CKPT at $(date)" >> "$LOG"
  else
    echo "[v19-$ARM-local] ERROR: resume requested but no checkpoint found (RESUME_CKPT unset, glob empty). ABORT — refusing silent fresh start." >&2
    exit 1
  fi
fi
exec env CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
  /home/xzy/miniconda3/envs/warpgan/bin/python scripts/train_inpainting_diffusion.py \
  exp_dir=$EXP \
  max_steps=${MAX_STEPS:-300000} $EXTRA \
  wplus.mode=s1_mapper wplus.init_from_stage_c=False \
  losses.x0.enable=True losses.x0.mode=invsr \
  losses.x0.t_max=300 losses.x0.w_ldif=1.0 losses.x0.w_lfm=2.0 losses.x0.w_ldis=0.1 \
  losses.x0.dis_warmup=3000 losses.x0.latent_bound=10.0 \
  losses.x0.hole_d.enable=True losses.x0.hole_d.weight=0.1 losses.x0.hole_d.warmup=2000 \
  losses.x0.llpips_enable=True losses.x0.llpips_ckpt=./weights/vgg16_sdturbo_lpips.pth \
  losses.feature_matching.weight=0 losses.adversarial.weight=0 \
  losses.real_pass1.forward=True \
  losses.real_pass1.mirror_anchor=True losses.real_pass1.visible_weight=1.0 \
  losses.real_pass1.hole_weight=0 losses.real_pass1.blind_struct_weight=0 \
  losses.real_pass1.anchor_median=5 \
  losses.synth_texture=full \
  losses.pixel.l1_weight=0 losses.pixel.resnet_pl_weight=0 losses.pixel.id_weight=0 \
  losses.latent.weight=0 losses.preserve.weight=0 \
  reference.use_mirror=True reference.real_primary=mirror reference.local_window_k=0 \
  warp.cond_erode_kernel=3 warp.cond_gaussian_blur_kernel=21 \
  smoke.check=True log.val_interval=2000 log.save_interval=10000 log.image_interval=500 \
  >> "$LOG" 2>&1
