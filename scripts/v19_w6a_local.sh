#!/bin/bash
# =============================================================================
# v19 W6-A LOCAL (single RTX 4090, this machine): W3-P recipe + anchor_median=5
# + real LPIPS + adv×1.0. Local adaptation of the server's scripts/v19_w6a.sh:
#   - ROOT -> this machine's repo path
#   - GPU default 0 (single GPU box)
#   - MAX_STEPS default 100000 (user request; server ran 300K)
#   - dropped zz_cuda_home.sh (absent locally; torch 2.8.0+cu128 ships runtime)
# Recipe flags are IDENTICAL to scripts/v19_w6a.sh (v19 latest, W6-A arm).
# Log: appended live to train_logs/v19_w6a_100k.log (PYTHONUNBUFFERED=1).
# Usage: bash scripts/v19_w6a_local.sh          (fresh, 100K)
#        bash scripts/v19_w6a_local.sh resume   (resume from latest ckpt)
# =============================================================================
ARM=w6a
GPU=${GPU:-0}
ROOT=/data/xzy/warpgan20260803/20260803/warpgan_orig/WarpGAN-main
EXP=$ROOT/experiments/train_inpainting_diffusion/v19_${ARM}_100k
LOG=$ROOT/train_logs/v19_${ARM}_100k.log
mkdir -p $ROOT/train_logs
cd $ROOT || exit 1
export CONDA_PREFIX=/home/xzy/miniconda3/envs/warpgan
export PATH=$CONDA_PREFIX/bin:$PATH
EXTRA=""
if [ -n "$1" ] && [ "$1" = "resume" ]; then
  CKPT=$(ls -1 $ROOT/experiments/train_inpainting_diffusion/*_v19_${ARM}_100k/checkpoints/*.pt 2>/dev/null | sort | tail -1)
  if [ -n "$CKPT" ]; then EXTRA="checkpoint_path='$CKPT'"; echo "[v19-$ARM] resume from $CKPT at $(date)" >> "$LOG"; fi
fi
exec env CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
  /home/xzy/miniconda3/envs/warpgan/bin/python scripts/train_inpainting_diffusion.py \
  exp_dir=$EXP \
  max_steps=${MAX_STEPS:-100000} $EXTRA \
  wplus.mode=s1_mapper wplus.init_from_stage_c=False \
  losses.x0.enable=True losses.x0.mode=invsr \
  losses.x0.t_max=300 losses.x0.w_ldif=1.0 losses.x0.w_lfm=2.0 losses.x0.w_ldis=1.0 \
  losses.x0.dis_warmup=3000 losses.x0.latent_bound=10.0 \
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