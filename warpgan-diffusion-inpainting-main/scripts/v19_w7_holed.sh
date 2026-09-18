#!/bin/bash
# =============================================================================
# v19 W7 (GPU 2): W6-B recipe + pass1 hole photo-statistics D pair.
# The diffusion-port RESTORATION of the orig WarpGAN real-batch novel-branch
# D (static coach L415-420: fake=pred_novel, real=PHOTO x):
#   fake = pass1 x0_hat masked to the hole (soft 13px latent mask, v12 recipe)
#   real = the SAME mask on clean PHOTO latents (.mode() encode)
#   G side adv-only (weight 0.1), D-only warmup 2000 steps from first pair.
# Synth D pair (real=render) intentionally UNTOUCHED — that is orig behavior
# (static coach L554-558 'image': src_img; synthetic identities have no photo).
# RESUME from W6-B's 80K checkpoint: identical lineage -> the still-running
# W6-B (GPU 1) is this arm's direct control from the fork point on.
# Base: W6-B = W3-P recipe + anchor_median=5 + real LPIPS + adv x0.1.
# =============================================================================
ARM=w7_holed
GPU=${GPU:-2}
ROOT=/home/xzy/warpgan-diffusiong-inpainting/warpgan-diffusion-inpainting-main
EXP=$ROOT/experiments/train_inpainting_diffusion/v19_${ARM}_300k
LOG=$ROOT/train_logs/v19_${ARM}.log
mkdir -p $ROOT/train_logs
cd $ROOT || exit 1
export CONDA_PREFIX=/home/xzy/miniconda3/envs/warpgan
export PATH=$CONDA_PREFIX/bin:$PATH
. $CONDA_PREFIX/etc/conda/activate.d/zz_cuda_home.sh
EXTRA=""
if [ -n "$1" ] && [ "$1" = "resume" ]; then
  CKPT=$(ls -1 $EXP/checkpoints/*.pt 2>/dev/null | sort | tail -1)
  if [ -n "$CKPT" ]; then EXTRA="checkpoint_path='$CKPT'"; echo "[v19-$ARM] resume from $CKPT at $(date)" >> "$LOG"; fi
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
