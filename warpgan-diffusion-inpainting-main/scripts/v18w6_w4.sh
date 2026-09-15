#!/bin/bash
# =============================================================================
# v18.6 W4 arm (GPU 0, FROM SCRATCH, 100K): the full adjusted assembly per the
# 2026-09-13 mid-adjudication (user verdicts):
#   1. SDEdit-300 inference CONFIRMED (vintage-oil-paint reduced) -> deployed
#      as the official sampler (infer.t_start=300) AND the training domain is
#      matched to it (real_pass1.t_max=300 — the InvSR pattern: training
#      timesteps = the inference start domain).
#   2. prompt does nothing on empty-trained ckpts -> goes into TRAINING
#      (prompt.preset=invsr — their exact quality string; teacher ② also
#      intended W+ to align WITH text features).
#   3. latent-LPIPS (their MAIN loss term) replaces the dead FM substitute.
# Structure: W3-P (pass1 photo-evidence supervision won the mid-adjudication).
# Baselines still running: W3-D/W3-P (GPU1/2, to 100K).
# =============================================================================
ARM=w4
GPU=${GPU:-0}
ROOT=/home/xzy/warpgan-diffusiong-inpainting/warpgan-diffusion-inpainting-main
EXP=$ROOT/experiments/train_inpainting_diffusion/v18w6_${ARM}_100k
LOG=$ROOT/train_logs/v18w6_${ARM}.log
mkdir -p $ROOT/train_logs
cd $ROOT || exit 1
export CONDA_PREFIX=/home/xzy/miniconda3/envs/warpgan
export PATH=$CONDA_PREFIX/bin:$PATH
. $CONDA_PREFIX/etc/conda/activate.d/zz_cuda_home.sh
EXTRA=""
if [ -n "$1" ] && [ "$1" = "resume" ]; then
  CKPT=$(ls -1 $ROOT/experiments/train_inpainting_diffusion/*_v18w6_${ARM}_100k/checkpoints/*.pt 2>/dev/null | sort | tail -1)
  if [ -n "$CKPT" ]; then EXTRA="checkpoint_path='$CKPT'"; echo "[v18w6-$ARM] resume from $CKPT at $(date)" >> "$LOG"; fi
fi
exec env CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
  /home/xzy/miniconda3/envs/warpgan/bin/python scripts/train_inpainting_diffusion.py \
  exp_dir=$EXP \
  max_steps=${MAX_STEPS:-100000} $EXTRA \
  prompt.preset=invsr \
  infer.t_start=300 \
  losses.real_pass1.forward=True losses.real_pass1.t_max=300 \
  wplus.mode=s1_mapper wplus.init_from_stage_c=False \
  losses.x0.enable=True losses.x0.mode=invsr \
  losses.x0.t_max=300 losses.x0.w_ldif=1.0 losses.x0.w_lfm=2.0 losses.x0.w_ldis=0.1 \
  losses.x0.dis_warmup=3000 losses.x0.latent_bound=10.0 \
  losses.x0.llpips_enable=True losses.x0.llpips_ckpt=./weights/vgg16_sdturbo_lpips.pth \
  losses.feature_matching.weight=0 losses.adversarial.weight=0 \
  losses.real_pass1.mirror_anchor=True losses.real_pass1.visible_weight=1.0 \
  losses.real_pass1.hole_weight=0 losses.real_pass1.blind_struct_weight=0 \
  losses.synth_texture=full \
  losses.pixel.l1_weight=0 losses.pixel.resnet_pl_weight=0 losses.pixel.id_weight=0 \
  losses.latent.weight=0 losses.preserve.weight=0 \
  reference.use_mirror=True reference.real_primary=mirror reference.local_window_k=0 \
  warp.cond_erode_kernel=3 warp.cond_gaussian_blur_kernel=21 \
  smoke.check=True log.val_interval=2000 log.save_interval=10000 log.image_interval=500 \
  >> "$LOG" 2>&1
