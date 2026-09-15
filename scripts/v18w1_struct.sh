#!/bin/bash
# =============================================================================
# v18-wave1 STRUCT arm (GPU 3): FULL recipe + real-batch pass1 DISABLED.
# The orig coach NEVER supervised pred_novel on real data (the anchor's blind
# region is fabricated render evidence — the suspected oil-paint/sandpaper
# saboteur). Blind-region content transfers from pass2 (photo-supervised,
# now with distribution supervision) + synth (paired novel views). The mirror
# PHOTO rides pass2's bank (orig kept its mirror channel in the supervised
# pass, coach L255-266). Inference (pass1 sampling) is UNCHANGED.
# =============================================================================
ARM=struct
GPU=${GPU:-3}
ROOT=/home/xzy/warpgan-diffusiong-inpainting/warpgan-diffusion-inpainting-main
EXP=$ROOT/experiments/train_inpainting_diffusion/v18w1_${ARM}_50k
LOG=$ROOT/train_logs/v18w1_${ARM}.log
mkdir -p $ROOT/train_logs
cd $ROOT || exit 1
# Replicate `conda activate warpgan` under nohup (see v18w1_ctrl.sh note).
export CONDA_PREFIX=/home/xzy/miniconda3/envs/warpgan
export PATH=$CONDA_PREFIX/bin:$PATH
. $CONDA_PREFIX/etc/conda/activate.d/zz_cuda_home.sh
EXTRA=""
if [ -n "$1" ] && [ "$1" = "resume" ]; then
  # fresh runs get a [timestamp]_ prefix (addtime2path) — glob it
  CKPT=$(ls -1 $ROOT/experiments/train_inpainting_diffusion/*_v18w1_${ARM}_50k/checkpoints/*.pt 2>/dev/null | sort | tail -1)
  if [ -n "$CKPT" ]; then EXTRA="checkpoint_path='$CKPT'"; echo "[v18w1-$ARM] resume from $CKPT at $(date)" >> "$LOG"; fi
fi
exec env CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
  /home/xzy/miniconda3/envs/warpgan/bin/python scripts/train_inpainting_diffusion.py \
  exp_dir=$EXP \
  max_steps=50000 $EXTRA \
  wplus.mode=s1_mapper wplus.init_from_stage_c=False \
  losses.x0.enable=True losses.x0.l1_weight=1.0 \
  losses.feature_matching.weight=100 losses.adversarial.weight=10 \
  losses.real_pass1.forward=False \
  losses.real_pass1.mirror_anchor=True losses.real_pass1.visible_weight=1.0 \
  losses.real_pass1.hole_weight=0 losses.real_pass1.blind_struct_weight=0 \
  losses.synth_texture=full \
  losses.pixel.l1_weight=0 losses.pixel.resnet_pl_weight=0 losses.pixel.id_weight=0 \
  losses.latent.weight=0 losses.preserve.weight=0 \
  reference.use_mirror=True reference.real_primary=mirror reference.local_window_k=0 \
  warp.cond_erode_kernel=3 warp.cond_gaussian_blur_kernel=21 \
  smoke.check=True log.val_interval=2000 log.save_interval=10000 log.image_interval=500 \
  >> "$LOG" 2>&1
