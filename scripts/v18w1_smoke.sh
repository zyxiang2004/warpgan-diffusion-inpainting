#!/bin/bash
# =============================================================================
# v18-wave1 30-step SMOKE for one arm: usage: v18w1_smoke.sh <ctrl|full|struct> <gpu>
# Verifies: EFFECTIVE CONFIG, smoke audits, loss composition, W+ s1 mapper
# grads (synth-only), memory peak, STRUCT pass1 skip. No checkpoint save.
# =============================================================================
ARM=$1; GPU=${2:-1}
ROOT=/home/xzy/warpgan-diffusiong-inpainting/warpgan-diffusion-inpainting-main
cd $ROOT || exit 1
# Replicate `conda activate warpgan` under nohup (see v18w1_ctrl.sh note).
export CONDA_PREFIX=/home/xzy/miniconda3/envs/warpgan
export PATH=$CONDA_PREFIX/bin:$PATH
. $CONDA_PREFIX/etc/conda/activate.d/zz_cuda_home.sh
EXP=$ROOT/experiments/train_inpainting_diffusion/smoke_v18w1_${ARM}
LOG=$ROOT/train_logs/smoke_v18w1_${ARM}.log
mkdir -p $ROOT/train_logs
case $ARM in
  ctrl)   X0="losses.x0.enable=False losses.feature_matching.weight=0 losses.adversarial.weight=0 losses.real_pass1.hole_weight=1.0 losses.real_pass1.forward=True";;
  full)   X0="losses.x0.enable=True losses.x0.l1_weight=1.0 losses.feature_matching.weight=100 losses.adversarial.weight=10 losses.real_pass1.hole_weight=0 losses.real_pass1.forward=True";;
  struct) X0="losses.x0.enable=True losses.x0.l1_weight=1.0 losses.feature_matching.weight=100 losses.adversarial.weight=10 losses.real_pass1.hole_weight=0 losses.real_pass1.forward=False";;
  *) echo "bad arm: $ARM"; exit 1;;
esac
exec env CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
  /home/xzy/miniconda3/envs/warpgan/bin/python scripts/train_inpainting_diffusion.py \
  exp_dir=$EXP max_steps=30 \
  wplus.mode=s1_mapper wplus.init_from_stage_c=False $X0 \
  losses.real_pass1.mirror_anchor=True losses.real_pass1.visible_weight=1.0 \
  losses.real_pass1.blind_struct_weight=0 \
  losses.synth_texture=full \
  losses.pixel.l1_weight=0 losses.pixel.resnet_pl_weight=0 losses.pixel.id_weight=0 \
  losses.latent.weight=0 losses.preserve.weight=0 \
  reference.use_mirror=True reference.real_primary=mirror reference.local_window_k=0 \
  warp.cond_erode_kernel=3 warp.cond_gaussian_blur_kernel=21 \
  smoke.check=True log.val_interval=20 log.save_interval=10000 log.image_interval=10 \
  >> "$LOG" 2>&1
