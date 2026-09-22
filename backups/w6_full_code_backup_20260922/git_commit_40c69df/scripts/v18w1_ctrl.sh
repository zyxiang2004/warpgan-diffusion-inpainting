#!/bin/bash
# =============================================================================
# v18-wave1 CTRL arm (GPU 1): the distribution-supervision-OFF control.
# Base = v17-prime structure + P1 fix + W+ s1 ladder (teacher ⑤). The ONLY
# difference vs FULL: the whole x0/FM/adv family is OFF — pure eps-MSE at the
# same base. This isolates the causal effect of the distribution family.
# Full-band pass1 anchor eps is achieved via hole_weight=1.0 on the weighted
# path (w_map == 1 everywhere) — identical semantics to FULL's x0-enable
# full-band branch, minus the distribution terms.
# =============================================================================
ARM=ctrl
GPU=${GPU:-1}
ROOT=/home/xzy/warpgan-diffusiong-inpainting/warpgan-diffusion-inpainting-main
EXP=$ROOT/experiments/train_inpainting_diffusion/v18w1_${ARM}_50k
LOG=$ROOT/train_logs/v18w1_${ARM}.log
mkdir -p $ROOT/train_logs
cd $ROOT || exit 1
# Replicate `conda activate warpgan` under nohup (activate.d hooks set
# CUDA_HOME/CC/CXX/CPATH — required for the splatting/StyleGAN JIT extensions;
# sourcing ONLY zz_cuda_home.sh also pins CC=gcc (conda gcc 14.3 is rejected
# by CUDA 12.x nvcc — ENV_SETUP_NOTES §1).
export CONDA_PREFIX=/home/xzy/miniconda3/envs/warpgan
export PATH=$CONDA_PREFIX/bin:$PATH
. $CONDA_PREFIX/etc/conda/activate.d/zz_cuda_home.sh
# Resume discipline (accident log #10/#11): ALL overrides restated on EVERY
# launch; resume picks the newest checkpoint automatically.
EXTRA=""
if [ -n "$1" ] && [ "$1" = "resume" ]; then
  # fresh runs get a [timestamp]_ prefix (addtime2path) — glob it
  CKPT=$(ls -1 $ROOT/experiments/train_inpainting_diffusion/*_v18w1_${ARM}_50k/checkpoints/*.pt 2>/dev/null | sort | tail -1)
  if [ -n "$CKPT" ]; then EXTRA="checkpoint_path='$CKPT'"; echo "[v18w1-$ARM] resume from $CKPT at $(date)" >> "$LOG"; fi
fi
exec env CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
  /home/xzy/miniconda3/envs/warpgan/bin/python scripts/train_inpainting_diffusion.py \
  exp_dir=$EXP \
  max_steps=${MAX_STEPS:-50000} $EXTRA \
  wplus.mode=s1_mapper wplus.init_from_stage_c=False \
  losses.x0.enable=False \
  losses.feature_matching.weight=0 losses.adversarial.weight=0 \
  losses.real_pass1.forward=True \
  losses.real_pass1.mirror_anchor=True losses.real_pass1.visible_weight=1.0 \
  losses.real_pass1.hole_weight=1.0 losses.real_pass1.blind_struct_weight=0 \
  losses.synth_texture=full \
  losses.pixel.l1_weight=0 losses.pixel.resnet_pl_weight=0 losses.pixel.id_weight=0 \
  losses.latent.weight=0 losses.preserve.weight=0 \
  reference.use_mirror=True reference.real_primary=mirror reference.local_window_k=0 \
  warp.cond_erode_kernel=3 warp.cond_gaussian_blur_kernel=21 \
  smoke.check=True log.val_interval=2000 log.save_interval=10000 log.image_interval=500 \
  >> "$LOG" 2>&1
