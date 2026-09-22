#!/bin/bash
# =============================================================================
# v19 W10-C (GPU 2): §7.30 fix + EMA (0.999, InvSR convention) — the
# combination arm. Forked from W6-A@160K. vs W10 single variable = +EMA;
# vs EMA control single variable = +anchor fix. Attribution:
#   W10 (fix) / EMA (stabilizer) / W10C (both) vs frozen W6-A@160K baseline.
# =============================================================================
ARM=w10c
GPU=${GPU:-2}
ROOT=/home/xzy/warpgan-diffusiong-inpainting/warpgan-diffusion-inpainting-main
EXP=$ROOT/experiments/train_inpainting_diffusion/v19_w10c_20k
LOG=$ROOT/train_logs/v19_w10c.log
mkdir -p $ROOT/train_logs
cd $ROOT || exit 1
export CONDA_PREFIX=/home/xzy/miniconda3/envs/warpgan
export PATH=$CONDA_PREFIX/bin:$PATH
. $CONDA_PREFIX/etc/conda/activate.d/zz_cuda_home.sh
BASE=$ROOT/experiments/train_inpainting_diffusion/[20260915-085452]_v19_w6a_300k/checkpoints/iteration_0159999.pt
# train_inpainting_diffusion.py DERIVES exp_dir from the ckpt path — a fork
# MUST copy the base ckpt into ITS OWN checkpoints/ dir first (W10 smoke
# misfire lesson, 2026-09-22).
if [ -n "$1" ] && [ "$1" = "resume" ]; then
  CKPT=$(ls -1 $EXP/checkpoints/*.pt 2>/dev/null | sort | tail -1)
  [ -z "$CKPT" ] && CKPT=$EXP/checkpoints/iteration_0159999.pt
  EXTRA="checkpoint_path='$CKPT'"
else
  mkdir -p $EXP/checkpoints
  if [ ! -e "$EXP/checkpoints/iteration_0159999.pt" ]; then
    echo "[v19-$ARM] copying base ckpt (5.6G) ..." >> "$LOG"
    cp "$BASE" $EXP/checkpoints/
  fi
  EXTRA="checkpoint_path='$EXP/checkpoints/iteration_0159999.pt'"
fi
echo "[v19-$ARM] fork from $EXTRA at $(date)" >> "$LOG"
exec env CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
  /home/xzy/miniconda3/envs/warpgan/bin/python scripts/train_inpainting_diffusion.py \
  exp_dir=$EXP \
  max_steps=${MAX_STEPS:-179999} $EXTRA \
  ema.rate=0.999 \
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
  losses.real_pass1.anchor_clean=selective_hole \
  losses.real_pass1.anchor_despeckle_tau=0.07 \
  losses.synth_texture=full \
  losses.pixel.l1_weight=0 losses.pixel.resnet_pl_weight=0 losses.pixel.id_weight=0 \
  losses.latent.weight=0 losses.preserve.weight=0 \
  reference.use_mirror=True reference.real_primary=mirror reference.local_window_k=0 \
  warp.cond_erode_kernel=3 warp.cond_gaussian_blur_kernel=21 \
  smoke.check=True log.val_interval=2000 log.save_interval=10000 log.image_interval=500 \
  log.max_ckpt_keep=2 \
  >> "$LOG" 2>&1
