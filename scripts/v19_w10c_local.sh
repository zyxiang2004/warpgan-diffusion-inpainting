#!/bin/bash
# =============================================================================
# v19 W10-C LOCAL (single RTX 4090D): §7.30 fix (anchor_clean=selective_hole)
# + EMA 0.999 combination arm. Local adaptation of server's
# scripts/v19_w10c_anchorclean_ema.sh — recipe flags IDENTICAL; only plumbing:
#   - ROOT/GPU/log adapted; zz_cuda_home.sh dropped (torch cu128 ships runtime)
#   - BASE = local W6-A@150K iteration_0149999.pt (server forks W6-A@160K;
#     local-only lineage) -> max_steps 169999 = same 20K decision window
#   - fork copies base ckpt into ITS OWN dir first (server W10 smoke lesson:
#     train script DERIVES exp_dir from ckpt path)
#   - resume guarded (RESUME_CKPT override + refuse silent fresh)
# ckpt rotation: log.max_ckpt_keep=2 (disk guard, val PNGs unaffected).
# Usage: bash scripts/v19_w10c_local.sh          (fresh fork from W6-A@150K)
#        bash scripts/v19_w10c_local.sh resume   (resume from latest own ckpt)
# =============================================================================
ARM=w10c
GPU=${GPU:-0}
ROOT=/data/xzy/warpgan20260803/20260803/warpgan_orig/WarpGAN-main
EXP=$ROOT/experiments/train_inpainting_diffusion/v19_${ARM}_20k_local
LOG=$ROOT/train_logs/v19_${ARM}_LOCAL.log
BASE=$ROOT/experiments/train_inpainting_diffusion/[20260915-111722]_v19_w6a_100k/checkpoints/iteration_0149999.pt
mkdir -p $ROOT/train_logs
cd $ROOT || exit 1
export CONDA_PREFIX=/home/xzy/miniconda3/envs/warpgan
export PATH=$CONDA_PREFIX/bin:$PATH

if [ -n "$1" ] && [ "$1" = "resume" ]; then
  if [ -n "$RESUME_CKPT" ]; then CKPT="$RESUME_CKPT"
  else CKPT=$(ls -1 $EXP/checkpoints/*.pt 2>/dev/null | sort | tail -1); fi
  if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
    EXTRA="checkpoint_path='$CKPT'"
    echo "[v19-$ARM-local] resume from $CKPT at $(date)" >> "$LOG"
  else
    echo "[v19-$ARM-local] ERROR: resume requested, no ckpt found. ABORT (refuse silent fresh)." >&2
    exit 1
  fi
else
  if [ ! -f "$BASE" ]; then echo "[v19-$ARM-local] ERROR: base ckpt missing: $BASE" >&2; exit 1; fi
  mkdir -p $EXP/checkpoints
  if [ ! -e "$EXP/checkpoints/iteration_0149999.pt" ]; then
    echo "[v19-$ARM-local] copying base ckpt (5.6G) ..." >> "$LOG"
    cp "$BASE" $EXP/checkpoints/ || exit 1
  fi
  EXTRA="checkpoint_path='$EXP/checkpoints/iteration_0149999.pt'"
  echo "[v19-$ARM-local] fork from $EXTRA at $(date)" >> "$LOG"
fi

exec env CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
  /home/xzy/miniconda3/envs/warpgan/bin/python scripts/train_inpainting_diffusion.py \
  exp_dir=$EXP \
  max_steps=${MAX_STEPS:-169999} $EXTRA \
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
