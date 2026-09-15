#!/bin/bash
# =============================================================================
# v18.3 W3-D arm (default GPU 1): InvSR-calibrated supervision + D 结构.
#   * pass2/synth: latent-space MSE(z0̂,target)×1 + FM(D feats)×2 + adv×0.1,
#     t<=300 only, timestep-conditioned multi-scale UNet-D (hinge, warmup 3K)
#   * real batch pass1 DISABLED (real_pass1.forward=False) — the original
#     FFC bet: hole-filling learned at pass2 (now properly supervised)
#     transfers to the novel view; pass2 bank = [warp_warp_img, x_mirror PHOTO]
#   * everything else = the v18 base (s1_mapper W+ ladder, mirror bank,
#     cond soften erode3/blur21, full timestep eps-MSE, 50-step DDIM val)
# CTRL (GPU 0, still running) is the oil-paint control/reference.
# =============================================================================
ARM=w3d
GPU=${GPU:-1}
ROOT=/home/xzy/warpgan-diffusiong-inpainting/warpgan-diffusion-inpainting-main
EXP=$ROOT/experiments/train_inpainting_diffusion/v18w3_${ARM}_30k
LOG=$ROOT/train_logs/v18w3_${ARM}.log
mkdir -p $ROOT/train_logs
cd $ROOT || exit 1
export CONDA_PREFIX=/home/xzy/miniconda3/envs/warpgan
export PATH=$CONDA_PREFIX/bin:$PATH
. $CONDA_PREFIX/etc/conda/activate.d/zz_cuda_home.sh
EXTRA=""
if [ -n "$1" ] && [ "$1" = "resume" ]; then
  CKPT=$(ls -1 $ROOT/experiments/train_inpainting_diffusion/*_v18w3_${ARM}_30k/checkpoints/*.pt 2>/dev/null | sort | tail -1)
  if [ -n "$CKPT" ]; then EXTRA="checkpoint_path='$CKPT'"; echo "[v18w3-$ARM] resume from $CKPT at $(date)" >> "$LOG"; fi
fi
exec env CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
  /home/xzy/miniconda3/envs/warpgan/bin/python scripts/train_inpainting_diffusion.py \
  exp_dir=$EXP \
  max_steps=${MAX_STEPS:-100000} $EXTRA \
  wplus.mode=s1_mapper wplus.init_from_stage_c=False \
  losses.x0.enable=True losses.x0.mode=invsr \
  losses.x0.t_max=300 losses.x0.w_ldif=1.0 losses.x0.w_lfm=2.0 losses.x0.w_ldis=0.1 \
  losses.x0.dis_warmup=3000 losses.x0.latent_bound=10.0 \
  losses.feature_matching.weight=0 losses.adversarial.weight=0 \
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
