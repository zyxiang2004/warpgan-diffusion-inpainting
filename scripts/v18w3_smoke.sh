#!/bin/bash
# 30-step smoke for one v18.3 arm: v18w3_smoke.sh <w3d|w3p> <gpu>
ARM=$1; GPU=${2:-1}
ROOT=/home/xzy/warpgan-diffusiong-inpainting/warpgan-diffusion-inpainting-main
cd $ROOT || exit 1
export CONDA_PREFIX=/home/xzy/miniconda3/envs/warpgan
export PATH=$CONDA_PREFIX/bin:$PATH
. $CONDA_PREFIX/etc/conda/activate.d/zz_cuda_home.sh
EXP=$ROOT/experiments/train_inpainting_diffusion/smoke_v18w3_${ARM}
LOG=$ROOT/train_logs/smoke_v18w3_${ARM}.log
mkdir -p $ROOT/train_logs
case $ARM in
  w3d) P1F=False;;
  w3p) P1F=True;;
  *) echo "bad arm $ARM"; exit 1;;
esac
exec env CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
  /home/xzy/miniconda3/envs/warpgan/bin/python scripts/train_inpainting_diffusion.py \
  exp_dir=$EXP max_steps=30 \
  wplus.mode=s1_mapper wplus.init_from_stage_c=False \
  losses.x0.enable=True losses.x0.mode=invsr \
  losses.x0.t_max=300 losses.x0.w_ldif=1.0 losses.x0.w_lfm=2.0 losses.x0.w_ldis=0.1 \
  losses.x0.dis_warmup=3000 losses.x0.latent_bound=10.0 \
  losses.feature_matching.weight=0 losses.adversarial.weight=0 \
  losses.real_pass1.forward=$P1F \
  losses.real_pass1.mirror_anchor=True losses.real_pass1.visible_weight=1.0 \
  losses.real_pass1.hole_weight=0 losses.real_pass1.blind_struct_weight=0 \
  losses.synth_texture=full \
  losses.pixel.l1_weight=0 losses.pixel.resnet_pl_weight=0 losses.pixel.id_weight=0 \
  losses.latent.weight=0 losses.preserve.weight=0 \
  reference.use_mirror=True reference.real_primary=mirror reference.local_window_k=0 \
  warp.cond_erode_kernel=3 warp.cond_gaussian_blur_kernel=21 \
  smoke.check=True log.val_interval=20 log.save_interval=100000 log.image_interval=10 \
  >> "$LOG" 2>&1
