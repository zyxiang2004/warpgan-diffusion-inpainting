#!/bin/bash
# v17-prime: GATE-FREE run — fresh from scratch, 100K steps.
# Recipe (first-principles, no hand gates):
#   eps-MSE full-band everywhere (dual-band/sigma retired)
#   x0 re-noised symmetric pairs -> L1(ab_t-weighted) + FM(10) on pass2+synth
#   D real = PHOTOS on both batches; adv = 0 (FM-first)
#   pass2 bank primary = warp_warp_img (teacher rule 1)
LOG=/data/xzy/warpgan20260803/20260803/train_logs/step4_v17prime_100k.log
cd /data/xzy/warpgan20260803/20260803/warpgan_orig/WarpGAN-main || exit 1
if [ -n "$1" ] && [ "$1" = "resume" ]; then
  CKPT=$(ls -1 experiments/train_inpainting_diffusion/step4_v17prime_gatefree_100k/checkpoints/*.pt 2>/dev/null | sort | tail -1)
  if [ -z "$CKPT" ]; then echo "[v17] no ckpt, fresh start" >> "$LOG"; EXTRA=""
  else EXTRA="checkpoint_path=$CKPT"; echo "[v17] resume from $CKPT at $(date)" >> "$LOG"; fi
else
  EXTRA=""
fi
exec env PATH=/home/xzy/miniconda3/envs/warpgan/bin:$PATH CUDA_VISIBLE_DEVICES=0 \
  PYTHONUNBUFFERED=1 \
  /home/xzy/miniconda3/envs/warpgan/bin/python scripts/train_inpainting_diffusion.py \
  exp_dir=./experiments/train_inpainting_diffusion/step4_v17prime_gatefree_100k \
  max_steps=100000 $EXTRA \
  losses.x0.enable=True losses.x0.l1_weight=1.0 \
  losses.feature_matching.weight=10 losses.adversarial.weight=0 \
  losses.real_pass1.mirror_anchor=True losses.real_pass1.visible_weight=1.0 \
  losses.real_pass1.blind_struct_weight=0 losses.real_pass1.hole_weight=0 \
  losses.synth_texture=full \
  losses.pixel.l1_weight=0 losses.pixel.resnet_pl_weight=0 losses.pixel.id_weight=0 \
  losses.latent.weight=0 losses.preserve.weight=0 \
  reference.use_mirror=True reference.real_primary=mirror reference.local_window_k=0 \
  warp.cond_erode_kernel=3 warp.cond_gaussian_blur_kernel=21 \
  smoke.check=False log.val_interval=1000 log.save_interval=10000 log.image_interval=500 \
  >> "$LOG" 2>&1
