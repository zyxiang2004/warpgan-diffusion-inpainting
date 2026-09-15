#!/bin/bash
# v17-prime 100K liveness watchdog. Exits once the 100K ckpt exists.
LOG=/data/xzy/warpgan20260803/20260803/train_logs/step4_v17prime_100k.log
CKPT_DIR=experiments/train_inpainting_diffusion/step4_v17prime_gatefree_100k/checkpoints
cd /data/xzy/warpgan20260803/20260803/warpgan_orig/WarpGAN-main || exit 1
while true; do
  sleep 600
  [ -f "$CKPT_DIR/iteration_0099999.pt" ] && exit 0
  pgrep -f 'train_inpainting_diffusion.py' > /dev/null && continue
  timeout 10 ls ./data/ > /dev/null 2>&1 || continue
  nohup /data/xzy/warpgan20260803/20260803/warpgan_orig/WarpGAN-main/scripts/v17prime_fresh.sh resume >> "$LOG" 2>&1 < /dev/null &
done
