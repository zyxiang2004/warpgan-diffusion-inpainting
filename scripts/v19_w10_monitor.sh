#!/bin/bash
# W10 monitor: every 30 min — disk check + validator scoring of the three
# live arms + W6-A reference. Appends to train_logs/w10_monitor.log.
ROOT=/home/xzy/warpgan-diffusiong-inpainting/warpgan-diffusion-inpainting-main
PY=/home/xzy/miniconda3/envs/warpgan/bin/python
cd $ROOT || exit 1
while true; do
  {
    echo "===== $(date) ====="
    df -h / | tail -1
    for D in "$ROOT/experiments/train_inpainting_diffusion/[20260915-085452]_v19_w6a_300k" \
             "$ROOT/experiments/train_inpainting_diffusion/v19_w10_20k" \
             "$ROOT/experiments/train_inpainting_diffusion/v19_w10c_20k" \
             "$ROOT/experiments/train_inpainting_diffusion/v19_ema_300k"; do
      if [ -d "$D/logs/images/val" ]; then
        $PY scripts/score_val_frames.py "$D" 2>/dev/null | grep -v Warning
      fi
    done
    # hard stop line: validator good-rate of a 10-frame window > 50% logged
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  } >> train_logs/w10_monitor.log 2>&1
  sleep 1800
done
