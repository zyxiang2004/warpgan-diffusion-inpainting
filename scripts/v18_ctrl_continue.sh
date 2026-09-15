#!/bin/bash
# =============================================================================
# CTRL continuation: when the control arm reaches its 50K budget and exits,
# relaunch it (resume) with MAX_STEPS=100000 so the final evaluation compares
# all arms at MATCHED budgets. Polls every 10 min, up to 3 days.
# =============================================================================
ROOT=/home/xzy/warpgan-diffusiong-inpainting/warpgan-diffusion-inpainting-main
cd $ROOT || exit 1
END=$((SECONDS + 60*60*24*3))
while [ $SECONDS -lt $END ]; do
  CK=experiments/train_inpainting_diffusion/*_v18w1_ctrl_50k/checkpoints/iteration_00049999.pt
  # shellcheck disable=SC2206
  FOUND=($CK)
  if [ -f "${FOUND[0]}" ] && [ "$(pgrep -cf v18w1_ctrl_50k)" = "0" ]; then
    echo "[ctrl-ext] 50K reached and process exited -> relaunching to 100K at $(date)" \
      >> train_logs/v18w1_ctrl.log
    MAX_STEPS=100000 GPU=0 setsid nohup bash scripts/v18w1_ctrl.sh resume \
      >/dev/null 2>&1 </dev/null &
    exit 0
  fi
  sleep 600
done
