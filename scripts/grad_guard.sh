#!/bin/bash
# =============================================================================
# grad_guard.sh ARM [THRESHOLD] — W8-lite guardrail (7.22 lesson): kill the
# arm's training if the last 3 logged g_grad_abs all exceed THRESHOLD
# (default 3500 = ~5x the W6-era baseline ~660). Zero-risk: reads the log,
# never touches training code. Usage: nohup bash scripts/grad_guard.sh w8lite &
# =============================================================================
ARM=${1:-w8lite}
THRESH=${2:-3500}
ROOT=/home/xzy/warpgan-diffusiong-inpainting/warpgan-diffusion-inpainting-main
LOG=$ROOT/train_logs/v19_${ARM}.log
echo "[grad_guard] watching $ARM, threshold=$THRESH, every 300s"
while true; do
  sleep 300
  LAST3=$(grep -oE 'g_grad_abs=[0-9.]+' "$LOG" 2>/dev/null | tail -3 | cut -d= -f2)
  N=$(echo "$LAST3" | grep -cE '')
  if [ "$(echo "$LAST3" | grep -c '.')" -ge 3 ]; then
    OVER=$(echo "$LAST3" | awk -v t=$THRESH '$1 > t {c++} END {print c+0}')
    if [ "$OVER" -ge 3 ]; then
      echo "[grad_guard] $(date): 3 consecutive g_grad_abs > $THRESH in $ARM -> KILL" >> "$LOG"
      pkill -f "v19_${ARM}" && echo "[grad_guard] killed" >> "$LOG"
      exit 0
    fi
  fi
done
