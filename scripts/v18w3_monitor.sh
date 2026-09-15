#!/bin/bash
# =============================================================================
# v18.3 health monitor for the W3 arms (same pattern as v18w1_monitor.sh):
# every 10 min, per-arm status -> train_logs/v18w3_health.status, for up to 7 d.
# Operator alarm conditions:
#   tracebacks > 0 or alive=0 before target step  -> crash -> resume via
#     `MAX_STEPS=100000 GPU=<n> bash scripts/v18w3_{d,p}.sh resume`
#   d_real/d_fake pinned (>0.95 / <-0.95 across many polls) -> D saturation
#   val_novel_full stuck above 0.2 for many polls -> sampling regression
# =============================================================================
ROOT=/home/xzy/warpgan-diffusiong-inpainting/warpgan-diffusion-inpainting-main
cd $ROOT || exit 1
STATUS=train_logs/v18w3_health.status
mkdir -p train_logs
END=$((SECONDS + 60*60*24*7))
while [ $SECONDS -lt $END ]; do
  {
    echo "===== $(date '+%F %T') ====="
    for A in w3d w3p; do
      L=train_logs/v18w3_${A}.log
      if [ ! -f "$L" ]; then echo "[$A] no log"; continue; fi
      ALIVE=$(pgrep -cf "v18w3_${A}_30k")
      TB=$(grep -c 'Traceback' "$L" || true)
      STEP=$(grep -oE 'step [0-9]+' "$L" | tail -1)
      echo "[$A] alive=$ALIVE tb=$TB $STEP"
      for KEY in real_p2_ldif real_p2_lfm real_p2_gen_adv synth_x0_ldif \
                 discr_adv g_grad_abs val_novel_full; do
        V=$(grep -E "${KEY}=" "$L" | tail -1 | grep -oE "${KEY}=[-0-9.]+" | tail -1)
        [ -n "$V" ] && echo "[$A] $V"
      done
    done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/[gpu] /'
  } >> "$STATUS" 2>&1
  sleep 600
done
