#!/bin/bash
# =============================================================================
# v18-wave1 health monitor: appends a per-arm status block every 10 min for
# up to 40 h. Reads only text logs / process table — zero GPU cost.
# Alarm conditions the operator (agent) checks each poll:
#   tracebacks > 0            -> crash, restart via the arm script with 'resume'
#   alive=0 before step 50000 -> process died, check log tail
#   p2_discr_real_out or x0_discr_real_out pinned (>0.95 or <0.05 for many
#                                polls) -> D saturation (shortcut suspicion)
#   g_grad_abs explosion vs CTRL's value -> divergence stop-loss evaluation
# =============================================================================
ROOT=/home/xzy/warpgan-diffusiong-inpainting/warpgan-diffusion-inpainting-main
cd $ROOT || exit 1
STATUS=train_logs/v18w1_health.status
mkdir -p train_logs
END=$((SECONDS + 60*60*40))
while [ $SECONDS -lt $END ]; do
  {
    echo "===== $(date '+%F %T') ====="
    for ARM in ctrl full struct; do
      L=train_logs/v18w1_${ARM}.log
      if [ ! -f "$L" ]; then echo "[$ARM] no log yet"; continue; fi
      ALIVE=$(pgrep -cf "v18w1_${ARM}_50k")
      TB=$(grep -c 'Traceback' "$L" || true)
      STEP=$(grep -oE 'step [0-9]+' "$L" | tail -1)
      echo "[$ARM] alive=$ALIVE tracebacks=$TB $STEP"
      for KEY in real_p1_eps real_p2_eps real_p2_x0_l1 real_p2_gen_fm \
                 synth_x0_gen_fm discr_adv g_grad_abs; do
        V=$(grep -E "^\s*${KEY} = " "$L" | tail -1 | sed 's/^.*= //')
        [ -n "$V" ] && echo "[$ARM] $KEY = $V"
      done
      grep -E 'p2_discr_(real|fake)_out = |x0_discr_(real|fake)_out = ' "$L" | tail -2 | sed 's/^/[$ARM] /'
      VAL=$(grep -E 'val_novel' "$L" | tail -1)
      [ -n "$VAL" ] && echo "[$ARM] $VAL"
    done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/[gpu] /'
  } >> "$STATUS" 2>&1
  sleep 600
done
