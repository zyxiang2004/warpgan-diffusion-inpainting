#!/bin/bash
# =============================================================================
# "Safe to reboot" signal: every 5 min, report whether STRUCT/FULL have
# saved their first (10K-step) checkpoint. Writes train_logs/reboot_ready.status.
# STRUCT=ready -> the recommended reboot point (keeps 10K steps).
# FULL=ready too -> alternative A ready (keeps both arms' 10K).
# =============================================================================
ROOT=/home/xzy/warpgan-diffusiong-inpainting/warpgan-diffusion-inpainting-main
cd $ROOT || exit 1
OUT=train_logs/reboot_ready.status
END=$((SECONDS + 60*60*12))
while [ $SECONDS -lt $END ]; do
  {
    echo "===== $(date '+%F %T') ====="
    for A in struct full; do
      CK=$(ls experiments/train_inpainting_diffusion/*_v18w1_${A}_50k/checkpoints/*.pt 2>/dev/null | sort | tail -1)
      ST=$(grep -oE 'step [0-9]+' train_logs/v18w1_$A.log 2>/dev/null | tail -1)
      if [ -n "$CKPT" ] || [ -n "$CK" ]; then
        echo "[$A] READY ckpt=$(basename "${CK:-none}")  $ST"
      else
        echo "[$A] not-ready  $ST"
      fi
    done
    echo "CTRL is dead (failed GPU) and cannot restart before reboot — see PREREG_V18W1_INCIDENTS.md"
  } >> "$OUT" 2>&1
  sleep 300
done
