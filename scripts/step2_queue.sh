#!/bin/bash
# Step-2 v2 completion queue: wait for mirror-gate0 training, then A/B vs baseline.
export PATH=/home/xzy/miniconda3/envs/warpgan/bin:$PATH
cd /data/xzy/warpgan20260803/20260803/warpgan_orig/WarpGAN-main

TRAIN_PID="$1"
LOGS=/data/xzy/warpgan20260803/20260803/train_logs

while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 120; done
echo "[queue] training $TRAIN_PID exited $(date) -> running A/B" >> "$LOGS/step2v2_queue.log"

/home/xzy/miniconda3/envs/warpgan/bin/python scripts/eval_step2_ab.py \
    > "$LOGS/step2v2_ab.log" 2>&1
echo "[queue] A/B rc=$? $(date) DONE" >> "$LOGS/step2v2_queue.log"
