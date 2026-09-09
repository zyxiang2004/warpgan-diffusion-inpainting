#!/bin/bash
# Serial review queue (setsid-protected): wait for probeB -> final review -> frozen ablation.
export PATH=/home/xzy/miniconda3/envs/warpgan/bin:$PATH
cd /data/xzy/warpgan20260803/20260803/warpgan_orig/WarpGAN-main

PROBE_B_PID="$1"
CKPT="./experiments/train_inpainting_diffusion/[20260823-180346]_step1b_epsonly_25k/checkpoints/iteration_0024999.pt"
LOGS=/data/xzy/warpgan20260803/20260803/train_logs

while kill -0 "$PROBE_B_PID" 2>/dev/null; do sleep 60; done

echo "[queue] probeB finished $(date) -> final review" >> "$LOGS/review_queue.log"
/home/xzy/miniconda3/envs/warpgan/bin/python scripts/eval_final_review.py "$CKPT" \
    > "$LOGS/final_review.log" 2>&1
echo "[queue] final review rc=$? $(date) -> frozen ablation" >> "$LOGS/review_queue.log"
/home/xzy/miniconda3/envs/warpgan/bin/python scripts/eval_frozen_ablation.py "$CKPT" \
    > "$LOGS/ablation_24999.log" 2>&1
echo "[queue] ablation rc=$? $(date) QUEUE_DONE" >> "$LOGS/review_queue.log"
