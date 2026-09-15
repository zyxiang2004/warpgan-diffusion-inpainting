#!/bin/bash
# =============================================================================
# v18w1 post-reboot recovery — run ONCE after the server is rebooted:
#   bash scripts/v18w1_recover_after_reboot.sh
# - Excludes the FAILED card by PCI bus id (0000:3b:00.0, Xid 79) — indices
#   may have been renumbered by the reboot, PCI id is the stable identifier.
# - CUDA-probes each remaining card; only healthy ones get training.
# - STRUCT resumes from its latest ckpt; FULL resumes if it has one, else
#   fresh; CTRL fresh (its 1500-step run had no checkpoint).
# - Restarts the health monitor.
# =============================================================================
ROOT=/home/xzy/warpgan-diffusiong-inpainting/warpgan-diffusion-inpainting-main
BAD_PCI=00000000:3b:00.0     # nvidia-smi prints bus id in this 16-digit form
cd $ROOT || exit 1
export CONDA_PREFIX=/home/xzy/miniconda3/envs/warpgan
export PATH=$CONDA_PREFIX/bin:$PATH

echo "[recover] enumerating GPUs (excluding $BAD_PCI)..."
mapfile -t CAND < <(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader \
  | tr -d ' ' | grep -v -i "3b:00.0" | cut -d, -f1)
echo "[recover] candidate GPUs: ${CAND[*]:-NONE}"

HEALTHY=()
for G in "${CAND[@]}"; do
  echo -n "[recover] probing GPU $G ... "
  if timeout 60 env CUDA_VISIBLE_DEVICES=$G $CONDA_PREFIX/bin/python -c \
      "import torch; x=torch.randn(64,64,device='cuda'); float((x@x).sum())" \
      >/dev/null 2>&1; then
    echo "OK"; HEALTHY+=("$G")
  else
    echo "FAILED (skipping)"
  fi
done
echo "[recover] healthy GPUs: ${HEALTHY[*]:-NONE}"
[ ${#HEALTHY[@]} -ge 1 ] || { echo "[recover] NO healthy GPU — abort."; exit 1; }

GPU_CTRL=${HEALTHY[0]}
GPU_FULL=${HEALTHY[1]:-$GPU_CTRL}
GPU_STRUCT=${HEALTHY[2]:-$GPU_CTRL}
echo "[recover] assignment: ctrl->$GPU_CTRL full->$GPU_FULL struct->$GPU_STRUCT"

# --- launch (resume where a checkpoint exists; the arm scripts' resume
# branch re-states ALL overrides — accident-log discipline) ---
echo "[recover] launching STRUCT (resume if ckpt)..."
GPU=$GPU_STRUCT setsid nohup bash scripts/v18w1_struct.sh resume >/dev/null 2>&1 </dev/null &
sleep 2
echo "[recover] launching FULL (resume if ckpt)..."
GPU=$GPU_FULL setsid nohup bash scripts/v18w1_full.sh resume >/dev/null 2>&1 </dev/null &
sleep 2
echo "[recover] launching CTRL (fresh)..."
GPU=$GPU_CTRL setsid nohup bash scripts/v18w1_ctrl.sh >/dev/null 2>&1 </dev/null &
sleep 2
echo "[recover] restarting health monitor..."
setsid nohup bash scripts/v18w1_monitor.sh >/dev/null 2>&1 </dev/null &

sleep 20
echo "[recover] processes: $(pgrep -cf train_inpainting_diffusion.py) monitor: $(pgrep -cf v18w1_monitor)"
echo "[recover] DONE — check: tail train_logs/v18w1_*.log"
