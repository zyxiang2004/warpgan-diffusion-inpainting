# BrushNet Cycle Training — Orig-Faithful Dual-DDIM (512 Resolution)

## Overview

This is the **complete orig-faithful cycle** that fixes black pinhole artifacts and
fragmented textures in NOVEL SD output's hole region. The key fix is implementing
the **second inpaintor pass** that orig WarpGAN does but we never had:

```
① DDIM sample (novel-view condition) → pred_novel
② forward_warp(pred_novel, c_novel→c) → warp_warp_img (source-view hole)
③ DDIM sample (source-view condition from warp_warp_img) → pred_inv_warp
④ All orig losses on pred_inv_warp vs source photo
```

The loss acts on `pred_inv_warp` (source-view, same domain as source photo),
NOT on `pred_back` (which has empty holes). This is the critical difference from
all 20+ previous failed iterations.

## What Changed (vs. old 256-resolution single-pass)

| Component | Old (failed) | New (orig-faithful) |
|-----------|-------------|---------------------|
| Resolution | 256 | **512** (matches validation) |
| DDIM passes | 1 (novel only) | **2** (novel + source re-inpaint) |
| Gradient steps | last 2/40 only | **all 40 steps** (full chain) |
| Loss target | pred_back (empty holes) | **pred_inv_warp** (inpainted) |
| L1 weight | 1.0 | **10.0** (orig weight_known) |
| L1 mask | visible-only | **full-image** (with_mask=False) |
| ID loss | 0.0 | **0.5** (orig losses.id.weight) |
| Discriminator sees | [pred_novel, pred_back] | **[pred_novel, pred_inv_warp]** |

## Loss Structure (exact orig translation)

From `cal_inpaintor_loss()` L1054-1074 + `cal_inpaintor_rec_loss()` L1118-1212:

```
with_mask = False  →  mask = zeros  →  full-image L1
weight_known = 10, weight_missing = 0  →  all pixels weighted 10

L1:     F.l1_loss(pred_inv_warp, source) × 10
GAN:    softplus(-D(pred)) × 10, on [pred_novel, pred_inv_warp] vs source
FM:     feature_matching(D_fake_feats, D_real_feats) × 100
ID:     id_loss(pred_inv_warp, source, source) × 0.5
```

## Memory Requirements

32-bit AdamW (orig BrushNet recipe, **not** 8-bit) + per-step gradient checkpointing:

```
Model weights (SD UNet + VAE + BrushNet + RefNet + Disc): ~11.5 GB
32-bit AdamW optimizer state (618M × 8 bytes):            ~4.6  GB
Second chain activations (UNet 1-step + VAE + BrushNet):   ~9    GB
Discriminator forward:                                     ~4    GB
Other (data / CUDA context / fragmentation):               ~3    GB
─────────────────────────────────────────────────────────────────
Total at 512 res / 40 DDIM steps:                          ~32   GB
```

| GPU | 显存 | 预估剩余 | 评价 |
|-----|------|---------|------|
| A100 40GB | 40 GB | ~8 GB | ⚠️ 可用但偏紧，建议加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| **A600 48GB** | 48 GB | ~16 GB | ✅ **推荐** — 充裕 |
| A100 80GB | 80 GB | ~48 GB | ✅ 理想 — 可增大 batch_size |
| V100 32GB | 32 GB | ~0 GB | ❌ 太紧 |
| 3090/4090 24GB | 24 GB | — | ❌ 不够（512 分辨率无法运行） |

## Quick Start

### 1. Set up environment

```bash
cd /path/to/warpgan20260803/20260803
bash scripts/setup_env.sh
```

### 2. Start training (5000-step probe)

```bash
source warpgan_env/bin/activate

# For A100 40GB (tight, ~8GB headroom):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 python scripts/run_brushnet_cycle_finetune.py

# For A600 48GB / A100 80GB (comfortable):
CUDA_VISIBLE_DEVICES=0 python scripts/run_brushnet_cycle_finetune.py
```

### 3. Long training (300k steps)

```bash
WARP_GAN_MAX_STEPS=300000 CUDA_VISIBLE_DEVICES=0 python scripts/run_brushnet_cycle_finetune.py
```

### 4. Resume from checkpoint

```bash
WARP_GAN_RESUME_CHECKPOINT=experiments/_brushnet_cycle_512_dual/checkpoints/iteration_5000.pt \
CUDA_VISIBLE_DEVICES=0 python scripts/run_brushnet_cycle_finetune.py
```

## Configuration

All config is in `scripts/run_brushnet_cycle_finetune.py`. Key options:

| Env var | Default | Description |
|---------|---------|-------------|
| `WARP_GAN_MAX_STEPS` | 5000 | Training steps |
| `WARP_GAN_EXP_DIR` | `experiments/_brushnet_cycle_512_dual` | Output directory |
| `WARP_GAN_OVERWRITE` | 0 | Set to 1 to overwrite existing exp dir |
| `WARP_GAN_RESUME_CHECKPOINT` | (none) | Checkpoint to resume from |
| `WARP_GAN_STAGE_A_CHECKPOINT` | `experiments/_real_reference_pretrain/checkpoints/iteration_10000.pt` | Stage A weights |
| `WARP_GAN_STAGE_C_CHECKPOINT` | `experiments/_wplus_identity_pretrain/checkpoints/iteration_100000.pt` | Stage C weights |

## Verification

Run the smoke test (256 res, 2 DDIM steps, fits 24GB):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test_dual_ddim.py
```

Expected output:
```
=== SMOKE TEST PASSED ===
  Dual DDIM cycle: novel -> back-warp -> source re-inpaint -> loss
  Loss components: loss_cycle_l1, loss_cycle_gan, loss_cycle_fm, loss_cycle_id
  BrushNet gradients flowing: True
  Peak GPU memory: ~15 GB (at 256 res, 2 steps)
```

## Monitoring

Training logs are in `experiments/_brushnet_cycle_512_dual/`:
- `brushnet_cycle_health.txt`: per-50-step metrics
- TensorBoard: `tensorboard --logdir experiments/_brushnet_cycle_512_dual`
- Checkpoints: `checkpoints/iteration_NNNN.pt` (every `save_interval` steps)
