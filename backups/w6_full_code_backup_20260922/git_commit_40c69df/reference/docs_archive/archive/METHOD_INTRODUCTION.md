# WarpGAN Diffusion-SVINet 方法说明

> 对照原版 `/data/xzy/warpgan_orig/WarpGAN-main/` 撰写  
> 日期：2026-07-30

---

## 1. 问题

原版 WarpGAN 的 SVINet 使用 LaMa/FFC-ResNet 前馈网络完成新视角补洞。其局限：

1. **生成先验弱**：FFC-ResNet 的生成能力远不如现代 latent diffusion；
2. **EG3D 材质直接泄漏**：`warp.hybrid=True` 时用 EG3D novel render 填充空洞，把 EG3D 的平滑材质直接带入输出；
3. **真实感监督不足**：真实数据通过 inverse warp 回源视角间接监督，不直接约束最终 novel-view 的真实感；
4. **验证指标偏离目标**：原版验证主要衡量回源重建误差，不等价于 novel-view 真实感。

结果是：原版输出结构和身份基本正确，但材质稳定偏平滑（"油画感"）。

---

## 2. 灵感与解决方法

**核心思想**：保留 WarpGAN 的 3D warp、相机和身份优势，用 Stable Diffusion + BrushNet 的生成先验替换 FFC-ResNet。

具体方案：

| 原版组件 | 替换为 | 理由 |
|---|---|---|
| FFC-ResNet generator | 冻结 SD 1.5 UNet + BrushNet | diffusion 生成先验远强于前馈 CNN |
| EG3D novel render 作为 RGB 条件 | 仅作为几何监督（弱权重） | 避免平滑材质直接泄漏 |
| 无 ReferenceNet | 冻结 ReferenceNet + 独立 K/V/O adapter | 从真实 source photo 注入皮肤/头发高频 |
| 无 W+ cross-attention | WProjModel + 独立 Q/K/V/O RCA | 将 3D 身份潜码映射为 SD token |
| 全图 L1/LPIPS | 区域分离：hole 强、visible 弱 | 聚焦缺失区域，不污染已知区 |
| 无 GAN | PatchGAN + R1 | 像素级真实感约束 |

**关键创新**：同一真实 batch 执行两次职责隔离的梯度更新（factorized joint）：

- **appearance 更新**：真实照片自重建 → 只更新 Reference K/V/O；
- **identity 更新**：EG3D novel 伪目标 → 只更新 W+ mapper/RCA。

---

## 3. 实施手段

### 3.1 整体架构

```
Source Photo (512×512)
    │
    ├→ GOAE encoder → W+ [B,14,512]
    │                   │
    │              WProjModel (可训练)
    │                   │ [B,18,768]
    │              W+ RCA (独立 Q/K/V/O + gate)
    │
    ├→ SD VAE encode → ReferenceNet (冻结) → 多尺度特征
    │                   │
    │              Reference K/V/O adapter (可训练)
    │
    └→ Warper(depth, c_src→c_novel)
           │
      warp_img + hole mask
           │
      VAE encode → BrushNet condition
           │
      BrushNet (冻结) → down/mid/up residuals
           │
      冻结 SD 1.5 UNet (attn replaced)
           │
      50-step DPM-Solver++
           │
      VAE decode → SD full-frame (最终输出)
```

**与原版的关键差异**：最终输出是 SD 解码的完整图像，不再执行 `generated * mask + warp * (1-mask)` 硬拼接。

### 3.2 可训练与冻结模块

| 模块 | 状态 | 参数量 |
|---|---|---|
| SD 1.5 VAE | 冻结 | — |
| SD 1.5 UNet 主体 | 冻结 | — |
| BrushNet | 冻结 | — |
| ReferenceNet backbone | 冻结 | — |
| ID Loss / LPIPS 网络 | 冻结 | — |
| WProjModel | **可训练** | ~2M |
| W+ RCA (Q/K/V/O + gate) | **可训练** | ~44M |
| Reference K/V/O adapter + gate | **可训练** | ~37M |
| PatchGAN discriminator | **可训练** | ~3M |

### 3.3 数据流与复用

每个真实 batch 同时构造两个任务，共享同一 source photo 和 W+ codes：

#### 任务 A：Real-Ref Self-Reconstruction（appearance 分支）

```
source photo → novel-view warp → 获取 disocclusion mask 形状
                                    │
                    mask 放回 source frame，挖洞
                                    │
              target = 同一张 source photo（精确像素配对）
              W+ 禁用，强制 Reference 负责纹理
              BrushNet 条件 = masked source + mask
              ReferenceNet 输入 = 完整 source photo
```

**复用关系**：warp 只取 mask 形状，不取 RGB；source photo 同时作为 target 和 ReferenceNet 输入。

#### 任务 B：Novel-View Pseudo-Target（identity 分支）

```
source photo → novel-view warp → warp_img + hole mask
                                    │
              target = EG3D 在 c_novel 下的渲染 y_hat_novel
              W+ 启用，提供身份条件
              BrushNet 条件 = warp_img + mask
              ReferenceNet 输入 = 完整 source photo
```

**复用关系**：同一 source 的 W+ codes 同时用于两个任务；ReferenceNet 特征在两个任务间共享。

### 3.4 梯度路由

```
任务 A (real-ref)
  loss = noise_fg + visible_noise × 1.0 + x0_fg + visible_x0 × 0.5 + LPIPS + GAN
  → backward → 清除 W+ 梯度 → 只更新 Reference K/V/O + PatchGAN

任务 B (novel)
  loss = noise_fg + visible_noise × 0.25 + x0_fg + visible_x0 × 0.1 + W+ contrast
  → backward → 清除 Reference 梯度 → 只更新 W+ mapper + W+ RCA
```

### 3.5 输出契约

```python
# 旧版（已废弃）
final = generated * mask + warp * (1 - mask)

# 当前
final = generated  # SD VAE decode 的完整 512×512 图像
```

warp 和 mask 只作为 BrushNet 条件输入，最终所有像素由 SD 生成。

---

## 4. 核心细节

### 4.1 输入

| 输入 | 来源 | 维度 | 用途 |
|---|---|---|---|
| source photo | 真实人脸数据集 | [B,3,512,512] | W+ 编码、ReferenceNet、warp 源 |
| depth | EG3D 渲染 | [B,1,512,512] | forward warp |
| c_source | 相机参数 | [B,25] | 源相机 |
| c_novel | 相机参数 | [B,25] | 目标相机 |
| W+ codes | GOAE encoder | [B,14,512] | 身份条件 |
| EG3D y_hat_novel | EG3D 渲染 | [B,3,512,512] | novel 伪目标（仅几何） |
| warp_img | Warper | [B,3,512,512] | BrushNet 条件 |
| hole mask | 1 - visibility | [B,1,512,512] | BrushNet 条件 + 区域加权 |

### 4.2 训练目标

| 任务 | 目标 | 域 | W+ | 权重特征 |
|---|---|---|---|---|
| Real-Ref | 同一 source photo | 真实照片 | 禁用 | hole 强 + visible 强 |
| Novel | EG3D y_hat_novel | EG3D 渲染 | 启用 | hole 强 + visible 弱 |

### 4.3 训练参数

```yaml
max_steps: 300000
batch_size: 2
initialization_path: null        # 从零门控适配器
arm_after_first_pass: True       # 安全停止延迟激活
identity_lr: 2e-5               # W+ mapper + W+ RCA
appearance_lr: 5e-6             # Reference K/V/O
discriminator_lr: 2e-5          # PatchGAN
val_interval: 500
save_interval: 2000
```

### 4.4 损失函数

```
L_total = L_noise_fg
        + w_visible_noise × L_noise_bg
        + L_x0_fg (SNR 平衡)
        + w_visible_x0 × L_x0_visible (SNR 平衡)
        + L_pixel (LPIPS + GAN, 仅 real-ref, t<200)
        + 0.05 × L_preserve (W+ known-region anchor)
        + 0.1 × L_contrast (correct W+ vs wrong W+)

其中:
  real-ref:  w_visible_noise = 1.0,  w_visible_x0 = 0.5
  novel:     w_visible_noise = 0.25, w_visible_x0 = 0.1

L_pixel = lpips_w × LPIPS(fake_patch, target_patch)
        + adv_w × GAN_loss(fake_patch)
        + R1 正则 (每 16 步)
```

### 4.5 验证

每 500 步生成一张 `overview_step_XXXXXX.png`：

- 每个身份两行：NOVEL 行和 REAL-REF 行；
- NOVEL 行：source → EG3D target → warp → **SD output (FINAL)**；
- REAL-REF 行：real target → masked → **SD output (FINAL)** → target；
- 图顶标注 full/hole/visible L1；
- `best_model.pt` 按 `real_ref_full` 改善且 `novel_full ≤ 0.090` 选择。

### 4.6 推理

```bash
# configs/infer.yaml
inpainting_backend: diffusion
ckpt_inpaintor: <best_model.pt 路径>

CUDA_VISIBLE_DEVICES=0 python scripts/infer.py
```

推理输出为 SD full-frame，不拼接 warp。

---

## 5. 与原版的关键代码对照

| 功能 | 原版文件 | 当前文件 | 主要差异 |
|---|---|---|---|
| 训练主循环 | `training/coach_inpainting.py` | `training/coach_inpainting_static.py` | diffusion 替换 FFC；factorized 双分支；full-frame 输出 |
| 修补器 | `models/saicinpainting/` | `models/diffusion_inpaintor.py` | SD+BrushNet 替换 LaMa |
| 采样 | 不存在 | `utils/diffusion_inpainting.py` | 50-step DPM-Solver++ |
| W+ 注入 | 不存在 | `models/mapper/w_proj.py` | W+ → SD token |
| Reference 注入 | 不存在 | `models/referencenet/attention_processor.py` | 独立 K/V/O RCA |
| ReferenceNet | 不存在 | `models/referencenet/unet_2d_condition.py` | 多尺度特征提取 |
| 推理入口 | `scripts/infer.py` | `scripts/infer.py` | 双后端：diffusion / legacy_lama |
| 数据集 | `datasets/dataset_inpainting.py` | `datasets/dataset_inpainting_static.py` | 新增 `fixed_novel_view` |
| 配置 | `configs/train_inpainting.yaml` | `configs/train_inpainting.yaml` | 新增 pixel_sup、fullframe_loss、validation |

---

## 6. 当前状态与训练命令

当前配置：从零门控适配器初始化，训练 300000 步。

```bash
cd /data/xzy/WarpGAN

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/xzy/miniconda3/envs/warpgan/bin/python scripts/train_inpainting.py
```

查看结果：

```text
logs/images/val/overview_step_*.png   # 验证总览图
logs/val_metrics.txt                   # 数值指标
checkpoints/best_model.pt              # 自动选择的最佳模型