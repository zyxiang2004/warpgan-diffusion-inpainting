# WarpGAN Inpainting 架构概述与问题诊断报告

> 版本：2026-08-09 | 基于 20260803 代码库 + 原版 WarpGAN 对比分析

---

## 概述

本项目目标是将 EG3D 三维人脸新视角合成中的空洞填补步骤，从原版 WarpGAN 的 CNN inpainting 网络迁移到 Stable Diffusion (SD) pipeline，以利用 SD 预训练的强生成先验。

**试错历程**：

- 原版 WarpGAN 使用 FFC style resnet 直接做像素级 inpainting，配合 L1/MSE/LPIPS/ID/GAN/Cycle 等像素空间损失，以及 EG3D encoder 的 latent consistency loss。该方案 hole L1 数值好，但纹理细节受限于 CNN 容量，整体偏"油画感"。
- 随后迁移到 SD pipeline：冻结 SD 1.5 UNet/VAE，在其 attention 层上挂载 W+ adapter（身份/几何）和 Reference adapter（外观）两个可训练分支，配合 BrushNet 提供空间条件。早期版本从已携带"油画感"的旧 checkpoint 初始化，导致输出材质固化；v5 改为从零门控初始化后该问题消除。
- 探索期尝试了调节损失权重、增减损失项、加入 W+ 对比/保持正则等手段，在好坏配置间反复，未能稳定。
- 当前版本采用 factorized joint 训练：每个 batch 先用 real_ref 自重建数据更新 Reference adapter，再用 novel 新视角数据更新 W+ adapter，两组参数互斥、梯度隔离；real_ref 分支额外加 LPIPS + PatchGAN 像素监督，novel 分支加 W+ preservation/contrastive 正则。

**当前状态**：三个条件模块（WProjModel、Reference adapter、BrushNet）均在运作，SD UNet 主干及 ReferenceNet/BrushNet 冻结。训练日志显示 real_ref 自重建分支收敛良好（hole L1 ≈ 0.035，edge ≈ 0.016），而 novel 新视角分支的 hole L1 停在 ≈ 0.11、edge ≈ 0.027，肉眼可见 hole 区域纹理不连贯的细碎孔洞。

**核心问题**：当前问题集中在 novel 分支，real_ref 是成功的对照组。两者共享同一套冻结 SD 底座，差别只在监督强度——

- **real_ref（成功）**：hole 区有完整监督链——真实照片像素级 target + Reference 恒等对齐 + LPIPS/PatchGAN 像素监督；
- **novel（出问题）**：hole 区同时缺这三项——没有像素级真实 GT、hole latent target 仍来自 EG3D 渲染的平滑 latent、local reference 路径在 hole 区 validity=0 不提供几何对应。

因此这不是"两个分支互相污染"，而是 novel 分支自身的监督缺口。具体哪一项缺口是纹理不连贯的主因，仍需单变量消融确认（见第 6 节）。

---

## 目录

1. [项目核心架构](#1-项目核心架构)
2. [训练流程详解（含代码溯源）](#2-训练流程详解含代码溯源)
3. [EG3D 在当前代码中的角色](#3-eg3d-在当前代码中的角色)
4. [SD 如何被改造成 Inpainting 器](#4-sd-如何被改造成-inpainting-器)
5. [两个分支的语义与问题表现](#5-两个分支的语义与问题表现)
6. [当前代码诊断（事实与待验证解释）](#6-当前代码诊断事实与待验证解释)

---

## 1. 项目核心架构

### 1.1 双分支结构（factorized joint）

当前代码（`training/coach_inpainting_static.py`）使用分解式双分支训练。SD UNet 主干、VAE、ReferenceNet、BrushNet 均冻结；实际可训练的是两组互斥的 adapter 参数。

```
                         real batch
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
       WProjModel    Reference     BrushNet
       (identity)    adapter       (冻结)
            │            │            │
            ▼            ▼            ▼
       W+ codes      真实源照片     warp图+mask
       → SD tokens   多尺度特征     → VAE latent
       attn2残差     attn1残差       BrushNet条件
            │            │            │
            └────────────┼────────────┘
                         ▼
               冻结 SD 1.5 UNet
                         │
                         ▼
                 VAE Decoder → SD output
```

三个条件注入模块的角色：
- **WProjModel（Identity）**：输入数据集中已有的 W+ codes（`[B,14,512]`），映射为 SD 可接收的条件 tokens（`[B,18,768]`）；随后由 `ReferenceAttentionProcessor` 的独立 Q/K/V/O 残差路径注入 UNet 的 cross-attention（attn2）。它不是把 code 再映射到 StyleGAN W+，也不是 ResBlock style modulation。
- **Reference adapter（Appearance）**：冻结的 ReferenceNet 从真实源照片提取多尺度特征；可训练的 Reference K/V/O、global gate 和 local gate 将这些特征注入 UNet 的 self-attention（attn1），负责学习"如何使用真实照片外观信息"。
- **BrushNet（Spatial Condition）**：接收 warp 图像的 VAE latent 与 mask，输出 down/mid/up residual 条件。当前 BrushNet 从预训练权重加载并被冻结，本次训练不更新其参数。

训练策略：两步交替优化（`_train_factorized_joint`）。每步反向传播后，将另一组参数的梯度置空（`forbidden_parameters`）。SD UNet 主干始终不更新。

### 1.2 关键代码入口

| 文件 | 关键函数 | 作用 |
|---|---|---|
| `coach_inpainting_static.py:L596` | `cal_diffusion_loss()` | 主loss计算入口 |
| `coach_inpainting_static.py:L691-L750` | x0 latent + pixel loss | latent空间x0预测loss + real_ref pixel监督 |
| `coach_inpainting_static.py:L1009` | `_optimize_factorized_branch()` | 单分支优化步骤 |
| `coach_inpainting_static.py:L1028` | `_train_factorized_joint()` | 主训练循环 |
| `coach_inpainting_static.py:L1143` | `validate()` | 验证（NOVEL + REAL-REF 50-step full-frame生成） |

### 1.3 可训练参数边界（代码确认）

- **冻结**：SD 1.5 VAE、SD 1.5 UNet 主干、ReferenceNet backbone、BrushNet。
- **identity 组**：`WProjModel` 参数 + cross-attention（attn2）的 W+ RCA adapter（`to_q/k/v/out_wplus` + `wplus_scale`）。
- **appearance 组**：self-attention（attn1）的 Reference adapter（`to_k/v/out_reference` + `reference_scale` + `reference_local_scale`）。
- 两组参数互斥，有 overlap 检查（`_factorized_parameter_partition`）。

### 1.4 训练循环流程

每个 batch 执行两次优化，两组参数交替更新、互不干扰：

```
┌───────────────────────────────────────────────────────┐
│ Step 1: Appearance update（real_ref 自重建数据）       │
│                                                         │
│  real_ref batch ──► forward（禁用 W+）                 │
│       │                                                 │
│       ▼                                                 │
│  cal_diffusion_loss                                     │
│   = noise MSE + x0 latent MSE + LPIPS/PatchGAN         │
│       │                                                 │
│       ▼                                                 │
│  backward ──► 清除 identity 组梯度 ──► optimizer.step  │
│       │                                                 │
│       ▼                                                 │
│  更新对象：Reference adapter（attn1）                   │
│  （可选）PatchGAN discriminator 随后单独更新             │
├───────────────────────────────────────────────────────┤
│ Step 2: Identity update（novel 新视角数据）            │
│                                                         │
│  novel batch ──► forward（启用 W+）                    │
│       │                                                 │
│       ▼                                                 │
│  cal_diffusion_loss                                     │
│   = noise MSE + x0 latent MSE + W+ preservation/contrast│
│       │                                                 │
│       ▼                                                 │
│  backward ──► 清除 appearance 组梯度 ──► optimizer.step│
│       │                                                 │
│       ▼                                                 │
│  更新对象：WProjModel + W+ adapter（attn2）            │
└───────────────────────────────────────────────────────┘

注：SD UNet 主干、VAE、ReferenceNet、BrushNet 在两个 step 中均不更新。
    real_ref 前向禁用 W+，故 identity 参数变化不影响 real_ref 输出；
    novel 前向使用 Reference adapter，故会受 Step 1 更新结果影响（单向耦合）。
```

---

## 2. 训练流程详解（含代码溯源）

### 2.1 损失计算链路

核心函数 `cal_diffusion_loss`（L596-L825）的完整调用链：

```
cal_diffusion_loss(batch)
  │
  ├─ 1. VAE encode target latent
  │     └─ real_ref: 真实照片 → latent
  │     └─ novel (EG3D孔洞填充): warp可见区 + EG3D孔洞区 → composite latent
  │         关键：该 latent 既用于加噪，也作为 noise/x0 loss 的 target
  ├─ 2. 加噪: add_noise(latent, noise, timesteps)
  ├─ 3. SD UNet前向: denoising_unet(noisy_latent, timesteps, ...)
  │     └─ 注入 Reference adapter 外观特征（self-attention 残差）
  │     └─ 注入 WProjModel 条件 tokens（cross-attention 残差，real_ref 禁用）
  │     └─ 注入 BrushNet 空间条件（down/mid/up residual）
  ├─ 4. Noise prediction loss
  │     ├─ loss_noise_fg = MSE(noise_pred, noise) × hole_mask
  │     └─ loss_noise_bg = MSE(noise_pred, noise) × visible_mask × 衰减系数
  │         real_ref可见区权重1.0 / novel可见区权重0.25
  ├─ 5. x0 latent loss ← SNR加权的latent空间监督
  │     ├─ _predict_x0_from_eps → x0_pred (从epsilon预测反推x0)
  │     ├─ loss_x0_fg = _masked_snr_balanced_x0_loss(hole区域)
  │     └─ loss_x0_visible = _masked_snr_balanced_x0_loss(visible区域)
  ├─ 6. Pixel loss (仅real_ref分支) ← 像素空间监督
  │     └─ _pixel_generator_loss: LPIPS + PatchGAN
  │         从原生 32×32 latent patch 解码为 256×256 RGB patch 后计算
  │         仅 batch_kind=='real_ref' 时进入
  └─ 7. W+正则化 (仅novel分支，W+ Mapper训练时)
        ├─ loss_preserve: MSE(noise_pred, noise_pred_without_W+) × visible
        └─ loss_contrast: 正确W+必须优于随机W+（margin=0.05）
```

**关键代码位置**：noise prediction loss 在 L669-L681，x0 latent loss 在 L691-L698，pixel loss 在 L705-L712，W+正则化在 L772-L823。

### 2.2 原版 WarpGAN 的原始损失（对比）

原版 WarpGAN 的 `training/coach_inpainting_static.py` 使用 CNN inpainting 网络和 pixel-space 损失：

```
cal_inpaintor_loss(batch)
  │
  ├─ cal_inpaintor_rec_loss(batch, conf_map=None)
  │     ├─ L1 loss (pixel space, hole/visible加权)
  │     ├─ MSE loss (pixel space, hole/visible加权)
  │     ├─ VGG Perceptual loss
  │     ├─ LPIPS loss
  │     ├─ ID loss (ArcFace人脸识别网络)
  │     ├─ Adversarial loss (PatchGAN判别器)
  │     ├─ Feature matching loss
  │     └─ ResNet perceptual loss
  │
  └─ Latent code loss: EG3D encoder(pred_novel_256) → codes MSE
      （将inpaint结果编码回latent space，与原始latent code比较）
```

**关键差异**：

| 维度 | 原版 WarpGAN | 当前版本 |
|---|---|---|
| 计算空间 | Pixel space（RGB像素） | VAE latent space（noise + x0预测）+ 256px patch |
| 生成器 | 直接CNN inpainting网络（FFC style resnet） | 冻结 SD UNet + 可训练 adapter |
| 判别器 | PatchGAN（判别 pixel-space RGB） | PatchGAN（仅 real_ref；从原生 latent patch 解码出的 256×256 RGB patch） |
| Cycle机制 | forward() 中 `pred_inv_warp` + `pred_inv_pred`（受 `warp_pred` 配置控制） | 无 |
| 身份保持 | ArcFace ID loss + EG3D encoder latent MSE | W+ contrastive + preservation loss |
| EG3D角色 | 提供 W+ codes、渲染图/深度/新视角条件，并参与 latent consistency loss | 继续通过数据提供 W+ codes、深度和 novel pseudo target；不再在 coach 内前向 EG3D encoder 计算 latent loss |

---

## 3. EG3D 在当前代码中的角色

### 3.1 与原版 WarpGAN 的关键区别

原版 WarpGAN 中，EG3D 既参与输入构造（codes 作为 generator 输入、y_hat_novel 作为 hole 填充、depth 用于 warp），又在损失中调用冻结的 EG3D encoder 对预测结果计算 latent consistency loss：

```python
# 原版 L1078-L1084
pred_novel_256 = F.adaptive_avg_pool2d(batch['pred_novel'], (256, 256))
w_pred_novel = self.gan.encoder_forward(pred_novel_256)
latent_value = F.mse_loss(codes, w_pred_novel)
```

该 loss 的语义是"预测的新视角结果重新编码后，其 W+ code 应接近原始 identity code"。但 EG3D encoder 对纹理敏感，这个约束实际上是"纹理+几何"的混合约束，而非纯几何约束。

### 3.2 当前版本：不再前向 EG3D encoder，但离线产物仍参与

当前 `Coach` 中不存在 `EG3D encoder(prediction) → code MSE`。EG3D 的可微 loss 角色已被移除，替换为 W+ preservation loss 与 contrastive loss。但 EG3D 的离线产物仍有三类用途：

1. 数据集提供的 W+ `codes` 输入 `WProjModel`，映射为 SD 条件 tokens；
2. EG3D/反演深度用于 `Warper.forward_warp` 构造 novel warp 和 hole mask；
3. real/novel 分支的 `target` 是 EG3D novel-view pseudo target，并在复合目标中填充 hole 区域。

换言之：**EG3D 不再作为 coach 内部的可微 latent-loss 网络，但其离线产物仍参与条件构造和 novel 监督目标。**

---

## 4. SD如何被改造成Inpainting器

### 4.1 问题：SD本质是"文生图"而非"修补"

Stable Diffusion 的原始训练目标是从噪声+文本描述生成完整图像。它没有内建的 inpainting 能力。将其用于 inpainting 需要解决三个问题：
1. **空间约束**：如何告诉模型"哪里是洞、哪些是已知区域"？
2. **外观一致性**：如何让填充内容与真实照片的风格/纹理/光照一致？
3. **身份保持**：如何确保不同视角下是同一个人？

### 4.2 三个条件注入模块的工作原理

**BrushNet（空间约束）**：

BrushNet 作为预训练条件网络，接收以下输入：
- **warp 图像的 VAE latent**
- **Hole mask**（下采样到 latent 分辨率）

BrushNet 将这些空间信息编码为多尺度特征图，通过 down/mid/up residual 注入冻结的 SD UNet。

在当前训练代码中 BrushNet 从预训练权重加载并被冻结（`coach_inpainting_static.py:L117-L120`）。因此本次训练不会让 BrushNet 的 zero-convolution 从零逐步激活；它始终以既有权重提供空间条件。

**Reference adapter（外观一致性）**：

Reference adapter 的工作方式是将真实参考图特征以残差方式注入 SD UNet 的 self-attention 层（attn1）。

具体来说，`ReferenceAttentionProcessor` 在 self-attention 中增加可训练的残差分支：
- 使用独立的 K/V/O 投影计算 global reference attention residual
- 使用对齐后的 reference feature 计算 local residual
- 两条残差分别由 `reference_scale` 与 `reference_local_scale` 的 `tanh` 门控后叠加到底座输出

这种方式的意义是：SD UNet 在去噪过程中每层的 pixel 特征可以在参考图中"查找"相似的纹理、颜色、光照模式，从而在 hole 区域生成与参考图一致的纹理。

**W+ adapter（身份保持）**：

数据集中的输入已经是 W+ codes。`WProjModel` 将 14 个 512 维 W+ token 映射为 18 个 768 维 SD condition token；随后 W+ adapter 在 UNet cross-attention（attn2）中使用独立 Q/K/V/O 计算残差，并由 `wplus_scale` 门控注入。

对于 inpainting 来说，这条路径的作用是把 W+ 身份/结构先验翻译为 SD attention 能接收的 token 条件。代码将该组参数命名为 identity/geometry 分支；它具体解耦了多少"几何"与"外观"，仅凭代码不能证明。

**W+正则化的双重约束**：当前代码不仅训练 W+ adapter 来改善 hole 区域预测，还通过两个额外的 loss 约束其行为：
- **preservation loss**：W+ 的加入不应改变 visible 区域的噪声预测
- **contrastive loss**：正确的 W+ 必须在 hole 区域产生比随机 W+ 更低的 x0 预测误差

### 4.3 三者协作的完整流程

以一个 novel view inpainting 为例：

1. **输入准备**：
   - 真实源照片 → 冻结 ReferenceNet → 多尺度外观特征
   - 数据集 W+ codes → `WProjModel` → SD condition tokens
   - warp 图像 + hole mask → VAE encode → BrushNet 条件

2. **SD UNet 去噪过程**（训练为单步预测，验证为50步 DPM-Solver++ 采样）：
   - BrushNet 通过 down/mid/up residual 注入空间条件
   - self-attention（attn1）接收 global/local Reference residual
   - cross-attention（attn2）接收 W+ RCA residual（real_ref 禁用）

3. **VAE decode**：最终 latent → VAE Decoder → RGB图像

4. **Loss计算**：latent space 做 noise prediction MSE + x0 prediction MSE，仅 real_ref 分支加 pixel LPIPS/PatchGAN

**关键理解**：BrushNet 提供空间条件，W+ adapter 提供 identity/geometry 条件，Reference adapter 提供真实照片外观条件。三者进入同一个**冻结的** SD UNet，但只有 W+ adapter 与 Reference adapter 参数可训练；不能把问题简单归因于"共享 UNet 被两个分支同时改写"。

---

## 5. 两个分支的语义与问题表现

### 5.0 为什么有两个分支？（前置理解）

训练代码中的 `_train_factorized_joint`（L1028）在一个 batch 内执行两次优化：一次 appearance update，一次 identity update。这两次优化分别使用**不同的数据构造**（appearance 用 `_parse_reference_batch`，identity 用 `_parse_real_batch`），因此产生了两个分支：

| 分支名 | 数据构造方式 | batch_kind | target（监督目标） | W+ Mapper |
|---|---|---|---|---|
| **real_ref**（自重建） | 真实照片 + 自身 warp 产生的 hole mask → 真实照片本身作为 target | `'real_ref'` | 真实照片（像素级 ground truth） | 禁用（`disable_wplus=True`） |
| **novel**（新视角） | 真实照片 warp 到新视角；EG3D novel render 作为 `target`，复合目标再构造 composite latent | `'real'` | composite：warp real visible + EG3D hole | 启用 |

**real_ref 分支的核心语义**：让模型学会"给定真实照片的 hole shape，重建同一张真实照片"。因为 target 就是真实照片，所以这里的 hole 区域有像素级 ground truth——可以安全使用 LPIPS 和 PatchGAN（L709-L710）。W+ 被禁用，因为自重建不需要新视角几何映射。

**novel 分支的核心语义**：让 identity/W+ adapter 学习在新视角 warp 条件下利用 W+ codes。其 hole target 来自 EG3D novel render，不是真实照片 ground truth，因此默认不启用 LPIPS/PatchGAN。W+ 被启用并映射为 SD condition tokens。

**为什么区分两个分支**：代码需要将两种不同可信度的监督分别交给两组 adapter。real_ref 有精确真实照片 target，适合训练 appearance；novel 有新视角结构信息但 hole target 是 EG3D pseudo target，适合训练 identity/geometry。区分分支的直接目的，是避免 EG3D RGB 梯度更新 Reference appearance 参数，也避免 real_ref 自重建更新 W+ 参数。

**实际参数关系**：两者共享冻结底座及同一套前向条件模块，但可训练参数组互斥。novel step 清除 appearance 梯度，real_ref step 清除 identity 梯度；并且 real_ref 前向禁用 W+。因此代码不支持"novel 梯度直接把 real_ref 参数拉向 EG3D"的说法。反方向上，real_ref 更新后的 Reference adapter 会用于后续 novel 前向，所以 appearance 对 novel 存在单向耦合。

### 5.1 当前观察到的现象（两个分支的输出差异）

训练过程中，两个分支的输出质量出现明显分化：

| 输出类型 | 输入条件 | 表现 |
|---|---|---|
| **REAL-REF SD output** | real_ref 分支（真实视角自重建） | 质量好、收敛快（hole L1 ≈ 0.035，edge ≈ 0.016，稳定下降） |
| **NOVEL SD output** | novel 分支（EG3D 新视角） | hole 区域纹理不连贯、出现细碎孔洞（hole L1 ≈ 0.11，edge ≈ 0.027，平台期） |
| **Step 0（两者）** | 同上 | 纹理连续、色彩自然——说明初始化 checkpoint 本身没问题 |

这个分化说明 real_ref 的监督机制是有效的，问题集中在 novel 分支。real_ref 拥有真实照片像素级 GT + Reference 恒等对齐 + LPIPS/PatchGAN 三重保障；而 novel 在 hole 区缺少这些。

### 5.2 从 step 0 到后续训练：代码能确认什么

1. W+、Reference global 与 Reference local gate 均以 0 初始化，因此 step 0 时自定义 adapter residual 为 0，注意力输出与冻结 SD 底座等价（`attention_processor.py:L30,L45,L49,L134,L161,L187`）。
2. 训练后 gate 与对应投影参数开始更新，输出会逐渐偏离初始化底座。
3. real_ref step 更新 Reference adapter；novel step 更新 WProjModel 与 W+ adapter。SD UNet 主干、ReferenceNet、BrushNet 不更新。
4. real_ref 不使用 W+，所以 real_ref 的质量只取决于 Reference adapter 自身及其监督链，不受 novel/W+ 参数变化影响——这与日志中 real_ref 稳定收敛一致。
5. novel 同时使用 W+ 与 Reference，因此 novel 的输出受两组 adapter 共同影响；其 hole 区纹理问题可能来自 W+ adapter、Reference adapter（由 real_ref 学到但 hole 区无几何对应）、或 EG3D latent target 中的任一环节，仅看最终图无法确定责任模块。

这条时间线能够解释"为什么 step 0 两者都正常、训练后只有 novel 出问题"，但不能单独证明 novel 纹理破碎的具体成因。

---

## 6. 当前代码诊断（事实与待验证解释）

本节将"代码事实"和"机制解释"分开。代码可以确认监督如何流动，但视觉异常的具体成因仍需消融或日志证据支持。

### 6.1 Hole区域的x0 latent MSE — L691-L698

**代码做什么**：从 epsilon 预测反推 x0_pred（L691 `_predict_x0_from_eps`），然后对 hole 区域和 visible 区域分别与 target latent 做 SNR 加权的 MSE loss（L693-L698 `_masked_snr_balanced_x0_loss`）。

**可以确认**：训练直接优化单步随机 timestep 下的 latent x0 MSE，而最终验证是从纯噪声开始的 50-step DPM-Solver++ 完整采样。训练代理目标下降不等于最终像素纹理一定改善。

**合理但尚未证实的解释**：latent L2 可能偏向平滑、低频解，尤其当 target 本身缺少高频细节时（novel 的 hole target 来自 EG3D 渲染，天然平滑）；这与已有记录中"L1 改善但观感未同步改善"一致。但代码本身不能证明 x0 MSE 必然导致 novel 纹理破碎，更不能推出特定的"纹理块胜出"机制。

### 6.2 Pixel loss仅对real_ref生效 — L705-L712

**代码做什么**：

```python
pixel_domain_ok = False
if self.use_pixel_sup:
    real_pair_only = bool(getattr(self.pixel_sup, 'real_pair_only', True))
    pixel_domain_ok = (not real_pair_only) or batch['batch_kind'] == 'real_ref'
if pixel_domain_ok:
    loss_pixel, px_metrics, ... = self._pixel_generator_loss(...)
```

默认 `real_pair_only=True`，所以 `batch['batch_kind'] == 'real_ref'` 是进入 pixel loss 的唯一条件。novel 分支的实际 `batch_kind` 是 `'real'`，因此不计算 pixel loss。

**可以确认**：novel 分支没有可训练的 LPIPS/PatchGAN 信号；其中 `monitor_lpips` 只是每 100 step 的 detached 监控值，不加入 loss。novel 主要由 noise/x0 target 与 W+ preservation/contrastive 约束。

**边界**：这直接解释了 novel 分支 hole 区缺少真实感监督的来源。real_ref 恰好有完整的 pixel loss，因此 real_ref 收敛良好；问题只在 novel。需要注意的是，novel 只有 `monitor_lpips`（detached，不回传梯度），real_ref 才有可训练的 `px_lpips`——两者在日志中不可直接对比。

### 6.3 交替优化中的参数流 — L1028-L1069

**代码做什么**：

```python
# L1057-L1062: Appearance update（real_ref, 清除 identity 梯度）
appearance_loss, ... = self._optimize_factorized_branch(
    self._parse_reference_batch(batch_tuple),  # → batch_kind='real_ref'
    forbidden_parameters=identity_parameters,
)
# L1066-L1069: Identity update（novel, 清除 appearance 梯度）
novel_loss, ... = self._optimize_factorized_branch(
    self._parse_real_batch(batch_tuple),  # → batch_kind='real'
    forbidden_parameters=appearance_parameters,
)
```

**可以确认的参数流**：

- appearance step：更新 Reference adapter；identity/W+ 组梯度被清除；冻结 SD UNet 主干不更新。
- identity step：更新 WProjModel + W+ attention adapter；Reference adapter 梯度被清除；冻结 SD UNet 主干不更新。
- real_ref 前向设置 `disable_wplus=True`，所以 identity 参数变化不影响 real_ref 输出。
- novel 前向使用 Reference 与 W+ 两组条件，所以它会受到 appearance step 最新参数状态的影响。

因此实际耦合是**不对称且主要为 appearance → novel**：real_ref 更新的 Reference adapter 会影响后续 novel 前向，但 novel 的梯度不会反过来更新 Reference adapter。两个分支之间不存在双向污染。

### 6.4 Novel分支的EG3D孔洞填充复合目标 — L609-L614

**代码做什么**：

```python
composite_rgb = batch['warp_img'] * (1.0 - batch['mask']) + batch['target'] * batch['mask']
target_latents = self.vae.encode(composite_rgb * 2.0 - 1.0).latent_dist.sample() * 0.18215
```

代码注释称 composite latent "only for noise scheduling" 且"不对 EG3D pixels 计算 loss"，但实际数据流是：`target_latents` 先用于 `add_noise`，随后也进入 noise target 对应的去噪任务，并直接作为 `loss_x0_fg/loss_x0_visible` 的比较目标。因此，**novel hole 的 EG3D latent 确实参与训练监督**。

这能支持的结论是：novel identity 分支仍在追踪 EG3D hole 域，而不是完全摆脱 EG3D RGB。EG3D 渲染本身平滑、缺乏高频纹理，以它的 latent 作为 hole 监督目标，可能是 novel hole 区纹理不连贯的原因之一。由于 novel 梯度不能更新 Reference adapter、且 real_ref 禁用 W+，这条链路只影响 novel 分支自身，不波及 real_ref。

### 6.5 目前能够成立的诊断边界

| 现象 | 代码直接支持的解释 | 代码尚不能证明的解释 |
|---|---|---|
| step 0 两者正常、训练后只有 novel 出问题 | zero gate 初始化；adapter 随训练开始影响冻结底座；real_ref 监督链完整故保持良好 | 某一个 gate 必然导致特定视觉异常 |
| real_ref 收敛良好 | 真实照片 target + Reference 恒等对齐 + LPIPS/PatchGAN 三重监督；W+ 禁用不受 novel 影响 | — |
| novel hole 区纹理不连贯/细碎孔洞 | novel 无 pixel loss；hole latent target 来自 EG3D（平滑）；local reference 在 hole 区 validity=0 无几何对应 | 仅凭最终图判定是 W+ adapter、Reference adapter 或 x0 MSE 单独造成 |
| L1 改善但观感不改善 | 训练代理目标（单步 latent MSE）与 50-step 采样感知质量不完全一致 | L1 下降即可证明训练方向正确或错误 |

**结论**：当前问题集中在 novel 分支，而非两个分支互相污染。real_ref 的成功反过来印证了"完整监督链"的有效性；novel 的 hole 区正是因为缺少同等监督——没有像素级真实 GT、没有局部 reference 几何对应、hole latent target 仍来自平滑的 EG3D 渲染——才出现纹理不连贯。具体哪一项是主因仍需单变量消融确认，报告不将其写成已证实结论。

---

## 附录：关键代码文件索引

| 文件路径 | 关键内容 |
|---|---|
| `training/coach_inpainting_static.py` | 主训练逻辑、因子化参数划分、noise/x0/pixel loss、W+正则化 |
| `configs/train_inpainting.yaml` | 损失权重、训练超参数 |
| `models/mapper/w_proj.py` | W+ codes `[B,14,512]` → SD condition tokens `[B,18,768]` |
| `models/referencenet/attention_processor.py` | 冻结 attention 底座上的 W+ RCA 与 Reference global/local 残差分支 |
| `原版 WarpGAN training/coach_inpainting_static.py:L1054-L1089` | 原版 latent consistency loss 实现（cal_inpaintor_loss） |
| `原版 WarpGAN training/coach_inpainting_static.py:L1118-L1212` | 原版 rec loss 实现（L1/LPIPS/perceptual/ID/GAN） |