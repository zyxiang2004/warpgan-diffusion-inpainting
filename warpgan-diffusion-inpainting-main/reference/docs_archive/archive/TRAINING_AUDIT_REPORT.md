# WarpGAN SVINet 训练审计报告

生成日期：2026-07-15；v3 复审：2026-07-22；v4 复审：2026-07-28  
历史实验目录：`experiments/train_inpaintor/[20260714-215729]_rca_v2_novelTarget_synth50/`  

> **v3 最终复审说明**：本文 20k–35k 的数值属于历史 v2/mirror-ref 实验，保留用于追踪问题。最终训练方案已改为 `architecture_version=3`：多尺度 ReferenceNet 特征、严格配对的 real-ref self-reconstruction，以及最大空洞覆盖窗口的原生 latent patch LPIPS/PatchGAN（带 VAE 解码上下文）。旧 v2 checkpoint 可推理，但不可作为 v3 optimizer resume。

> **v4 最终复审说明**：Reference self-attention 已从“共享冻结 K/V/O + trainable gate”升级为独立可训练 K/V/O residual adapter。v3 checkpoint 只允许 function-preserving 初始化 v4，不允许 optimizer resume。当前推荐的研究 checkpoint 是 `experiments/_v4b_factorized_pilot_500/checkpoints/iteration_200.pt`；连续 alpha-bar 像素权重已完成公平对照，但不占 real-ref/novel Pareto 前沿，默认仍使用 `hard_low_noise (t<200)`。

> **52k 恐怖谷事故说明**：`[20260728-102109]_rca_v4a_reference_pretrain` 并非续 optimizer，
> 而是从 v3 300k 初始化后执行了错误的 300k appearance-only 计划。novel L1 从 step 0
> 的 `0.101610` 恶化到 step 54k 的 `0.210880`。该实验只能作为负结果，禁止作为默认续训起点。

---

## 一、整体网络架构

### 1.1 SVINet 组件总览

SVINet 是本仓库的新视角修补网络。它由五个协同工作的模块组成，其中三个为冻结的预训练模型，两个为可训练模块：

| 模块 | 类型 | 角色 | 可训练 |
|------|------|------|--------|
| **VAE** | SD 1.5 VAE | 图像↔latent编解码 | 否（冻结） |
| **BrushNet** | 官方预训练 | 结构性修补引导（warp+mask → UNet残差） | 否（冻结） |
| **冻结 UNet** | SD 1.5 UNet | 核心去噪网络，承载 RCA/ReferenceNet 分支 | 否（主体冻结） |
| **ReferenceAttentionProcessor** | 自定义 | 负责向 UNet 注入 W+ 身份和 ReferenceNet 纹理 | **是（gate + Q/K/V/O 投影）** |
| **WProjModel (W mapper)** | 自定义 | 将 [B,14,512] EG3D W+ latent 映射为 768-dim token 序列 | **是** |
| **ReferenceNet** | SD 1.5 UNet（只用 down+mid 部分） | 从源图提取多尺度空间特征（320/640/1280 通道） | backbone 冻结；独立 Reference K/V/O + gate 可训练 |

### 1.2 数据流

```
Source image (512×512, [0,1])
       │
       ├─── VAE encode (mode) ──→ ReferenceNet.extract_features()
       │         └──→ ref_feat = {channel: [matching spatial scales]}
       │
       └─── Warper.forward_warp(depth, c_src → c_novel) ──→ warp_img
                                                              │
                                            ┌─────────────────┴─────────────────┐
                                       VAE encode (sample)                  mask
                                       masked_image_latent              mask_latent
                                            └─────────────────┬─────────────────┘
                                                    brushnet_cond = cat([latent, mask], 1)
                                                              │
                                                     BrushNet (frozen)
                                                    down/mid/up residuals
                                                              │
                                            noisy_latent + timestep
                                                              │
                                                   frozen UNet (attn replaced)
                                                        attn1 (self-attn):
                                                          W+ gate=0 ──→ 冻结 SD self-attn
                                                          + ref_feat 门控残差 (reference_scale)
                                                        attn2 (cross-attn):
                                                          冻结文本分支 (empty_prompt)
                                                          + W+ 独立 Q/K/V/O 门控残差 (wplus_scale)
                                                              │
                                                         noise_pred
                                                              │
                                              DDPM 加噪训练 / 50-step DPM-Solver++ 验证推理
                                                              │
                                                         composed output
                                                   (hole: generated, known: warp_img)
```

### 1.3 ReferenceAttentionProcessor 详解

每个 UNet attention layer 的 processor 被替换为自定义 `ReferenceAttentionProcessor`：

- **Cross-attention (attn2) 层**：拥有独立的 `to_q_wplus`, `to_k_wplus`, `to_v_wplus`, `to_out_wplus`，均从冻结 UNet 对应投影复制初始化。通过 `wplus_scale = tanh(raw_scale) → [−1,1]` 门控，门初始为 0（保持底座不变）。
- **Self-attention (attn1) 层**：v4 使用独立 `to_k_reference`, `to_v_reference`, `to_out_reference`，从冻结 self-attention K/V/O 复制初始化，并通过 `reference_scale = tanh(raw_scale)` 作为 residual gate。v3→v4 初始化严格复现原函数。
- **零门控初始化**：替换 processor 在 step 0 与原始 SD `AttnProcessor2_0` 输出完全一致（数值误差 < 1e-8），因此不影响 BrushNet 预训练。

### 1.4 训练目标

Phase-A (steps 0–20000)：

- **Real batch**（25%）：源图 warp → `c_novel`，监督目标 `y_hat_novel`（EG3D 渲染的 novel view 伪真值）
- **Synthetic batch**（50%）：精确对偶配对数据，监督目标 `target_img`
- 误差：空洞区域前景噪声 MSE（主）+ SNR 平衡 x0 MSE（辅）+ W+ 对比约束

当前 v3 任务周期：

- **Real batch**（25%）：same as above
- **Synthetic batch**（50%）：same as above
- **Real-ref self-reconstruction**（25%）：使用 novel warp 产生的 disocclusion mask 在源照片上挖洞，目标为同一张源照片；**W+ 被禁用**，仅结构与 ReferenceNet 纹理分支学习。输入/目标严格同坐标系。

损失权重：
```
loss = loss_noise_fg + 0.1 × loss_noise_bg
     + loss_x0_fg (SNR 平衡)
     + LPIPS + PatchGAN (仅 real-ref、t<200、原生 latent patch)
     + 0.05 × loss_preserve (背景 W+ 不破坏)
     + 0.1 × max(0, 0.05 + loss_x0_correct − loss_x0_wrong)  (身份对比)
```

---

## 二、遇到的问题及处理方式

### 2.1 [根本问题] 目标视角错配（0–20k 失效来源）

**问题**：旧代码中，warp_img 是按 `c_novel` 投影的新视角结果，但训练目标却是源视角原图 `batch['x']`；这让 UNet 同时面对"新视角几何"和"源视角内容"的矛盾监督。

**证据**：同一真实样本上，warp 与 y_hat_novel（目标）的可见区域 MAE 约 0.055，而与源图 x 的 MAE 约 0.097；目标选错后，两者冲突，网络无法收敛到任何物理上一致的解。

**修复**：`batch['x']` → `batch['y_hat_novel']`（索引 13），synthetic 使用 `target_img`（索引 7）。

**对旧 checkpoint 影响**：0–20k 的 `w_mapper_state_dict` 和 `rca_state_dict` 均学习了矛盾目标，不建议续训；已在 20k 处换目标后重启，参数结构相同可原位加载。

---

### 2.2 [结构问题] 旧 RCA 没有独立 Q/O，是简单的 K/V 注入

**问题**：旧 `ReferenceAttentionProcessor` 只有 `to_k_wplus`, `to_v_wplus`，查询 Q 共享冻结底座的 Q，输出也经由底座 `to_out`；这不是一个真正独立的残差交叉注意力（RCA），而是向文本 K/V 追加 W+ K/V。

**修复**：新 processor 增加 `to_q_wplus`, `to_out_wplus`，完成独立的 Q/K/V/O 路径；并加入 `initialize_from_attention()` 从冻结底座复制初始权重，避免突变引入噪声。

---

### 2.3 [验证问题] 官方 BrushNet 采样不注入背景 latent

**问题**：旧验证在每个 DDIM 时间步内部把干净背景 latent 注入高噪声状态，偏离 BrushNet 官方采样分布（官方做法是从纯噪声全局采样，BrushNet residual 约束结构，最终在像素空间合成已知区）。

**修复**：验证改为纯噪声初始化 `* scheduler.init_noise_sigma`，50步采样后才在像素空间做 `gen × mask + warp_img × (1-mask)`，并提取到独立共享函数 `utils/diffusion_inpainting.py`。

---

### 2.4 [推理断链] scripts/infer.py 使用旧 LaMa 网络

**问题**：正式 `infer.py` 的 `inpainting_backend='inpaint'` 分支调用 `make_generator()` 实例化 FFC-ResNet，期待 checkpoint 中的 `inpaintor_state_dict`。当前训练完全不存储这个 key；因此完成训练后直接推理会加载失败。

**修复**：新增 `inpainting_backend: diffusion`（默认）和 `legacy_lama` 两种模式；diffusion 后端实例化 `DiffusionInpaintor`，读取 `w_mapper_state_dict + rca_state_dict`；同时修复了 yaw/random 模式中的未初始化引用。

---

### 2.5 [ReferenceNet 失效] 训练前 20k 无纹理梯度

**问题**：在真实 novel-view 任务和合成配对任务中，目标图像都可以由 W+ 充分解释（因为这些目标本来就是 EG3D 用 W+ 渲染的）。因此任务中没有"只有源图高频纹理才能还原的信息"；ReferenceNet gate 受梯度更新但贡献趋零（20k 因果消融：no-ref vs full 的 hole_L1 差异只有 0.001）。

**历史方案**：v2 曾使用 mirror-ref（水平翻转照片 + 镜像相机）。它能增大 Reference gate，但水平翻转照片并不等于非对称三维人脸的精确镜像视角，因此 LPIPS 会混入几何标签噪声。

**v3 修复**：改为 real-ref self-reconstruction。novel warp 只提供真实任务形状的空洞 mask；mask 应用于源视角照片，目标仍为同一源照片。W+ 被禁用，ReferenceNet 必须恢复真实纹理，同时保持严格像素配对。

**多尺度修复**：ReferenceNet 旧实现仅按通道数存特征，1280 通道的 16×16 特征会被 8×8 mid 特征覆盖。v3 改为每个通道保存多个空间尺度，注入时按当前 query token 数选择最近尺度。

---

### 2.6 [损失问题] 旧 x0 MSE 在高噪声时间步爆炸

**问题**：`x0_pred = (noisy - sqrt(1-ᾱ)·ε) / sqrt(ᾱ)`；raw MSE(x0_pred, x0_target) ≈ `(1-ᾱ)/ᾱ · MSE(ε_pred, ε)`。在 t=999 时，该因子为 214，造成随机 timestep 引起的 loss 极度波动（旧日志中 loss 从 0.07 到 3.5 不等）。

**修复**：改用 SNR 平衡权重：`loss = ᾱ × (per_sample_x0_error / hole_area_size)`，相当于 `(1-ᾱ) × epsilon_error`，上界为 1，各 timestep 贡献平稳。数学验证：`alpha=[0.999, 0.276, 0.00466]` → raw = `[0.00085, 2.62, 213.6]` vs balanced = `[0.00085, 0.724, 0.995]`。

---

### 2.7 [环境问题] GOAE/Swin CPU tensor 设备不一致

**问题**：`swin_transformer.py` 中 `torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1./0.01)))` 每次前向都临时构造 CPU tensor 作为 `max` 参数，与 GPU 参数冲突。

**修复**：改为 `math.log(100.0)` 标量，数值完全相同，无设备歧义。

---

### 2.8 [可观测性问题] 旧日志仅记录 real batch，synthetic/real-ref 不可见

**修复**：日志条件改为 `global_step % 100 in (0, 1, 2)`，覆盖一个完整的 phase-0/1/2 周期，每组 real + synth + real_ref 均记录；TensorBoard 按 batch kind 分命名空间记录 LPIPS/GAN/R1。

---

### 2.9 [验证随机性] novel view 选取在进程重启后变化

**问题**：`dataset_inpainting_static.py` 的 `_load_raw_image()` 每次调用都 `random.randint(1,3)` 选择 novel view。即使 test loader `shuffle=False`，重启后第一个 batch 抽到的视角不同，验证曲线跨 checkpoint 不可比。

**修复**：`ImageFolderDataset` 新增 `fixed_novel_view: Optional[int]` 参数；训练集保持 `None`（随机），测试集设为 `fixed_novel_view=1`，并将 test loader 改为 `shuffle=False`。

---

## 三、量化收敛记录

### 3.0 v4 结构与目标审计结果

1. CPU 数学测试：v3 formula 与 v4 初始化输出最大绝对误差 `0`；zero gate 误差 `0`。
2. GPU 完整 smoke：80/80 个 Reference 参数张量有非零梯度；appearance pretrain 中 W+ 全部无梯度；checkpoint 可严格生产加载。
3. 完整硬门控路径峰值显存约 `13.36 GiB`；连续 alpha-bar 对所有样本建立 pixel graph 时约 `18.06 GiB`。
4. v4-A 1000 将 real-ref L1 从 `0.23990` 降到 `0.15677`；no-Reference 仍为 `0.24284`，构成明确因果证据。
5. v4-B 显式梯度路由在 step 200 得到 `real-ref=0.15730, novel=0.10533`，当前为最佳 Pareto。
6. v4-C 公平 resume 对照在 step 400 得到 `real-ref=0.19386, novel=0.09735`；虽然 novel 更好，但 appearance 回退，不采用为默认。
7. 长期 appearance-only 对照在 52k 后产生恐怖谷和 novel 崩坏，证明 appearance capability
   必须由 novel identity/geometry 分支约束，不能独立长训。

### 3.0.1 新的训练安全边界

- 正式 `Coach.train()` 已实现 factorized 双分支，不再依赖 pilot 脚本；
- 同一 batch 的 real-ref loss 只更新 Reference，novel pseudo-target loss 只更新 W+；
- appearance-only 超过 2000 步直接报错；
- 跨训练目标禁止 optimizer resume，必须使用 `initialization_path`；
- 可选择继承 v4 PatchGAN 权重/R1 counter，但重建低学习率 optimizer；
- 训练开始前保存零更新 baseline；
- 每 500 步同时验证两种任务；只有满足 novel L1 约束的 real-ref 改善才写入 `best_model.pt`；
- 连续四次 novel L1 > 0.115 自动停止。

### 3.0.2 Full-frame 输出修正（2026-07-29）

复审确认旧采样器返回两个结果：hard-composite 和 SD raw frame；正式推理曾错误地选择
hard-composite。由于 warp 可见区并非真实 target 的可靠逐像素观测，该行为会隐藏 SD 的
整图问题，也阻止模型修复可见区的投影瑕疵。

现已统一为：

- `sample_brushnet_inpainting()` 只返回 SD full-frame；
- warp/mask 只作为 BrushNet 条件；
- `DiffusionInpaintor.forward()` 和 `scripts/infer.py` 直接输出 full-frame；
- pixel LPIPS/GAN 直接作用于 SD 生成 patch；
- real-ref visible 区使用 `noise=1.0, x0=0.5` 的完整帧监督；
- novel visible 区只使用 `noise=0.25, x0=0.1` 的弱结构监督；
- PatchGAN 因输入分布改变而重新初始化，不继承 hard-composite 判别器；
- validation 同时记录 `full/hole/visible`，不再用人为为 0 的 composed known-L1；
- 每个 step 只保存一张 `overview_step_*.png`；
- 新 checkpoint 写入 `output_contract=sd_full_frame_no_composite`，旧契约 checkpoint 只能
  作为 `initialization_path`，不能直接恢复 optimizer。

下一结构审计目标是 target-aligned Reference feature warp。当前 source feature 与 target query 不在同一坐标系，
是比 sampler、训练步数或 camera token 更直接的剩余瓶颈。

### 3.1 Phase-A (0–20k)，目标有缺陷

| 指标 | step 0 | step 10k | step 20k |
|------|--------|---------|---------|
| val hole_l1 (固定样本) | 0.515 | ~0.191 | ~0.169 |
| warp hole_l1 (baseline) | 0.566 | 0.566 | 0.566 |
| improvement | +0.051 | +0.375 | +0.397 |
| W+ gate (mean) | 0.000 | −0.007 | −0.007 |
| Ref gate (mean) | 0.000 | +0.0006 | +0.0005 |

注：虽然图像改善，但 Ref gate 几乎未动（ReferenceNet 功能性为零）；这属于有效收敛，W+ 提供了主要身份信息。

### 3.2 Phase-B (20001 起，目标/损失/任务均修正)

| 指标 | step 20k (过渡) | step 21k | step 30k | step 34k |
|------|----------------|---------|---------|---------|
| val hole_l1 | 0.169 | 0.160 | 0.096 | 0.108 |
| improvement | +0.397 | +0.406 | +0.129 | +0.117 |
| Ref gate max |  | +0.003 | | +0.085 |

Phase-B 初期验证样本改变（固定到 view 1），所以 20k→21k 的基线从 0.566 变为 0.225；在相同基线上，hole_l1 从 21k 的 0.160 降到 30k 的 0.096，至 34k 略有波动（0.108），尚未平台化。

### 3.3 因果消融（step 20k checkpoint）

| 条件 | hole_L1 |
|------|---------|
| full (W+ + ReferenceNet + BrushNet) | **0.152** |
| wrong W+ (另一张脸的 W+) | 0.226 |
| no W+ | 0.376 |
| no Reference | 0.153 |
| BrushNet only | 0.375 |

**历史结论**：W+ 是 v2 的主要改善来源；ReferenceNet 在 phase-A 结束时贡献可忽略（约 0.001）。mirror-ref gate 扩大结果仅证明梯度存在，不证明其监督几何正确；v3 需重新训练并重新做因果消融。

---

## 四、收敛展望

- 当前有效步数约 35k（20k phase-A + 15k phase-B）
- Phase-B 的 hole_L1 从 0.160 降到约 0.096，降幅约 40%，尚未平台化
- 预计 75–100k 步后 hole_L1 稳定；具体参考 phase-A 同等降幅所需的约 10k 步

建议观测指标（每 5000 步）：
1. `hole_l1` 是否继续下降
2. `Ref Gate max` 是否持续扩大（ReferenceNet 逐渐生效）
3. 当 `hole_l1 < 0.05` 且连续 5k 步波动 < 0.01 时可停止

---

## 五、模块分类：Active / Frozen / Ghost

### 5.1 Active（真正训练的参数）

| 模块 | 参数数量 |
|------|---------|
| WProjModel (w_mapper) | 1,967,616 |
| ReferenceAttentionProcessor (cross-attn: wplus_scale, Q/K/V/O) | ~43,962,574 |
| ReferenceAttentionProcessor (self-attn: Reference K/V/O + gate) | ~37,183,696 |

### 5.2 Frozen（加载并参与前向，不更新）

- SD 1.5 VAE
- SD 1.5 frozen UNet（主体权重）
- BrushNet
- ReferenceNet（提取特征的网络权重）

### 5.3 Legacy-only（配置文件中存在，当前 coach 不读取）

`configs/train_inpainting.yaml` 中以下字段对 diffusion coach 无效，仅 `gen_synthimg.py`, `gen_novelview.py`, `infer.py (legacy_lama 模式)` 使用：

- `generator.*`（FFC-ResNet 架构参数）
- `discriminator.*`（当 `losses.pixel_sup.enable=False` 时才是 legacy-only）
- `losses.l1/mse/perceptual/adversarial/feature_matching/resnet_pl/latent/depth/mirror.*`
- `optimizers.discriminator`（当前由 pixel supervision 使用）

这些字段不会造成训练错误，但会误导新贡献者认为对抗训练和 FFC generator 正在运行。

---

## 六、操作命令

### v4 初始化 / v4 checkpoint 续训

```bash
# 用 initialization_path 指向 v3/v4 checkpoint，只加载模型权重初始化 v4；
# 用 checkpoint_path 仅严格 resume architecture_version=4 checkpoint。
CUDA_VISIBLE_DEVICES=0 python scripts/train_inpainting.py
```

当前默认命令会在新时间戳目录中，从上一轮 factorized step-9500 best 权重初始化
full-frame 训练，并重建 optimizer。
不要将 `checkpoint_path` 指向 52k appearance-only checkpoint；如果旧进程仍在运行，应先停止。

### 正式推理（diffusion 后端）

```bash
# 先将测试图放入 ./data/test_img/（参见 README 预处理步骤）
# 修改 configs/infer.yaml 中的 ckpt_inpaintor 为目标 checkpoint
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py
```

### 遗留 LaMa 推理（需要旧论文 checkpoint）

```yaml
# configs/infer.yaml
inpainting_backend: legacy_lama
ckpt_inpaintor: ./pretrained_models/inpaintor/inpaintor.pt
data:
  batch_size: 4
```

---

## 七、关键文件变更摘要

| 文件 | 核心变更 |
|------|---------|
| `models/referencenet/attention_processor.py` | 完整独立 Q/K/V/O + 零初始化门控 + 从冻结 UNet 复制初始权重 |
| `models/diffusion_inpaintor.py` | 生产级推理封装，兼容 v2 单尺度与 v3 多尺度 checkpoint |
| `utils/diffusion_inpainting.py` | 共享 BrushNet/DPM-Solver++ 采样函数，训练验证与正式推理共用 |
| `training/coach_inpainting_static.py` | v4 appearance/factorized 模式、显式梯度职责、hard/continuous pixel objective、v4 resume 边界 |
| `datasets/dataset_inpainting_static.py` | `fixed_novel_view` 参数使测试集 novel view 固定 |
| `scripts/infer.py` | diffusion/legacy_lama 双后端、修复 yaw/random 模式、跳过 diffusion 不需要的 EG3D novel render |
| `scripts/train_inpainting.py` | Resume 时原位续写实验目录而非创建新目录 |
| `models/goae/swin_transformer.py` | 修复 GOAE/Swin logit_scale CPU tensor 设备不一致 |
| `configs/train_inpainting.yaml` | v4 初始化配置；pixel_sup 仅作用于严格 real-ref 配对，默认 hard_low_noise |
| `configs/infer.yaml` | inpainting_backend=diffusion；batch_size=1 |

---

## 八、v5 专项审计：油画感与眼部结构退化

2026-07-30 的 fixed overview 复审确认：姿态已经正确，但 SD raw 对整帧的重绘会损坏原 warp 中较自然的睫毛、眼睑、瞳孔边缘、发丝与皮肤纹理。进一步代码追踪发现 target-aligned Reference 虽在训练前向中被计算，却没有同时满足“真实照片监督可训练、固定验证传参、正式推理传参”三个条件，属于结构存在但功能断链。

v5 将该路径改成同一适配器的等变训练：real-ref 以恒等 feature 对应提供严格真实照片监督，novel-view 以 3D warp 后的 feature 提供目标坐标条件。最终输出采用高置信可见核心区的软数据一致性，而不是旧式全可见区 hard composite；同时保留 raw SD 输出并以 raw 指标选择 checkpoint。新增图像梯度误差与真实 patch 梯度损失，专门覆盖 L1 无法充分反映的眼部边缘和微纹理退化。

### 3.0.3 v4-fullframe 失败与从零重启（2026-07-30）

v4-fullframe 10k 实验证实：指标改善不等于视觉真实感改善。real_ref_full 从 0.099 降至 0.081，但油画感与修改前一致。

根因：初始化权重已携带油画域。所有先前 checkpoint（v3/v4-A/v4-B/v4-factorized-safe）在 EG3D 平滑目标上训练了数万步，条件模块已固化到该域。

决策：`initialization_path: null`，从零门控适配器训练 300000 步。`arm_after_first_pass: True` 保证从零训练初期不会被安全停止误终止。之前停在 10000 步是因为 `max_steps: 10000`，非安全停止触发。
