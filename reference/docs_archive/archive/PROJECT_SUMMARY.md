# WarpGAN SVINet 项目总结文档

生成日期：2026-07-21；v4 复审：2026-07-28  
作者：研发团队（记录来源：训练日志 + 代码审计）

---

## 一、项目目标与模型架构

### 1.1 目标

WarpGAN（NeurIPS 2025）的 SVINet（Style-View Inpainting Network）负责对三维人脸新视角合成的最后一步：将 EG3D 编码器产生的 3D warp 投影图填补空洞，生成完整、逼真的新视角人脸图像。

### 1.2 整体架构

```
Real Photo → GOAE(EG3D encoder) → W+ codes [B,14,512]
                                        │
                           WProjModel (可训练)
                                        │ [B,18,768]  ← 映射为 SD 的 token 格式
                                        ▼
Source Photo → Warper.forward_warp(depth, c_src→c_novel) → warp_img + mask
                                        │
                           VAE encode → conditioning latent
                                        │
                                 BrushNet（冻结）← warp+mask条件
                                        │ down/mid/up residuals
                          ┌─────────────▼─────────────┐
                          │    冻结 SD 1.5 UNet         │
                          │  attn1(自注意力):           │
                          │    + ReferenceNet门控残差   │
                          │  attn2(交叉注意力):         │
                          │    空文本路径（冻结）         │
                          │    + W+ RCA门控残差（可训练）│
                          └─────────────┬─────────────┘
                                        │ noise_pred
                               50-step DPM-Solver++
                                        │
                                VAE decode
                                    │
                      SD 直接生成完整目标视角图像
                 （warp/mask 只作条件，不做最终像素拼接）
```

### 1.3 可训练参数（v4 factorized joint）

| 模块 | 参数量 |
|------|--------|
| WProjModel（W+ Mapper） | 1,967,616 |
| ReferenceAttentionProcessor (cross-attn RCA: Q/K/V/O + gate) | ~43,962,574 |
| ReferenceAttentionProcessor (self-attn Reference K/V/O + gate) | ~37,183,696 |

**冻结模块（参与前向但不更新）：** SD 1.5 VAE、SD 1.5 UNet 主干、BrushNet、ReferenceNet 特征提取 backbone。v4 中 Reference adapter 的 K/V/O 与 gate 可训练。

### 1.4 v4 当前结论（2026-07-28）

- v3→v4 初始化数学等价：最大绝对误差 `0`；zero gate 与 frozen base 最大误差 `0`。
- 完整系统测试中 Reference 80/80 个可训练张量有非零梯度，W+ 在 appearance pretrain 中无梯度。
- v4 checkpoint 可由生产 `DiffusionInpaintor` 严格加载。
- 当前推荐研究 checkpoint：`experiments/_v4b_factorized_pilot_500/checkpoints/iteration_200.pt`。
- 禁止使用 `[20260728-102109]_rca_v4a_reference_pretrain` 的 52k+ checkpoint 作为续训起点：
  它是 300k appearance-only 误配置，novel L1 已从 `0.1016` 漂移至最高 `0.2109`。

### 1.5 当前安全训练入口

默认 `configs/train_inpainting.yaml` 现在使用：

- `training.mode: factorized_joint`；
- v4-B step 200 模型与 PatchGAN 权重初始化；
- 新建 optimizer，不继承 discovery pilot 的高学习率 Adam；
- W+/identity LR=`2e-5`，Reference appearance LR=`5e-6`；
- 每 500 step 同时生成 novel-view 与 real-ref 固定验证图；
- `best_model.pt` 使用约束式 Pareto 选择，而不是只按一个 loss 或最后一步选择；
- novel hole L1 超过 `0.115` 连续 4 次时自动停止。

### 1.6 Full-frame 输出契约（2026-07-29）

旧验证曾同时保存：

```text
generated * mask + warp * (1-mask)   # hard composite
generated                             # SD raw frame
```

这不符合本项目的最终任务。Warp 投影在“可见区”仍可能存在拉伸、错位、重影和颜色误差，
因此不能把它当作最终不可修改像素。当前生产契约统一为：

```text
warp image + reliability mask -> BrushNet condition
W+ + Reference features       -> attention conditions
SD VAE decode                 -> 唯一最终完整图像
```

- 不再执行任何 `generated * mask + warp * (1-mask)`；
- 正式 `scripts/infer.py` 保存 SD full-frame；
- real-ref LPIPS/PatchGAN 直接监督 SD patch，不再先与真实 target 拼接；
- real-ref 对完整 frame 的 visible noise/x0 使用强监督；novel visible 只保留弱结构监督；
- 旧 PatchGAN 看过 hard-composite 分布，新版不继承其权重；
- 验证只生成 `logs/images/val/overview_step_XXXXXX.png` 一张总览；
- 每个身份两行：NOVEL 与 REAL-REF；
- checkpoint 以 full-frame 的 novel/real-ref 指标选择。

---

## 二、训练数据与任务设计

### 2.1 数据比例（phase-B, step 20001 起）

| 任务 | 比例 | 描述 |
|------|------|------|
| real 新视角伪目标 | 25% | 真实图 warp → c_novel，目标：EG3D 渲染的 y_hat_novel |
| synthetic 精确配对 | 50% | EG3D 同身份双视角精确对，无真实纹理噪声 |
| real-ref self-reconstruction | 25% | 用真实 novel warp 生成空洞形状，在源照片上挖洞并重建同一张真实照片；禁用 W+ |

### 2.2 损失函数

```
总 loss = loss_rec + loss_x0_fg + loss_pixel
         + 0.05 × loss_preserve
         + 0.1  × max(0, 0.05 + loss_x0_correct - loss_x0_wrong)

loss_rec    = noise_MSE(fg) + 0.1 × noise_MSE(bg)    [latent 空间]
loss_x0_fg  = alpha_bar加权 x0_MSE(fg)               [SNR 平衡]
loss_pixel  = lpips_w × LPIPS(fake256, tgt256)        [像素空间, t<200时生效]
              + adv_w  × GAN_loss(fake256)
loss_preserve = MSE(bg, no-W+ baseline)
loss_contrast = margin(correct_W+ vs wrong_W+)
```

---

## 三、遇到的问题与修复过程

### 3.1 目标视角严重错配（最核心错误，0–8万步完全无效）

**问题**：warp 是按 `c_novel` 投影的新视角图，但训练目标却是源视角原图 `batch['x']`。网络同时被要求"保持新视角几何"和"回归源视角内容"，目标矛盾，无法收敛。

**现象**：0–74k 步验证图逐帧变化仅 1.8/255，停止后量化确认。

**修复**：将训练目标改为 `y_hat_novel`（数据集 tuple 索引 13），synthetic 批次使用 `target_img`（索引 7）。

---

### 3.2 旧 RCA 结构缺陷

**问题**：旧的 `ReferenceAttentionProcessor` 只有独立的 K/V 投影，Q 共享冻结底座，输出也经底座 `to_out`。这不是真正的残差交叉注意力（RCA），而是简单地向文本 K/V 追加 W+ K/V。

**修复**：增加独立的 `to_q_wplus`、`to_out_wplus`，形成完整独立的 Q/K/V/O 路径；初始化从冻结底座复制权重，零门控（`wplus_scale=0`）确保 step 0 与原 SD 完全等价。

---

### 3.3 验证采样分布偏差

**问题**：旧验证在每个 DDIM 步内把干净背景 latent 硬回填到高噪声状态，偏离 BrushNet 官方采样（官方从纯噪声出发，BrushNet 的残差约束整图）。结果油画感更重，且不可复现。

**修复**：改为纯噪声初始化，50 步采样后在像素空间做 `gen × mask + warp × (1-mask)`。提取到共享函数 `utils/diffusion_inpainting.py`。

---

### 3.4 ReferenceNet 功能性失效

**问题（在 20k 时因果消融确认）**：

| 条件 | hole L1 |
|------|---------|
| full (W+ + Reference + BrushNet) | **0.152** |
| no Reference | 0.153 |
| no W+ | 0.376 |
| BrushNet only | 0.375 |

W+ 是主要信号，Reference 贡献几乎为零（差异 0.001）。根因：real/synthetic 任务的目标均为 EG3D 渲染，W+ 就能完全解释；ReferenceNet 无法找到"只有真实源图才能提供"的信息。

**早期方案及问题**：曾引入 mirror-ref 任务（源图水平翻转 + 镜像相机），但水平翻转照片并不等于非对称三维人脸从镜像相机观察到的精确新视角，因此会给 LPIPS 带来几何标签噪声。

**最终修复**：改为 view-mask real self-reconstruction（25%）。真实 novel warp 只用于产生与任务匹配的 disocclusion mask；该 mask 被应用到源视角真实照片，目标仍是同一张源照片。W+ 被禁用，因此 ReferenceNet 被迫独立恢复真实皮肤/头发纹理，同时输入和目标是严格像素配对。

---

### 3.5 x0 MSE 在高噪声步爆炸

**问题**：raw x0 MSE ∝ `(1-ᾱ)/ᾱ × ε_error`，t=999 时放大约 214 倍，导致 loss 剧烈波动（0.07 ~ 3.5）。

**修复**：SNR 平衡权重 `loss = ᾱ × (per_sample_x0_error / hole_area)`，等价 `(1-ᾱ) × ε_error`，上界为 1，各 timestep 贡献均匀。

---

### 3.6 推理断链

**问题**：`scripts/infer.py` 加载旧 LaMa FFC-ResNet（`inpaintor_state_dict`），新 checkpoint 存的是 `w_mapper_state_dict + rca_state_dict`，完全不兼容。

**修复**：新增 `models/diffusion_inpaintor.py`（生产级封装类），`infer.py` 支持 `diffusion`/`legacy_lama` 双后端；`configs/infer.yaml` 默认 `inpainting_backend: diffusion`，`batch_size: 1`（24GB 显存边界）。

---

### 3.7 GOAE/Swin 设备不兼容

**问题**：`swin_transformer.py` 第 161 行 `torch.clamp(..., max=torch.log(torch.tensor(1./0.01)))` 每次前向临时构造 CPU tensor，与 CUDA 参数冲突，CUDA 推理下报错。

**修复**：改为 `math.log(100.0)` 标量，数值完全相同。

---

### 3.8 验证随机性（跨 checkpoint 不可比）

**问题 A**：test loader `shuffle=True`，重启后样本变化。  
**修复**：改为 `shuffle=False`。

**问题 B**：`_load_raw_image` 每次调用 `random.randint(1,3)` 选 novel view，重启后视角不同。  
**修复**：新增 `fixed_novel_view` 参数，测试集设为 `fixed_novel_view=1`。

---

### 3.9 油画感（300k 步仍未解决）

**分析**：oil-painting 来自**权重**，与采样器无关（DPM-Solver++ 验证）。根因：

1. **监督目标平滑**：75% 的信号是 EG3D 神经渲染，天生缺乏皮肤纹理级高频。
2. **损失全在 latent 空间做 L2**：回归到条件均值 → 高频细节被抹平。

**修复（B 步骤）**：  
在低噪声时间步（t < 200，x0 估计可靠时）对 VAE 解码的 x0 施加：
- **LPIPS 感知损失**（Zhang et al. CVPR 2018）
- **NLayerDiscriminator PatchGAN + non-saturating logistic + 懒惰 R1**（Wang et al. pix2pixHD CVPR 2018；Mescheder ICML 2018 / StyleGAN2）

围绕空洞中心裁取原生尺度 32×32 latent patch，解码为 256×256 RGB patch 后做感知/对抗计算；不对完整 latent 做 resize，避免低通损失高频。

---

### 3.10 像素监督引入 OOM（本次问题）

**问题**：加像素监督后，低噪声步触发 VAE 解码（保留梯度，512×512）+ LPIPS（AlexNet）+ PatchGAN（4层）三路并行，原基础训练已占 ~19.5GB，三路叠加超过 24GB 上限。

**根因总结**：

```
VAE 512×512 解码 + 反向传播中间激活   ≈ +3 GB
LPIPS (AlexNet on 512×512)            ≈ +0.5 GB
NLayerDiscriminator 4-layer 512×512   ≈ +1.5 GB
D 判别器 + 反向传播                   ≈ +1 GB
总计                                 ≈ +6 GB，超出边界
```

**第一次修复为何仍然 OOM**：最初实现是“先用 VAE 将 64×64 latent 解码为 512×512 RGB，再把 RGB resize 到 256×256”。resize 发生得太晚，512×512 VAE 计算图已经建立，因此没有降低真正的峰值显存。

**最终修复**：不再缩小完整 latent（该做法会低通滤波并损失高频），而是选择空洞覆盖率最高的原生 32×32 latent 窗口；额外带 4 latent 像素上下文解码，再裁掉上下文边缘，得到对应的 256×256 原生尺度 RGB patch，避免 VAE crop 边界伪影。LPIPS 和 PatchGAN 在该 patch 上训练，因此既保留高频纹理信号，又避免完整 512×512 VAE 反向图。主扩散训练、验证和最终推理仍保持 512×512，不改变网络架构或输出分辨率。

最终完整 4-step 周期测试（batch=2，含三身份 50-step DPM++ 验证、real-ref LPIPS+GAN+R1，以及 4-latent VAE 上下文）峰值为 **17.98 GiB**，低于设备的 23.53 GiB。

PatchGAN 的 R1 正则先将空间 patch logits 对每张图取平均，再计算输入梯度，避免 R1 数值随 patch 数量平方级放大；日志同时记录 `px_d_loss`（logistic 主项）与 `px_d_total`（包含 lazy R1 的总项）。

**如果仍然 OOM（换设备标准）**：若 24GB 不够，推荐 A6000（48GB）或 A100（40/80GB），此时可考虑将 `PX_SIZE` 回升到 384–512 以提高感知损失分辨率。

---

## 四、量化收敛记录

### 4.0 v4 双任务 Pareto 复审

| 模型 | real-ref hole L1 ↓ | novel hole L1 ↓ | no-Reference real-ref L1 |
|---|---:|---:|---:|
| v3 300k | 0.23990 | **0.10102** | 0.24284 |
| v4-A 1000 | **0.15677** | 0.12379 | 0.24284 |
| **v4-B 200** | **0.15730** | **0.10533** | 0.24284 |
| v4-B 500 | 0.15149 | 0.11892 | 0.24284 |
| v4-C continuous 400 | 0.19386 | 0.09735 | 0.24284 |

v4-B 200 是当前 Pareto 选择。v4-C 连续 alpha-bar 权重改善了 novel pseudo-target 指标，
但真实 Reference 修补明显回退，因此不作为默认目标。

### 4.0.1 Scheduler / step count 对照

| Scheduler | Steps | hole L1 ↓ | 三身份耗时 |
|---|---:|---:|---:|
| DDIM | 20 | 0.10672 | 4.37s |
| DDIM | 50 | 0.10159 | 10.42s |
| DDIM | 100 | 0.10058 | 20.64s |
| DPM++ | 20 | 0.10174 | 4.27s |
| **DPM++** | **50** | **0.10101** | **10.37s** |
| DPM++ | 100 | 0.10095 | 20.55s |

DPM++ 50→100 仅改善约 `0.00006`，耗时近翻倍；Laplacian/microtexture 还略降。
因此 50 步是合理默认值，油画感不是采样步数不足造成的。

| 阶段 | 步数 | hole L1 | warp 基线 | 改善 |
|------|------|---------|-----------|------|
| Phase-A（目标有缺陷） | 0 | 0.515 | 0.566 | +0.051 |
| Phase-A | 20k | 0.169 | 0.566 | +0.397 |
| Phase-B（目标修正）| 21k | 0.160 | 0.225 | +0.065 |
| Phase-B | 30k | 0.096 | 0.225 | +0.129 |
| Phase-B（plateau） | 300k | 0.157 | 0.335 | +0.178 |

（注：Phase-B step 20001 验证样本固定为 view 1，与 Phase-A 基线不同，所以数值跨阶段不可直接比较。300k 是新一轮训练包含 phase-B 修正后的完整运行。）

---

## 五、当前状态与后续建议

**现状**：当前代码采用 v4 checkpoint 语义（独立 Reference K/V/O、多尺度 ReferenceNet、
factorized joint 支持、real-ref 原生 latent patch LPIPS/GAN）。v3 checkpoint 可 function-preserving
初始化 v4，但不能 optimizer resume；当前推荐研究 checkpoint 为 v4-B step 200。

**推荐后续步骤**：

1. 先停止仍在运行的旧 appearance-only 进程；
2. 直接使用默认配置启动新的安全 factorized 实验；
3. 训练时只需查看 `overview_step_*.png`、`val_metrics.txt` 和 `best_model.pt`；
4. 在安全 factorized 基线上再实现 source→target 多尺度 Reference feature warp。

---

## 六、关键文件索引

| 文件 | 作用 |
|------|------|
| `training/coach_inpainting_static.py` | 训练主循环、损失、验证 |
| `models/referencenet/attention_processor.py` | RCA + ReferenceNet 门控注意力处理器 |
| `models/mapper/w_proj.py` | W+ latent → SD token 映射 |
| `utils/diffusion_inpainting.py` | 共享 BrushNet/DPM++ 采样函数 |
| `models/diffusion_inpaintor.py` | 生产级推理封装，加载 checkpoint |
| `datasets/dataset_inpainting_static.py` | 真实静态数据集，`fixed_novel_view` 支持 |
| `scripts/infer.py` | 双后端推理入口（diffusion / legacy_lama） |
| `configs/train_inpainting.yaml` | 训练超参，含 `pixel_sup` 配置块 |
| `configs/infer.yaml` | 推理超参，默认 diffusion backend |
| `docs/TRAINING_AUDIT_REPORT.md` | 详细技术审计报告 |
| `models/goae/swin_transformer.py` | 已修复 CUDA 设备不兼容 |
| `scripts/test_v5_reliability_and_adapter.py` | v5 可靠性投影、local Reference 与 LoRA gate CPU 单测 |

---

## 七、v5 等变纹理与可靠性修正（2026-07-30）

### 7.1 由最新固定验证图确认的现象

- NOVEL 的 EG3D geometry reference 与 SD 输出姿态基本一致，说明主要矛盾不再是相机视角；
- warp 非空洞区域通常比 SD raw 更自然，SD 整帧重绘会淡化睫毛/眼睑、扰动瞳孔与皮肤头发纹理；
- REAL-REF 身份稳定，但可见区发生变色和风格化，且 0–10k 的 L1 改善未对应肉眼高频质量改善。

### 7.2 新发现的代码根因

此前代码已经计算 `ref_feat_aligned`，但它在功能上是死链：

1. real-ref 精确照片监督不构造 aligned feature；
2. novel 分支构造 aligned feature，但 factorized training 禁止 appearance 参数更新；
3. 固定验证没有向采样器传 aligned feature；
4. 正式 `DiffusionInpaintor` 也没有源深度/相机接口，无法在推理时构造 aligned feature。

因此旧 10k 训练实际上只能继续优化全局 Reference 和整帧 appearance residual，无法学到眼睑、睫毛、瞳孔边缘与发丝的局部跨视角传输。

### 7.3 v5 修改

- **等变局部 Reference**：real-ref 使用恒等空间对应训练 local gate；novel/正式推理使用同一个 3D warper 把多尺度 source feature 对齐至 target query 坐标；
- **可学习且安全的门控**：local Reference 与高分辨率 appearance LoRA 均显式 gate；零 gate 保持初始化函数，LoRA 的微小非零 up 初始化保证 gate 首步有梯度；
- **真实高频监督常态化**：每个 real-ref batch 至少一个 `t<200` 样本，避免真实 LPIPS/GAN 仅随机覆盖约 35% batch；加入原生分辨率图像梯度损失，直接约束眼部轮廓、发丝和皮肤微纹理；
- **可靠性加权数据一致性**：不恢复旧 hard composite。仅对远离洞边界的高置信 warp 核心区进行 0.85 强度软投影，空洞和边界带仍完全由 SD 生成；
- **raw/fused 双重审计**：验证同时展示 `SD raw` 与 `reliability output`；checkpoint Pareto 选择只使用 raw SD 指标，防止软投影掩盖模型退化；
- **高频硬指标**：新增 full/hole/visible 图像梯度误差，L1 与边缘误差共同判断油画感、眼睑/睫毛缺失和纹理扰动。

### 7.4 checkpoint 契约

- 新 checkpoint：`architecture_version=5`；
- 新输出契约：`reliability_weighted_data_consistency`；
- v3/v4 只能通过 `initialization_path` 初始化 v5，不能继承 optimizer；
- 正式推理必须传入 source depth、source camera 与 target camera，才能启用等变 local Reference；
- 默认新实验目录：`experiments/train_inpaintor/*_rca_v5_equivariant_reliability`。

### 1.7 v4-fullframe 10k 失败与从零重启（2026-07-30）

v4-fullframe 实验从已有权重初始化、full-frame SD 输出、10k 步训练后：
- 硬指标持续改善（real_ref_full 0.099→0.081）；
- 但用户肉眼确认油画感与修改前几乎一致：纹理扰动感、睫毛缺失、瞳孔不对称、眼睑缺失、未挖洞区变色、背景杂色。

根因：所有先前 checkpoint 的条件模块已收敛到 EG3D 平滑材质域，从这些权重初始化无法跳出油画域吸引子。

决策：从零门控适配器重新开始训练 300000 步。配置已改为 `initialization_path: null`、`max_steps: 300000`、`arm_after_first_pass: True`。

从零训练命令：
```bash
cd /data/xzy/WarpGAN
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/xzy/miniconda3/envs/warpgan/bin/python scripts/train_inpainting.py
```
