# WarpGAN Diffusion-SVINet 项目演进、实现核查与当前问题报告

> 对比基线：`/data/xzy/warpgan_orig/WarpGAN-main`  
> 当前工程：`/data/xzy/WarpGAN`  
> 历史稳定基线：`experiments/train_inpaintor/[20260722-220118]_rca_v3_full`（300,000 步）  
> 当前 v4 Pareto checkpoint：`experiments/_v4b_factorized_pilot_500/checkpoints/iteration_200.pt`  
> 最终复审日期：2026-07-28  
> 文档目的：向导师完整说明原版方法、我们的修改过程、已发现和修复的问题、DDIM/DPM-Solver 步数是否可能导致当前现象、目前油画感的证据链，以及下一阶段应如何修改。

---

## 0. 一页结论

### 0.0.1 2026-07-28 52k appearance-only 训练事故复审

实验 `experiments/train_inpaintor/[20260728-102109]_rca_v4a_reference_pretrain` 的配置为：

```yaml
checkpoint_path: null
initialization_path: v3 iteration_300000.pt
training.mode: appearance_pretrain
max_steps: 300000
```

因此它**不是 optimizer 续训**，而是从 v3 300k 模型权重初始化一个新的 v4 appearance
实验。step 0 的 novel L1=`0.101610` 与 v3 一致，证明初始化正确；问题在于把本应只持续
约 1k–2k 步的 capability warm-up 错误配置成了 300k 长训。该模式每一步只更新 Reference
appearance adapter，W+ 完全冻结，也没有任何 novel-view loss 约束 Reference。

实际固定验证已经出现明确失控：

| step | novel hole L1 ↓ |
|---:|---:|
| 0 | 0.101610 |
| 8k | 0.108439 |
| 24k | 0.157948 |
| 46k | 0.175251 |
| 52k | 0.191767 |
| 54k | **0.210880** |
| 56k | 0.189191 |

同时 Reference gate 分布持续大幅重排（例如 gate min 从 step 20k 的约 `-0.305` 回到
step 50k 的约 `-0.676`），说明模型不是稳定微调，而是在单一同视角任务下不断重构
Reference 使用方式。视觉上的“局部像真人、整体脸部关系不自然”即恐怖谷，是同视角
appearance 能力与 novel-view 几何/身份职责脱耦后的典型结果。

**处置：**

- 该 52k+ 实验停止作为候选模型；不要从其 checkpoint 继续训练或推理选优；
- 默认训练改为正式 `factorized_joint`，同一真实 batch 分别执行 appearance 与 novel 更新；
- real-photo loss 只更新 Reference，EG3D novel loss 只更新 W+；
- appearance-only 模式增加 `appearance_max_steps=2000` 硬保护，超过即拒绝启动；
- 默认从 v4-B step 200 Pareto checkpoint 初始化，而不是从 52k 恐怖谷 checkpoint 初始化；
- 使用 `identity_lr=2e-5`、`appearance_lr=5e-6`、PatchGAN LR=`2e-5`；
- 每 500 step 同时验证 novel 和 real-ref；novel L1 必须 `<=0.115`；连续 4 次越界自动停止；
- step 0 初始化模型先保存为 `best_model.pt`，之后只有 real-ref 改善且 novel 不越界才覆盖。

### 0.0.2 2026-07-29 输出契约修正：warp 不是最终像素真值

旧实现最后执行：

```python
composed = generated * mask + warp_image * (1.0 - mask)
```

这适用于“已知区完全可靠”的经典 inpainting，但不适用于 WarpGAN：forward warp 即使在
visibility mask 内仍可能有拉伸、深度误差、双影和颜色偏差。最终任务需要 SD 根据 warp、
mask、W+ 和 Reference 条件重新生成完整目标视角，而不是只填白洞。

当前统一为：

```text
FINAL = SD full-frame decode
warp/mask = condition only
```

正式推理、验证、指标和 pixel supervision 已全部采用这一契约。验证输出合并为一张
`overview_step_*.png`：每个身份依次展示 NOVEL 行和 REAL-REF 行；图首直接标注
full/hole/visible L1。旧的 `composed/raw` 双列不再生成。

### 0.0 2026-07-28 v4 实验更新

在 v3 300k 基线上，我们已经完成而不再只是“建议”以下工作：

1. Reference self-attention 增加独立可训练 `K/V/O` residual adapter；
2. 从 v3 复制冻结 SD 的 K/V/O 并保留 gate，初始化前后函数严格等价；
3. 建立只训练 Reference 的真实照片 appearance pretraining（v4-A）；
4. 建立同 batch、显式梯度隔离的 factorized joint training（v4-B）；
5. 完成 DDIM/DPM++ 20/50/100 步与连续 alpha-bar 像素权重（v4-C）对照。

关键双任务结果（固定 3 个身份、DPM++ 50 步、seed 42）：

| checkpoint | real-ref hole L1 ↓ | novel EG3D-target hole L1 ↓ | 解释 |
|---|---:|---:|---|
| v3 300k | 0.23990 | **0.10102** | 结构强，真实 Reference 能力弱 |
| v4-A 1000 | **0.15677** | 0.12379 | Reference appearance 明显学会，但损伤 novel 一致性 |
| **v4-B 200** | **0.15730** | **0.10533** | 当前最佳 Pareto：保留真实 appearance，并恢复 novel 结构 |
| v4-B 500 | 0.15149 | 0.11892 | 真实重建继续改善，但 novel 过度漂移 |
| v4-C 400 | 0.19386 | **0.09735** | 连续权重改善 novel，但真实 appearance 明显回退 |

`no-Reference` 因果消融进一步证明 v4 的改善来自 Reference adapter：v4-A 1000 的
real-ref L1 从 full 的 `0.15677` 退化到 no-Reference 的 `0.24284`；而 v3 full/no-Reference
仅为 `0.23990/0.24284`。因此独立 K/V/O 不是“参数增加但没使用”，而是可测量的因果能力。

当前决策：

- **研究默认 checkpoint 选 v4-B step 200**；
- 生产像素目标保留 `hard_low_noise (t<200)`；连续 alpha-bar 保留为可复现负结果，不设为默认；
- 下一优先级是 **target-aligned Reference feature warp**，因为当前 Reference feature 位于 source
  坐标，而 target UNet query 位于 novel-view 坐标；camera token 只能提供全局姿态，不能解决纹理空间对应；
- 暂不继续增加采样步数，也不继续盲目延长 v4-B 训练。
- `appearance_pretrain` 只允许有限 warm-up，正式长训必须使用带双任务安全约束的 `factorized_joint`。

### 0.1 我们做的不是“把 DDIM 步数改了一下”

原版 WarpGAN 的 SVINet 是一个基于 LaMa/FFC-ResNet 的前馈修补器。我们把整个修补核心替换成了：

```text
3D warp + hole mask ───────────────→ BrushNet ──────┐
                                                    │
W+ latent → WProjModel → W+ cross-attention ───────┼→ SD 1.5 UNet → diffusion sampling
                                                    │
真实 source photo → ReferenceNet → self-attention ─┘
```

并重新设计了：

- 训练目标；
- W+ 条件注入；
- ReferenceNet 纹理条件；
- real/synthetic/real-reference 三类任务；
- SNR 平衡的 latent loss；
- LPIPS、PatchGAN 与 R1；
- 确定性验证；
- 共享采样器；
- 正式推理与 checkpoint 格式。

`training/coach_inpainting_static.py` 与原版相比约有 2252 行 diff；另外新增了 `models/diffusion_inpaintor.py`、`utils/diffusion_inpainting.py`、`models/mapper/w_proj.py` 和自定义 Reference attention 等模块。因此当前结果不能简单归因于“DDIM 步数设置不合适”。

### 0.2 当前 300k 结果说明了什么

最终 v3：

```text
step=300000
hole_l1          = 0.101007
warp_hole_l1     = 0.334948
hole improvement = 0.233942
known_l1         = 0.000000
```

这说明模型已经明显学会：

- 使用 warp 和 mask 完成空洞；
- 保持已知区域；
- 使用 W+ 区分身份；
- 使用 Reference 条件改变输出。

但 `raw_s42` 仍有稳定油画感。因此目前最准确的判断是：

> 模型不是没有拟合，而是已经收敛到“结构和身份正确、材质仍偏 EG3D/平滑”的视觉域。

### 0.3 当前油画感最可能的根因

按证据强弱排序：

1. **75% 的密集 diffusion 监督使用 EG3D 渲染图作为目标，导致可训练分支学习 EG3D 的平滑材质域。**
2. **v3 Reference 分支表达能力不足；v4 已补全 K/V/O，但跨视角 feature 仍未与 target query 对齐。**
3. **真实照片 LPIPS/GAN 监督覆盖较少，而且只训练同视角真实照片补洞，不训练真实 novel-view 纹理迁移。**
4. **SD 1.5 UNet 与 BrushNet 主体冻结，网络缺少足够自由度完成从一般扩散域到真实人脸新视角域的适配。**

已经通过实验基本排除：

- 训练步数不足；
- DPM-Solver++ 本身导致油画感；
- DDIM/DPM++ 选择是主要问题；
- SD VAE 是主要问题；
- checkpoint 没有加载；
- W+ 或 Reference 分支完全没工作；
- mask 方向、known-region compositing 等常见实现错误。

### 0.4 下一阶段最推荐的修改

不是继续增加采样步数，也不是锐化、硬阈值或后处理，而是：

1. 以已完成的 Reference K/V/O 和 v4-B step 200 为基线；
2. 将 source Reference feature 用 3D warp 对齐到 target query 坐标，并保留 visibility/validity mask；
3. 将 EG3D 的“几何监督”与“RGB 外观监督”进一步分开，避免把 EG3D 全频材质当作真实外观；
4. 必要时在高分辨率 UNet 层加入小型 LoRA；
5. 若 feature warp 后仍缺大视角姿态控制，再加入 camera token 或 depth/normal。

---

# 1. 原版 WarpGAN 的 SVINet 是什么

## 1.1 原版整体任务

WarpGAN 首先用 GOAE/EG3D 将真实人脸反演到 W+，然后根据源相机和目标相机进行三维 forward warp。由于目标视角会出现源视角看不到的区域，因此 warp 图存在空洞，SVINet 负责填补这些区域。

原版主要数据流为：

```text
source image x
   │
   ├→ EG3D inversion → W+ codes
   │                    └→ EG3D novel render y_hat_novel
   │
   └→ depth + c_source + c_novel → forward warp → warp_img + visibility mask

warp_img / mask / y_hat_novel / W+ / mirror condition
                    ↓
            LaMa-style FFC generator
                    ↓
              novel-view result
```

原版关键代码位于：

- `training/coach_inpainting_static.py`
- `models/saicinpainting/`
- `scripts/infer.py`
- `configs/train_inpainting.yaml`

## 1.2 原版不是普通 LaMa

原版 generator 是 `ffc_style_resnet`，不仅输入 masked image 和 mask，还可以输入：

- EG3D novel render `y_hat_novel`；
- W+ latent；
- 镜像 warp；
- 镜像 mask。

其 `get_inp()` 逻辑在 `warp.hybrid=True` 时会用 EG3D novel render 填充空洞：

```text
masked_img = warp_img × known + y_hat_novel × hole
```

随后再将 masked image、inversion 和 mask 拼接给 FFC generator。

因此原版的优势是目标 pose 条件很强，因为 `y_hat_novel` 本身就是 EG3D 在目标相机下的完整渲染。

## 1.3 原版训练监督

原版包含：

- forward novel-view inpainting；
- novel→source inverse warp；
- inverse reconstruction；
- synthetic paired views；
- L1；
- ID loss；
- latent re-encoding consistency；
- adversarial loss；
- feature matching；
- ResNet perceptual loss；
- 可选 mirror loss。

其中真实数据没有真实 novel-view GT，所以主要通过 inverse warp 回源视角，与真实 source image 比较。这是一种 cycle/self-supervision。

## 1.4 原版局限

原版的主要问题不是“不能工作”，而是：

1. FFC/LaMa 的生成先验弱于现代 latent diffusion；
2. `y_hat_novel` 作为 RGB 条件会直接携带 EG3D 的平滑、油画式材质；
3. 真实数据的监督主要落在 inverse reconstruction，而不是最终 novel-view realism；
4. 原版验证 loss 主要衡量回源重建，不等价于最终 novel-view 真实感；
5. 网络倾向确定性回归与平均化，难以恢复真实皮肤、发丝等高频细节。

我们的研究初心就是：

> 保留 WarpGAN 的 3D warp、相机和身份优势，用 diffusion 的生成先验替换相对落后的 FFC inpainting，使新视角结果更真实。

---

# 2. 当前 Diffusion-SVINet 的最终架构

## 2.1 结构总览

```text
Source photo
   │
   ├→ GOAE encoder → W+ [B,14,512]
   │                   ↓
   │               WProjModel
   │                   ↓ [B,18,768]
   │            W+ residual cross-attention
   │
   ├→ SD VAE → ReferenceNet → multiscale reference features
   │                              ↓
   │                    gated reference self-attention
   │
   └→ Warper(depth, c_src, c_novel)
                  ↓
           warp_img + hole mask
                  ↓
          VAE latent + mask latent
                  ↓
              BrushNet
                  ↓ residuals
       frozen SD 1.5 denoising UNet
                  ↓
       50-step DPM-Solver++ sampling
                  ↓
              VAE decode
                  ↓
generated × hole + warp × known
```

## 2.2 可训练与冻结模块

### 可训练

- `WProjModel`；
- W+ cross-attention 独立 Q/K/V/O；
- W+ gate；
- Reference self-attention 独立 K/V/O 与 gate；
- pixel supervision 的 PatchGAN discriminator。

### 冻结但参与前向

- SD 1.5 VAE；
- SD 1.5 UNet 主体；
- BrushNet；
- ReferenceNet 特征提取网络；
- ID 和 LPIPS 网络。

v3 的关键容量瓶颈是 Reference 只有 gate；v4 已补全独立 K/V/O，并通过
`scripts/test_v4_reference_adapter.py` 与 `scripts/test_v4_full_system.py` 验证了函数等价初始化、
80/80 参数梯度和生产加载。

---

# 3. 从开始到现在的修改时间线

以下版本名是为了说明演进逻辑，不代表每个中途目录都是严格发布版本。

## 3.1 初始 diffusion 迁移：用 BrushNet + SD 替换 FFC

### 修改内容

- 引入 SD 1.5 VAE 和 UNet；
- 引入预训练 BrushNet；
- 使用 warp image + mask 作为 BrushNet 条件；
- 将目标图编码到 latent，随机采样 DDPM timestep 和 noise；
- 训练条件分支预测噪声；
- 增加 W+ mapper；
- 用自定义 attention processor 注入 W+。

### 新增主要文件

- `models/mapper/w_proj.py`
- `models/referencenet/attention_processor.py`
- `models/referencenet/unet_2d_condition.py`

### 这一阶段的目标

先证明：

- 3D warp 可以作为 diffusion inpainting 条件；
- W+ 可以转为 SD token 并控制身份；
- 冻结 SD 主体时只训练小型条件模块也能收敛。

## 3.2 发现并修复目标视角错配

### 问题

早期代码中，输入 warp 已经是 `c_novel` 下的新视角，但监督目标仍然用了源视角 `x`。

这等价于同时要求模型：

```text
保持目标相机几何
又生成源相机图像
```

物理上相互矛盾。

### 修复

- real batch 目标改为 tuple 索引 13 的 `y_hat_novel`；
- synthetic batch 目标改为 tuple 索引 7 的 paired `target_img`。

### 意义

这是项目中最重要的一次正确性修复。错配阶段 checkpoint 学到的是矛盾目标，不能用来判断 diffusion 方法本身好坏。

## 3.3 将旧 W+ 注入改为完整 RCA

### 旧问题

旧 processor 只给 W+ 做 K/V 投影，Q 和输出投影共享冻结底座。它更接近“给文本注意力追加 token”，不是独立 residual cross-attention。

### 修改

增加：

```text
to_q_wplus
to_k_wplus
to_v_wplus
to_out_wplus
wplus_scale
```

并从原 SD attention 复制初始 Q/K/V/O 权重；gate 初始化为 0。

### 为什么这样做

零 gate 时：

```text
新 processor 输出 = 原始预训练 SD attention 输出
```

这让新增分支以 residual 方式逐步学习，不会在 step 0 破坏 BrushNet/SD 的预训练分布。

## 3.4 修复验证采样实现

### 早期问题

旧验证曾在每个 DDIM step 中强行把干净 known-region latent 回填到当前高噪声 latent。这样做会混合不匹配的噪声状态：

```text
hole：处于 timestep t 的高噪声状态
known：被直接换成干净 latent
```

这偏离 BrushNet 官方采样分布，并可能产生边界与纹理异常。

### 最终修改

在 `utils/diffusion_inpainting.py` 中统一为：

1. 从纯 scheduler-scaled noise 开始；
2. 每一步只使用 BrushNet residual 约束；
3. 完成所有采样步；
4. VAE decode；
5. 最后只在像素空间做：

```python
composed = generated * mask + warp_image * (1.0 - mask)
```

### 为什么最终 compositing 不是 hack

inpainting 的已知区本来就有观测值，不需要重新生成。最终投影相当于满足数据一致性：

```text
known region = observed warped pixel
hole region  = model prediction
```

验证图同时保存：

- `raw_s42(no_composite)`：完整 diffusion 原始输出；
- `gen_s42(composed)`：最终 inpainting 输出。

因此可以明确区分生成器本身的问题和最终合成的问题。

## 3.5 引入 ReferenceNet

### 动机

W+ 和 EG3D 可以表示身份与粗外观，但真实皮肤、发丝、痣、局部皱纹等信息只有 source photo 中存在。

### 设计

- 用 source photo 经 SD VAE 编码；
- 使用冻结 ReferenceNet 提取多尺度特征；
- 在 SD UNet self-attention 中，以 Reference features 作为额外 K/V；
- 通过 `reference_scale` 零初始化 residual gate 注入。

### 发现的问题

早期 20k 因果消融：

| 条件 | hole L1 |
|---|---:|
| full | 0.152 |
| no Reference | 0.153 |
| no W+ | 0.376 |
| BrushNet only | 0.375 |

说明 W+ 很有效，但 Reference 几乎没有因果贡献。

### 原因

real novel 和 synthetic 目标本身都是由 EG3D/W+ 生成或解释的。模型不需要 source photo 中独有的真实高频信息即可降低训练 loss，因此 Reference 分支没有明确任务。

## 3.6 mirror-ref 尝试与放弃

### 中途方案

曾使用 source 水平翻转图作为 mirror target，希望构造真实照片监督并激活 ReferenceNet。

### 问题

二维水平翻转照片不等于非对称三维人脸从镜像相机观察到的严格新视角：

- 左右脸不对称；
- 耳朵和遮挡关系不同；
- 发型与光照不是三维镜像；
- LPIPS 会把几何误差误当作纹理误差。

### 决策

不继续使用 mirror-ref 作为最终精确监督，避免用标签噪声换取 gate 变大。

## 3.7 v3 real-ref self-reconstruction

### 构造方式

1. 先做真实 novel-view warp，获得真实任务形状的 disocclusion mask；
2. 不使用 novel warp RGB；
3. 将该 mask 放回 source image 坐标系，在 source photo 上挖洞；
4. target 仍然是同一张 source photo；
5. 禁用 W+，强制 Reference 分支负责恢复真实纹理。

### 优点

- 输入与目标严格像素配对；
- target 是真实照片；
- mask 形状来自真实 novel-view 任务；
- 避免 mirror-ref 的几何错误；
- W+ 被禁用，Reference 不能被身份 latent 替代。

### 局限

它训练的是同视角补洞，不是真实跨视角纹理迁移。这一点后来成为当前油画感的重要原因之一。

## 3.8 ReferenceNet 多尺度修复

### 问题

旧实现按 channel 数保存一个 feature。ReferenceNet 中可能同时存在：

```text
1280 channels, 16×16
1280 channels, 8×8
```

只按 channel 建字典会被后一个覆盖前一个，导致某些 attention 层收到错误空间尺度。

### 修复

- 每个 channel 保存 feature list；
- 注入时根据当前 query token 数选择空间尺度最接近的 feature；
- 必要时双线性调整到当前 attention 尺度。

## 3.9 SNR 平衡 x0 loss

### 问题

从 epsilon 反推：

```text
x0_pred = (x_t - sqrt(1-alpha_bar) * eps_pred) / sqrt(alpha_bar)
```

高噪声时 raw x0 MSE 会被约 `(1-alpha_bar)/alpha_bar` 放大；接近 t=999 时可能放大数百倍。

### 修复

用 `alpha_bar` 对 per-sample x0 error 加权，使其等价于有界的 epsilon error 权重，避免随机高 timestep 支配优化。

## 3.10 真实照片 LPIPS + PatchGAN + R1

### 动机

latent L2/noise MSE 擅长学习结构，但容易平均化高频；油画感需要真实像素域约束。

### 设计

仅对严格配对的 real-ref 使用：

- LPIPS；
- PatchGAN non-saturating generator loss；
- discriminator logistic loss；
- lazy R1。

没有对 EG3D target 使用 PatchGAN real label，避免教判别器把 EG3D 平滑图当真实照片。

## 3.11 像素监督 OOM 与原生 latent patch

### 初始问题

完整 512×512 x0 VAE decode + LPIPS + PatchGAN + backward 超过 24GB。

### 无效的第一次优化

先 decode 512，再 resize 到 256。此时大 VAE 计算图已经建立，显存不会真正下降。

### 最终修改

- 在原生 64×64 latent 上选择洞覆盖率最高的 32×32 patch；
- 增加 4 latent 像素上下文；
- VAE decode 为带上下文 RGB；
- 裁掉上下文边缘，得到原生尺度 256×256 patch；
- LPIPS/PatchGAN 只在该 patch 上训练。

完整 batch=2、三身份 50-step 验证、LPIPS/GAN/R1 的峰值约 17.98 GiB。

## 3.12 确定性验证

为使 checkpoint 可比较，修改了：

- test loader：`shuffle=False`；
- test dataset：`fixed_novel_view=1`；
- validation seed：固定为 42；
- validation 样本固定为 3 个 identity；
- 每个 panel 加标签；
- 修复早期把 identity 0 输出广播到三 identity target 的指标 bug。

## 3.13 正式推理闭环

### 原问题

旧 `scripts/infer.py` 只加载 LaMa checkpoint 的：

```text
inpaintor_state_dict
```

而 diffusion checkpoint 保存：

```text
w_mapper_state_dict
rca_state_dict
```

训练完成后不能直接正式推理。

### 修复

- 新增 `models/diffusion_inpaintor.py`；
- `scripts/infer.py` 支持 `diffusion` 和 `legacy_lama` 双后端；
- diffusion 后端不再计算无用的 EG3D novel RGB render；
- 训练验证与正式推理共用 `sample_brushnet_inpainting()`；
- 支持 v2/v3 checkpoint，v3 使用多尺度 Reference features；
- v2 不允许作为 v3 optimizer resume。

## 3.14 其他工程修复

- 修复 GOAE/Swin 在 CUDA 上临时构造 CPU tensor 的设备冲突；
- resume 时继续写原实验目录，并从 checkpoint 下一步开始；
- checkpoint 写入 `architecture_version`、监督类型和 Reference feature 格式；
- 日志覆盖 real/synth/real-ref 完整周期；
- TensorBoard 按 batch kind 记录不同 loss。

---

# 4. 当前训练任务和损失到底是什么

## 4.1 任务比例

当前每四步：

| phase | 占比 | 任务 | 目标域 |
|---|---:|---|---|
| real | 25% | 真实 source warp 到 novel view | EG3D `y_hat_novel` |
| synth | 50% | EG3D synthetic paired novel views | EG3D/synthetic render |
| real-ref | 25% | source photo 自重建，使用 novel mask | 真实照片 |

## 4.2 总损失

```text
L = L_noise_fg
  + 0.1 L_noise_bg
  + L_x0_fg_balanced
  + L_pixel_real_ref
  + 0.05 L_known_preserve
  + 0.1 L_W+_contrast
```

其中：

```text
L_pixel_real_ref = LPIPS + 0.1 × generator adversarial loss
```

PatchGAN 独立执行 discriminator update，并使用 lazy R1。

## 4.3 配置中的 legacy 字段

`configs/train_inpainting.yaml` 仍保留原版 generator、LaMa loss 等字段，以兼容原始脚本和 legacy inference。但当前 diffusion coach 并不使用大部分这些字段。

例如顶层：

```yaml
losses:
  id:
    weight: 0.5
  lpips:
    weight: 0.1
```

主要用于加载监控网络；普通 real/synth batch 的 ID/LPIPS 只是 detached monitor，不直接加入总 loss。真正反向的 LPIPS 来自 `losses.pixel_sup.lpips_weight`，并只作用于 real-ref 合格样本。

这一点应在后续配置中进一步清理，避免读者误以为所有原版 loss 都仍参与训练。

---

# 5. 导师关心的问题：DDIM 步数是否有问题

## 5.1 先区分训练步数、diffusion timestep 和采样步数

这三个“步数”完全不同：

### 训练 iteration

```text
global_step = 0 ... 300000
```

表示参数更新次数。

### DDPM training timestep

训练时每个样本随机采：

```python
timesteps = randint(0, 1000)
```

然后使用 `DDPMScheduler.add_noise()` 构造 `x_t`。训练并没有跑 50 次 DDIM 循环；它是标准 diffusion 的单 timestep 噪声预测训练。

### inference sampling steps

验证和推理时才设置：

```text
num_inference_steps = 50
```

表示从纯噪声通过 scheduler 调用 UNet 50 次得到最终 latent。

因此：

> 50 步不会影响训练 loss 的定义，也不会解释为什么权重在 300k 后学到 EG3D 平滑材质。它只影响同一组权重如何数值求解反向生成轨迹。

## 5.2 当前默认并不是 DDIM，而是 DPM-Solver++

`utils/diffusion_inpainting.py` 默认：

```python
scheduler_type = "dpmpp"
DPMSolverMultistepScheduler.from_config(
    scheduler_config,
    algorithm_type="dpmsolver++",
    use_karras_sigmas=True,
)
```

DDIM 仅作为消融选项保留：

```python
scheduler_type = "ddim"
```

所以如果将当前结果称为“50 步 DDIM 结果”并不准确。当前正式验证是 50-step DPM-Solver++，训练 scheduler 是 DDPM noise scheduler。

## 5.3 当前采样流程是否正确

当前流程为：

1. 将 warp image 编码为 BrushNet condition latent；
2. mask nearest resize 到 latent 尺度；
3. 从纯高斯噪声乘 `scheduler.init_noise_sigma` 开始；
4. 每一步先执行 `scheduler.scale_model_input()`；
5. BrushNet 接收同一个 scaled latent 和 timestep；
6. UNet 接收 BrushNet residual、W+ 和 Reference；
7. `scheduler.step(...).prev_sample`；
8. 最后 VAE decode；
9. 最后一次像素空间 known-region compositing。

这是与 diffusers/BrushNet 采样范式一致的实现。早期每一步回填干净 background latent 的非标准实现已经删除。

## 5.4 DPM++ 与 DDIM 同 seed 消融

我们使用最终 300k checkpoint、同一输入、同一 seed=42、同样 50 步，比较：

- full DPM-Solver++；
- full DDIM。

结果：

```text
full DPM++: lap=0.05459, microtexture=0.02549
full DDIM:  lap=0.05792, microtexture=0.02572
两者平均像素差：0.01546
```

两者视觉分布与纹理统计非常接近。产物：

```text
experiments/_oil_ablation/seed42_branch_scheduler_ablation.png
```

这说明：

> 当前油画感主要写在训练后的条件权重与监督分布中，而不是由 DPM++ 或 DDIM 的选择产生。

## 5.5 50 步是否足够

50 步对 DPM-Solver++ 来说不是极端少步；它已经属于相对保守的采样设置。若模型在 10、20、50、100 步间仅有小幅清晰度变化而材质风格保持一致，那么增加步数只能改善求解误差，不能改变模型学到的目标分布。

严格步数消融已经完成：

| sampler | steps | seed | 输入/条件 |
|---|---:|---:|---|
| DDIM | 20 | 42 | 完全相同 |
| DDIM | 50 | 42 | 完全相同 |
| DDIM | 100 | 42 | 完全相同 |
| DPM++ | 20 | 42 | 完全相同 |
| DPM++ | 50 | 42 | 完全相同 |
| DPM++ | 100 | 42 | 完全相同 |

结果见 `experiments/_scheduler_steps_v3_300k/metrics.csv`。DPM++ 50→100 的 hole L1
仅从 `0.101009` 改善到 `0.100949`，三身份耗时从 `10.37s` 增至 `20.55s`；
纹理统计没有向真实照片方向改善。因此继续增加步数不再是优先实验。

---

# 6. 对“代码实现是否有问题”的专项回答

## 6.1 已发现并修复过的真实 bug

项目中并不是从一开始就完全正确。已经发现并修复：

1. novel warp 对 source target 的视角错配；
2. 非标准逐步 background latent 回填；
3. Reference feature 同 channel 不同尺度被覆盖；
4. validation shuffle 与随机 novel view；
5. validation identity broadcasting；
6. 正式 infer 仍加载 LaMa checkpoint；
7. resume 重复 step / 新建错误目录；
8. GOAE/Swin CPU/CUDA tensor 冲突；
9. raw x0 loss 高噪声爆炸；
10. R1 对 PatchGAN 所有 patch 求和导致数值异常；
11. 先 full decode 再 resize 无法解决 OOM；
12. mirror-ref 几何标签不严格。

因此我们没有假设代码天然正确，而是通过因果消融、固定验证和数学核查逐步修正。

## 6.2 当前已经核查的关键点

### checkpoint

- v3 明确检查 `architecture_version=3`；
- 检查 `supervision=novel_view_target`；
- 检查每个 RCA processor 是否存在；
- 正式推理加载 W mapper 和全部 RCA state；
- v2/v3 feature 格式按版本区分。

### VAE scaling

训练与推理均使用约 0.18215 / `vae.config.scaling_factor`。我们还做了 source photo VAE encode/decode 下限测试：

```text
source:    lap=0.05122, micro=0.03185
VAE recon: lap=0.05076, micro=0.03099
L1=0.02286
```

VAE 只有轻微重建损失，并没有把真实照片直接变成当前程度的油画图。

### mask 与 compositing

- mask=1 表示 hole；
- training、validation、inference 语义一致；
- known-region 最终精确保留，因此 `known_l1=0` 是设计结果；
- raw 输出单独保存，可观察 generator 本身，不会被 compositing 掩盖。

### W+ 与 Reference 是否真正生效

最终 300k 同 seed 条件消融：

```text
no Reference 与 full 平均差：0.08551
no W+        与 full 平均差：0.12835
Brush only   与 full 平均差：0.13440
```

说明条件分支确实参与生成，不是“训练了但推理没传进去”。

### 训练/推理一致性

- 同一个 VAE；
- 同一个 UNet；
- 同一个 BrushNet；
- 同一个 empty prompt；
- 同一个 W mapper；
- 同一个 Reference extractor；
- validation 和 infer 共用采样函数。

## 6.3 当前仍属于方法风险而不是明确代码 bug 的部分

1. Reference attention 仍是全局 token attention，没有明确源目标几何对应；
2. v4 已解决 Reference K/V/O 容量问题，但 feature 仍停留在 source 坐标；
3. W+ 与 Reference gate 可以达到较大正负值，但没有层级正则或可解释约束；
4. target latent 训练用 posterior sample，而 inference condition/reference 用 posterior mode；这是常见做法，但可做 deterministic ablation；
5. 当前没有显式 camera token，目标 pose 只通过 warp/mask 间接表达；
6. hard `t<200` 与 continuous alpha-bar 都已实测；连续目标改善 novel 指标但损伤 real-ref Pareto，因此当前保留 hard 默认；
7. patch 通过 `argmax` 选择最大 hole window，是有效的显存策略，但具有离散选择。

这些不等于实现错误，但应在下一版本做更优雅的建模。

---

# 7. 300k 最终结果与油画感证据链

## 7.1 不是未拟合

最终：

```text
warp hole L1 = 0.334948
model hole L1 = 0.101007
```

模型有约 0.234 的绝对改善。W+ 在最终日志中：

```text
correct x0 loss ≈ 0.0201
wrong-W+ x0 loss ≈ 0.1765
contrast loss = 0
```

说明正确身份已经显著优于错误身份。

## 7.2 真实像素监督覆盖率

日志统计：

```text
logged real-ref blocks = 3000
pixel supervision events = 1043
event rate = 34.77%
```

为什么不是 20%？因为 batch=2，只要两个样本至少一个满足 `t<200` 就会记录一次 pixel event：

```text
1 - 0.8² = 36%
```

实际 34.77% 与理论一致，说明 timestep selection 没有明显 bug。

但从总体 step 看：

```text
real-ref 占 25%
其中约 35% batch 触发 pixel loss
=> 约 8.7% 总 step
```

从单样本看，只有约 5% 全部训练样本接受真实 LPIPS/GAN，而 75% 任务持续接受 EG3D/synthetic latent supervision。

## 7.3 条件分支消融

同一最终 checkpoint、同一输入、同一 seed：

| 条件 | Laplacian | microtexture |
|---|---:|---:|
| source real photo | 0.05122 | 0.03185 |
| EG3D target | 0.02048 | 0.01253 |
| full | 0.05459 | 0.02549 |
| no Reference | 0.07894 | 0.04246 |
| no W+ | 0.16546 | 0.07816 |
| BrushNet only | 0.17201 | 0.08241 |
| full DDIM | 0.05792 | 0.02572 |

高频数值较高并不一定代表更真实，也可能是噪点和伪影。但该实验明确说明：

- full 模型的平滑不是 sampler 强制产生的；
- W+ 与 Reference 强烈改变输出；
- 两个条件联合作用后输出更接近平滑目标域；
- DDIM 与 DPM++ 结果接近。

## 7.4 我们目前对油画感的解释

当前模型面对的优化力量是：

```text
75%：密集 noise/x0 loss → EG3D/synthetic RGB latent
25%：real-ref latent loss → real photo
约 5% 样本：真实 LPIPS/GAN
```

v3 中真正负责 source texture 的 Reference 路径主要只有 gate 可训练；v4 已用独立 K/V/O
证明 real-ref 能力可从 `0.23990` 提升到约 `0.157`。当前剩余问题已从“容量不足”转为“跨视角坐标不对齐”。

所以合理的收敛点是：

```text
BrushNet 提供几何
W+ 提供身份
Reference 对结果有影响
但整体材质仍被大量 EG3D latent target 拉向平滑域
```

这就是“结构已学会、油画感稳定存在”的原因。

---

# 8. 目前面临的核心问题

## 8.1 伪目标同时承担几何与外观监督

我们信任 EG3D 的：

- target camera pose；
- depth 和 visibility；
- silhouette；
- identity；
- 低频三维结构。

但不应完全信任其：

- 皮肤微纹理；
- 头发细节；
- 高频真实感；
- 局部反射与光照。

当前全频 latent noise/x0 loss 没有区分这两类信息。

## 8.2 Reference 跨视角空间对应不足

v3 Reference 路径基本是：

```text
frozen Reference feature
→ frozen SD K/V/O
→ one scalar gate per layer
```

v4 已通过独立 K/V/O 解决这一容量问题，并显著改善 real-ref；但 Reference feature 仍位于
source 坐标，目标 query 位于 novel-view 坐标。全局 attention 必须自行搜索对应位置，仍容易平均化或错配。

## 8.3 real-ref 与最终任务存在 task gap

real-ref 是：

```text
同视角 source photo 挖洞 → 同一 source photo
```

最终任务是：

```text
source view → novel view 新出现区域
```

前者保证精确真实照片监督，但没有直接教跨视角纹理迁移。

## 8.4 缺少显式 camera conditioning

当前目标相机只通过 warp image 和 mask 间接进入 diffusion。小角度时足够，但大空洞区域中可能缺少明确 pose 信息。

## 8.5 视频一致性

每个 video frame 独立 diffusion sampling。即使 seed 固定，不同条件下也可能出现 temporal flicker。当前训练与验证主要是单帧，还没有系统评价时序一致性。

## 8.6 当前验证指标不等价于真实 novel-view GT

`hole_l1` 对比的是 EG3D novel pseudo target。该指标越低，只能说明更接近 EG3D，不保证更接近真实照片。

因此后续必须同时报告：

- EG3D pseudo-target consistency；
- photorealism；
- identity；
- pose consistency；
- temporal consistency。

---

# 9. 下一阶段修改方向

## 9.1 已完成：Reference 独立 K/V/O adapter

为 Reference 分支增加：

```text
to_k_reference
to_v_reference
to_out_reference
reference_scale
```

数学形式：

```text
h_out = h_SD
      + tanh(g_w) × A_w(h, W+)
      + tanh(g_r) × A_r(h, F_ref)
```

其中 `A_r` 不再完全复用冻结 SD 投影。

优点：

- 保持 residual 结构；
- gate=0 时仍严格等价于预训练底座；
- real-ref LPIPS/GAN 可以学习“如何使用纹理”，而不只是“使用多少纹理”；
- 修改范围小，便于单独消融。

## 9.2 已完成但不采用：连续 alpha-bar photorealism weighting

当前：

```python
select = timesteps < 200
```

有 x0 可靠性的理论动机，但在 199/200 之间不连续。

v4-C 实际实现并公平 resume 比较：

```text
w_photo(t) = alpha_bar(t)^gamma = [SNR/(1+SNR)]^gamma
```

它将 v4-C step 400 的 novel L1 改善到 `0.09735`，但 real-ref L1 回退到 `0.19386`，
未占据 v4-B 的 Pareto 前沿；同时完整 pixel graph 峰值显存约从 `13.36` 增至 `18.06 GiB`。
因此配置默认恢复 `hard_low_noise`，实验入口保留在 `scripts/run_v4c_continuous_pilot.py`。

## 9.3 分离 EG3D geometry 与 appearance

不要再让 EG3D 全频 RGB 同时成为结构和材质真值。

EG3D 建议监督：

- depth；
- normal；
- visibility；
- silhouette；
- landmark；
- pose；
- identity feature；
- 多尺度低频结构。

真实照片建议监督：

- LPIPS；
- real-photo discriminator；
- face identity；
- 局部纹理统计；
- Reference consistency。

不建议用一个人为频率阈值直接切高低频。更优雅的是连续 Gaussian scale-space、多尺度 feature loss 或明确几何表示。

## 9.4 高分辨率 UNet LoRA

若 Reference adapter 仍不足，可以在：

- high-resolution self-attention；
- up blocks；
- 部分高分辨率 ResNet；

加入低秩 LoRA，并主要由 real-ref/真实照片任务更新。

不建议立刻全量解冻 UNet，因为：

- 显存与优化难度大；
- 容易破坏 BrushNet/SD 预训练；
- 难以判断哪项修改有效。

## 9.5 显式 camera token

若移除 EG3D RGB 外观约束后姿态变差，可加入：

```text
(c_novel - c_source) → camera MLP → camera tokens
```

或者将 novel depth/normal/visibility 作为结构条件。

不建议重新强条件输入 `y_hat_novel` RGB，因为它会重新把 EG3D 材质直接带入结果。

## 9.6 真实多视角或更严格合成数据

从根本上解决 task gap，需要至少一种：

- 真实多视角人脸数据；
- 可控真实感 renderer；
- video 中经过姿态和光流筛选的跨帧配对；
- 高质量 synthetic-to-real domain adaptation。

---

# 10. 不建议的修改

为了保证理论和代码优雅，不建议：

- 对输出直接 sharpen；
- 高频增强后处理；
- 根据像素值硬切“油画区域”；
- 用边缘 mask 强制替换结果；
- 在某个 timestep 后直接 bypass UNet；
- 用手工阈值选择 Reference 或 W+；
- 为了姿态重新把 EG3D novel RGB 强塞入 hole；
- 只增加 DDIM 步数但不改变监督；
- 一次同时改五个模块，导致无法因果归因。

可以保留的合理硬约束：

- 几何 visibility mask；
- 最终 known-region data consistency；
- checkpoint architecture version 检查；
- 合法 shape/版本断言。

这些是任务定义或安全边界，不是为了掩盖视觉问题的 hack。

---

# 11. 建议的下一轮实验矩阵

## 11.1 已完成：导师关心的 sampler/steps 表

固定：

- checkpoint=300k；
- 三个 validation identities；
- novel view=1；
- seed=42；
- W+/Reference/BrushNet 全开。

运行：

```text
DDIM: 20 / 50 / 100
DPM++: 20 / 50 / 100
```

报告：

- raw image；
- composed image；
- hole L1；
- LPIPS；
- ID；
- NIQE（辅助）；
- runtime；
- Laplacian/microtexture（只作诊断，不作为真实性主指标）。

该实验成本低，能够正式回应“是不是步数问题”。

## 11.2 下一轮单变量实验

| 实验 | Reference adapter | Reference 坐标 | photo weight | 目标 |
|---|---|---|---|---|
| v4-B baseline | K/V/O | source/global | hard t<200 | 当前 Pareto |
| D1 | K/V/O | target-aligned feature warp | hard t<200 | 验证空间对应 |
| D2 | K/V/O | warped + global fallback | hard t<200 | 处理不可见区 |
| D3 | K/V/O | D2 + soft confidence | hard t<200 | 避免错误对应 |
| D4 | K/V/O | D3 | hard t<200 | 再决定 camera token |

每次只增加一项，至少在 5k/20k/50k 固定 checkpoint 做相同消融。

## 11.3 模块因果消融

每个版本至少输出：

- full；
- no W+；
- wrong W+；
- no Reference；
- no BrushNet；
- source VAE reconstruction；
- warp only；
- EG3D novel render。

这样可以判断：

- W+ 是否负责身份；
- Reference 是否负责纹理；
- BrushNet 是否负责结构；
- VAE 是否限制高频；
- EG3D 是否在控制材质域。

---

# 12. 关键文件修改索引

| 文件 | 原版 | 当前修改 |
|---|---|---|
| `training/coach_inpainting_static.py` | FFC generator + inverse cycle + legacy losses | diffusion 三任务、noise/x0 loss、W+ contrast、real-ref LPIPS/GAN、确定性验证、v3 checkpoint |
| `models/mapper/w_proj.py` | 不存在 | W+ `[14,512]` 映射为 SD 768-dim tokens |
| `models/referencenet/attention_processor.py` | 不存在 | 保留 SD base attention，加入 W+ RCA 与 Reference residual |
| `models/referencenet/unet_2d_condition.py` | 不存在于原版主工程 | ReferenceNet 多尺度 feature extraction |
| `utils/diffusion_inpainting.py` | 不存在 | 统一 DDIM/DPM++ sampling，验证与正式推理共用 |
| `models/diffusion_inpaintor.py` | 不存在 | 生产级 diffusion checkpoint loader |
| `scripts/infer.py` | 只支持 LaMa/FFC | diffusion/legacy 双后端，diffusion 不再需要 EG3D novel RGB |
| `datasets/dataset_inpainting_static.py` | test novel view 随机 | 增加 `fixed_novel_view` |
| `scripts/train_inpainting.py` | 每次新实验目录 | v3 resume 原目录续写、下一 step 继续 |
| `models/goae/swin_transformer.py` | clamp max 临时 CPU tensor | 改为等价 Python scalar，修复 CUDA 冲突 |
| `configs/train_inpainting.yaml` | LaMa/FFC 配置 | 增加 pixel supervision 与 diffusion ablation 配置 |
| `configs/infer.yaml` | legacy inpaint | 默认 diffusion backend 与采样配置 |

---

# 13. 证据与产物路径

## 13.0 复现前必须注意的配置状态

当前 `configs/infer.yaml` 已设置：

```yaml
inpainting_backend: diffusion
```

并默认指向当前 v4 Pareto checkpoint：

```text
experiments/_v4b_factorized_pilot_500/checkpoints/iteration_200.pt
```

因此直接运行 `scripts/infer.py` 会使用当前推荐的 v4-B step 200。若需要复现历史 v3 300k
scheduler 基线，再手动切换到对应历史 checkpoint。

## 13.0.1 原版和当前指标不能直接横向比较

原版 SVINet 验证主要计算：

```text
novel result → inverse warp / inpaint → source view reconstruction loss
```

当前 diffusion 验证主要计算：

```text
novel result hole region ↔ EG3D novel-view pseudo target
```

两者衡量的对象不同：

- 原版更偏 source-view cycle consistency；
- 当前更偏目标视角 pseudo-target consistency；
- 二者都不等价于真实 novel-view GT realism。

因此不能直接用原版日志中的 `gen_loss` 与当前 `hole_l1` 数字宣布方法优劣。公平对比必须在同一批 source、同一目标相机、同一 warp/mask 上重新运行两种方法，再统一计算 ID、LPIPS、pose、FID/KID/NIQE 和人工视觉评价。

## 最终训练

```text
experiments/train_inpaintor/[20260722-220118]_rca_v3_full/
```

包括：

- `checkpoints/iteration_300000.pt`
- `logs/val_metrics.txt`
- `logs/debug_tensor_health.txt`
- `logs/images/val/val_step_300000.png`

## scheduler / 条件分支消融

```text
experiments/_oil_ablation/seed42_branch_scheduler_ablation.png
```

包含：

- source；
- EG3D target；
- warp；
- full DPM++；
- no Reference；
- no W+；
- BrushNet only；
- full DDIM。

## VAE 下限

```text
experiments/_oil_ablation/source_vae_reconstruction.png
```

## 历史记录

- `docs/PROJECT_SUMMARY.md`
- `docs/TRAINING_AUDIT_REPORT.md`
- `experiments/_final_cycle_audit*/`
- `experiments/_smoke_train_v2/`
- `experiments/_stress_pixel_batch2/`

注意：前两份历史文档生成时 v3 尚未完成 300k，因此其中“建议开始 v3 训练”等内容是当时状态。本报告使用 300k 结果更新了最终判断。

---

# 14. 面向导师的最终回答

## “是不是 DDIM 步数不对？”

当前默认验证不是 DDIM，而是 50-step DPM-Solver++；训练使用标准随机 DDPM timestep，不依赖 50-step sampler。20/50/100 步严格表格已经完成，DPM++ 50→100 仅改善约 `0.00006` 且耗时翻倍，因此 sampler 不是主要根因。

## “是不是代码实现有问题？”

项目中确实发现过目标错配、非标准 background injection、Reference 尺度覆盖、验证广播等 bug，这些已逐项修复。当前版本的 checkpoint、VAE scaling、mask、sampling、训练/推理链路和条件传递均有代码与消融证据支持。不能声称绝对不存在任何问题，但当前油画感更符合监督域和可训练容量问题，而不是某一行明显实现错误。

## “为什么训练 300k 还油画？”

因为训练不是没收敛，而是大部分目标要求模型接近 EG3D/synthetic 渲染域。真实感监督较少，且 Reference 路径容量有限，所以最终学到的是结构和身份正确但材质偏平滑的解。

## “下一步怎么做？”

sampler step 表、Reference K/V/O、appearance pretrain、factorized joint 和连续权重对照均已完成。
当前应以 v4-B step 200 为 baseline，单独实现 target-aligned Reference feature warp；
不要继续盲目增加训练步数，也不要通过锐化或硬阈值掩盖问题。

---

# 15. 从“按比例轮流训练”走向完整 Inpaintor

## 15.1 当前问题不能只概括成“25%/75% 比例不对”

把当前问题描述为：

```text
25% 来自真实照片，75% 来自 EG3D，所以把比例改一下即可
```

并不准确，也不够优雅。

任何多数据源训练都必须隐含或显式定义各数据分布的权重，所以“存在比例”本身不是错误。当前真正的问题是：

> 我们使用离散 batch 类型和固定 phase 比例，间接规定了 BrushNet、W+、ReferenceNet 应该学什么；但不同 batch 的目标域不同，它们又共同更新部分相同参数，因此条件职责没有被明确分离。

例如：

- real/synth pseudo target 希望条件分支接近 EG3D RGB；
- real-ref 希望 Reference 分支恢复真实照片；
- W+ 与 Reference 同时影响 UNet，但只有 real-ref 禁用 W+；
- 真实感监督和结构监督不是在同一个 novel-view 样本上同时成立。

因此当前系统更像三个可运行的任务轮流优化同一个生成器，而不是一个统一定义的条件概率模型。

## 15.2 完整 Inpaintor 应建模什么

理想目标是：

```text
p(x_target | observed_warp, hole_mask, identity, source_appearance, target_camera)
```

每一个 novel-view 样本都应同时满足：

1. **数据一致性**：已知区服从真实 warp observation；
2. **几何一致性**：输出服从 target camera、depth、visibility 和脸部结构；
3. **身份一致性**：输出仍是源身份；
4. **外观一致性**：皮肤、头发和局部特征来自真实 source photo；
5. **自然图像先验**：不可见区域由 diffusion prior 合理生成，而不是复制 EG3D 材质。

统一目标可以写成：

```text
L = λdata Ldata
  + λgeo  Lgeometry
  + λpose Lpose
  + λid   Lidentity
  + λapp  Lappearance
  + λreal Lrealism
  + λprior Lprior
```

这些项的权重仍然需要定义，但它们表达的是明确的概率约束和物理职责，而不是“每四步中哪一步出现哪种 target”。

可进一步使用：

- uncertainty weighting；
- GradNorm；
- 每项按有效像素或有效 token 归一化；
- 按梯度尺度动态平衡；

减少人工固定比例造成的某一目标长期主导。

## 15.3 现有 BrushNet、W+ 和 ReferenceNet 是否足够

从信息种类看，三者已经覆盖了大部分必要条件：

| 条件 | 应负责的信息 |
|---|---|
| BrushNet | 哪些目标视角像素已有观测、空洞在哪里、局部结构如何延续 |
| W+ | 身份、脸型、语义属性和 EG3D identity prior |
| ReferenceNet | 真实 source photo 的皮肤、发型、局部颜色和高频外观 |
| SD/BrushNet prior | 对不可见区域生成自然且合理的内容 |

所以现阶段没有证据说明必须再加入一个与 BrushNet 或 ReferenceNet 同级的新大型生成 backbone。

问题主要是：

1. v4 已让 Reference 信息可训练地映射，但缺少跨视角对应；
2. Reference feature 没有与 target view 几何对齐；
3. target camera 没有独立条件表示；
4. loss 没有按条件职责路由；
5. 缺少真实 novel-view GT 是一个数据层面的根本限制。

## 15.4 最低可行 v4：不增加新大型模型

### 15.4.1 已完成：补全 Reference adapter

v3 Reference 路径主要为：

```text
Q = frozen SD query
K/V/O = frozen SD projection
trainable = one scalar gate per layer
```

v4 已改为：

```text
Q = target UNet query
Kref = trainable projection(reference feature)
Vref = trainable projection(reference feature)
Oref = trainable output projection
gref = zero-initialized residual gate
```

即：

```text
Href = Oref(Attention(Qtarget, Kref(Fsource), Vref(Fsource)))
Hout = Hbase + tanh(gref) × Href
```

第一版可以共享 target query，不必增加独立 `Qref`。这样既减少参数，也符合语义：目标位置提出需求，源图特征提供可用外观。

这不是增加新模型，而是把现有 ReferenceNet 条件通路补完整。

### 15.4.2 使用真实照片先训练 appearance capability

先执行一个职责明确的预训练阶段：

```text
masked real source + full source reference → original real source
```

mask 从真实 WarpGAN novel-view visibility 分布中采样，而不是使用普通矩形 mask。

该阶段：

- 训练 Reference K/V/O；
- 可选训练高分辨率 LoRA；
- 不训练 W+；
- 不使用 EG3D novel RGB；
- 使用标准 real-photo diffusion/noise objective、LPIPS 和 real discriminator。

这不是把 real-ref 永久设成 25%，而是先学习一个明确能力：

> 给定真实人脸 reference，模型能够生成真实照片域的缺失纹理。

### 15.4.3 联合阶段不再按 phase 隐式分工

联合阶段中，每个真实 source batch 同时构造：

1. **self-reconstruction branch**：保持 Reference 的真实照片能力；
2. **novel-view branch**：学习几何、姿态、身份和新视角生成。

两个分支在同一个 optimizer step 中形成总目标，而不是 `global_step % 4` 决定本步只学习其中一个。

如果计算成本过高，可以随机估计某些辅助项，但采样概率应从损失估计和计算预算推导，并进行归一化，而不是让 batch 类型决定模型语义。

## 15.5 参数职责应显式路由

完整系统不应让所有 loss 无差别更新所有 adapter。

建议的职责为：

| Loss | 主要更新参数 | 不应主导的参数 |
|---|---|---|
| real-photo score / LPIPS / GAN | Reference K/V/O、高分辨率 LoRA | W+ identity adapter |
| identity loss / wrong-W+ contrast | W mapper、W+ RCA | Reference appearance adapter |
| pose/depth/landmark loss | camera adapter、结构 adapter | appearance adapter |
| known-region consistency | Brush/结构路径、融合层 | 不应强迫 Reference 复制已知区 |
| EG3D structural target | camera/geometry/W+ identity | Reference、高分辨率 appearance LoRA |

这种梯度路由并不是为了遮盖失败而“直接拦截”，而是由条件概率分解决定的模块化优化：

```text
identity loss 监督 identity parameters
appearance loss 监督 appearance parameters
geometry loss 监督 geometry parameters
```

它比让 EG3D RGB loss 同时更新 W+ 和 Reference 更符合模型设计。

## 15.6 应如何继续使用 EG3D target

EG3D 不应被完全丢弃。它仍是当前唯一具有 target-view 三维信息的 teacher。

但应将其从“完整 RGB 外观真值”降级为“结构 teacher”。

### 推荐保留

- depth；
- visibility；
- target pose；
- silhouette；
- landmark；
- identity feature；
- 连续多尺度低频结构；
- source→target correspondence。

### 不应强制

- 全频 RGB latent noise target；
- 皮肤高频；
- 发丝细节；
- 局部真实光照和反射。

如果仍需 RGB 结构约束，可采用连续 scale-space：

```text
Lscale = Σs λs ||Gσs(xpred) - Gσs(xEG3D)||
```

其中权重随空间频率连续衰减，而不是人为设定一个硬频率区间。

更优雅的方案是直接监督 depth、landmark、segmentation 和 identity feature，避免 RGB 材质泄漏。

## 15.7 建议增加的轻量新内容

现有三模块信息接近完整，但还缺少两个明确条件。它们不需要新的大型 backbone。

### 15.7.1 Relative Camera Encoder

当前 target camera 只通过 warp/mask 间接表达。当 hole 很大时，那里没有 warp pixel，UNet 很难知道：

- 目标 yaw/pitch；
- 应出现左脸还是右脸；
- 耳朵应出现多少；
- 遮挡边界应如何变化。

建议：

```text
relative_camera = representation(c_novel relative to c_source)
relative_camera → small MLP → 4–8 camera tokens
```

条件职责变为：

```text
W+ tokens：这个人是谁
camera tokens：目标视角是什么
Reference features：这个人的真实外观是什么
```

这是最值得新增的轻量模块。

### 15.7.2 Target-aligned Reference Feature Warp

当前 Reference attention 是全局 all-to-all。目标位置要从源图所有位置自行搜索纹理，容易产生平均化和错误对应。

更符合 WarpGAN 核心思想的是：

```text
source multiscale Reference features
        ↓
使用 source depth 和 c_source→c_novel 做 feature warp
        ↓
target-aligned Reference features + continuous confidence
        ↓
Reference K/V/O adapter
```

可见区域使用几何对齐特征；不可见区域由：

- global Reference fallback；
- W+ identity；
- diffusion prior；

共同生成。

confidence 应来自软 splatting weight、depth consistency 或 forward/backward consistency，而不是人为二值阈值。这样不会引入硬边缘 hack。

Feature warp 可以复用现有 Warper 和相机/depth，不需要引入新大型网络。

## 15.8 是否需要 LoRA

Reference adapter 能传入纹理，但冻结 SD UNet 未必有足够自由度把这些特征转换成真实人脸新视角材质。

如果 Reference adapter 预训练后：

- real self-inpainting 明显改善；
- Reference 因果消融有效；
- novel view 仍然平滑；

再在高分辨率层加入小型 LoRA：

- up blocks；
- 高分辨率 self-attention；
- 必要的高分辨率 ResNet。

LoRA 应主要由真实照片 appearance loss 更新，避免再次被 EG3D RGB 域污染。

LoRA 是现有 SD 的轻量适配，不是新的生成 backbone。

## 15.9 一个更完整的 v4 架构

```text
                         relative camera
c_source, c_novel ─────→ Camera MLP ───────────→ camera tokens

W+ ────────────────────→ W mapper ─────────────→ identity tokens

source photo ─→ ReferenceNet ─→ feature warp ─→ Reference K/V/O
                     ↑              ↑
                source depth    camera transform

source RGB + depth + cameras ─→ RGB warp + mask ─→ BrushNet

              BrushNet structural residual
              + camera pose tokens
              + W+ identity tokens
              + target-aligned real appearance
                           ↓
                 SD UNet (+ optional LoRA)
                           ↓
                    novel-view inpainting
```

各模块职责可以一句话说明：

```text
BrushNet：目标视角哪里已有观测、哪里需要生成
Camera：目标视角是什么
W+：这个人是谁
Reference：这个人的真实外观是什么
SD/LoRA：如何生成自然图像和不可见内容
```

相比 25/50/25 phase，这种结构更容易解释、训练和消融，也更适合作为论文方法。

## 15.10 是否还需要新的大型内容

### 当前阶段：不需要

建议先完成：

1. Reference K/V/O adapter；
2. real-photo appearance pretraining；
3. 参数职责与 loss 路由；
4. EG3D geometry/appearance 解耦；
5. camera tokens；
6. target-aligned Reference feature warp；
7. 必要时 high-resolution LoRA。

这些都属于现有 BrushNet + ReferenceNet + W+ + SD 系统的完整化。

### 何时才需要新大型模块或新数据

如果上述方案完成后仍出现：

- 极大 yaw 下不可见脸侧无法保持身份；
- 源图不可见的耳朵、发型无法稳定生成；
- 视频中存在严重闪烁；
- 单图无法确定的内容要求精确复原；

那么问题已经超出普通单图 inpainting，进入：

- 单图不可见区域的三维生成；
- 多视角一致性；
- 时序一致性；
- 数据不可辨识性。

这时可能需要：

- EG3D triplane feature conditioning；
- 更强的 3D feature volume；
- deformable/correspondence attention；
- temporal module；
- 真实多视角训练数据。

其中**新数据或三维对齐信息的价值通常高于再增加一个普通图像 backbone**。因为单张 source photo 中不存在的纹理无法被任何网络精确恢复，只能由 prior 合理猜测。

## 15.11 推荐实施顺序

### v4-A：完成 Reference 能力

- 增加 Reference K/V/O；
- output/gate 零初始化；
- 用真实照片和 view-shaped masks 预训练；
- 做 full/no-Reference 因果消融。

### v4-B：统一职责训练

- 每个 real batch 同时构造 self-reconstruction 和 novel branch；
- appearance loss 只更新 appearance parameters；
- identity loss 只更新 W+ parameters；
- geometry loss 更新 camera/structure parameters；
- 不再依赖 `global_step % 4` 定义模块语义。

### v4-C：geometry/appearance 解耦

- EG3D 只提供结构、pose、depth、identity teacher；
- 真实照片提供外观与真实感；
- 逐步删除 EG3D 全频 RGB 对 appearance adapter 的梯度。

### v4-D：加入 camera tokens 和 feature warp

- relative camera MLP；
- multiscale Reference feature warp；
- soft geometric confidence；
- global Reference 作为不可见区域 fallback。

### v4-E：可选 LoRA

- 只有前四步仍无法跨越材质域时加入；
- 先限制在高分辨率层；
- 由真实照片 appearance objective 主导。

## 15.12 最终判断

当前问题的核心确实位于训练和监督设计，但不是简单修改 25%/75% 比例。

更准确的表述是：

> 当前系统用不同 batch 的出现比例代替了几何、身份和外观的显式因子分解，且真实外观分支容量不足，因此不同目标域共同更新同一条件网络时，EG3D 平滑材质成为稳定收敛点。

现有 BrushNet、ReferenceNet、W+ 和 SD prior 已包含完成任务的大部分信息。近期不需要再引入新的大型生成模型；应优先补完整 Reference adapter、加入轻量 camera condition、对齐 Reference features，并建立统一且职责明确的目标函数。

只有在这些完成后仍无法解决大视角不可见区域和视频一致性时，才有充分理由加入新的 3D/temporal 内容或真实多视角数据。

### 0.0.3 2026-07-30 v4-fullframe 10k 实验失败分析与从零重启决策

**实验**：`[20260729-114621]_rca_v4_fullframe`，从上一轮 factorized step-9500 best 权重初始化，full-frame SD 输出契约，训练 10000 步。

**硬指标表现**：

| step | novel_full | novel_hole | real_ref_full | real_ref_hole | real_ref_visible |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.079595 | 0.090100 | 0.098919 | 0.150175 | 0.080459 |
| 5000 | 0.076178 | 0.086142 | 0.088458 | 0.143727 | 0.068552 |
| 9500 | 0.076650 | 0.087508 | 0.080760 | 0.137646 | 0.060272 |
| 10000 | 0.076932 | 0.086787 | 0.081068 | 0.141199 | 0.059411 |

指标持续改善，best_model.pt 选在 step 9500。

**用户肉眼观察**：

NOVEL 行：
1. SD output 存在浓厚油画感，与修改前几乎一致；
2. 肌肤/头发纹理扰动感强，眼睛周围毛发淡化、睫毛缺失、瞳孔不对称；
3. 视角与 EG3D reference 基本一致，目标姿态正确；
4. warp 虽有空洞但去除空洞后尚自然，SD output 虽无空洞但非常不自然；
5. 瞳孔方向不一致，眼睑缺失。

REAL-REF 行：
1. 身份未变；
2. 未挖洞区域出现变色风格化；
3. 背景挖洞区域有时生成杂色；
4. 生成质量随步数增加未改善。

**结论：指标改善 ≠ 视觉真实感改善**

real_ref_full 从 0.099 降到 0.081，real_ref_visible 从 0.080 降到 0.060，但这些改善来自 latent L2 对平滑目标的回归收敛和 PatchGAN 低频判别，而非真实高频纹理学习。油画感未被消除，因为初始化权重本身已携带油画域。

**决策：从零门控适配器重新开始**

- `initialization_path: null`
- `max_steps: 300000`
- `arm_after_first_pass: True`
- `align_reference_features: True`
- PatchGAN 从头初始化
