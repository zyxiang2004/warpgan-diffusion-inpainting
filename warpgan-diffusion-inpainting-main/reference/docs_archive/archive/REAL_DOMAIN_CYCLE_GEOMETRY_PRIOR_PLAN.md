# 真实域 cycle 几何约束训练方案：EG3D 低频几何 + W+ 身份 + Reference/BrushNet 高频补全

> 本文档记录当前 NOVEL hole 碎孔问题之后的方案收敛。
> 核心目标不是继续在 mask、Q/K/V/O 或阈值上搜索，而是重新定义训练第一性原则：
> 在无真实 novel-view 配对 GT 的条件下，用真实 source photo 的 cycle 监督训练一个
> reference-guided novel-view inpainting prior；EG3D 只提供低频几何，W+ 只提供身份/低频形状，
> 高频真实纹理由 ReferenceNet、BrushNet/SD 先验和真实域训练获得。

---

## 1. 结论确认：用户提出的思路没有原则性错误

用户提出的方向可以概括为：

```text
真实 source photo x
  -> 采样多个 target camera c_novel
  -> 用 3D warp 得到破碎 novel warp + hole mask
  -> 用 EG3D depth / silhouette / low-pass render 提供 target-view 低频几何
  -> 用 W+ 提供身份 / 脸型 / 低频先验
  -> 用 ReferenceNet + BrushNet/SD prior 补高频纹理
  -> 生成 pred_novel
  -> inverse warp 回 source view
  -> 与真实 source photo x 做真实域 pixel / perceptual / ID / realism 监督
```

这个思路是可行的，且与任务本质一致。WarpGAN 原版和当前项目都没有真实配对 novel-view GT，
任务本来就是单张图片条件下“由已知 source view 推未知 target view”。因此不能要求模型拥有
真实 target-view 高频像素答案；合理目标是生成 identity-consistent、geometry-consistent、photo-realistic
的合理新视角补全。

当前问题的根源不是“没有配对 GT”，而是在没有配对 GT 的设定下，NOVEL 分支过度依赖 EG3D
pseudo RGB / latent reconstruction，并让高容量 W+ cross-attention 分支直接追这个伪目标。该职责错配
在 production 中表现为 hole 单像素黑孔与碎裂纹理，在 shared/external contract 中表现为全局脏化、
油画化和身份崩坏。

---

## 2. 与 warpgan_orig 的关系

### 2.1 原版也没有真实 novel-view 配对 GT

原版 WarpGAN 的训练方式是：

```text
source photo -> forward warp to novel -> inpaint pred_novel -> inverse warp back source -> compare source photo
```

它没有真实 target novel image。真实监督来自 source view，因此它同样是在单图条件下用 cycle
解决未知视角补全。

### 2.2 原版的关键优点

原版的第一性原则有三点值得继承：

1. **target-view 几何不是让 W+ 硬猜。** 原版 `warp.hybrid=True` 时，hole 内直接填入 EG3D novel render，
   给生成器逐像素的 target-view 结构草图。
2. **真实 appearance 监督绕回 source view。** 原版通过 inverse warp 回源视角后和真实 source photo 做
   pixel、ID、GAN 等监督。
3. **W+ 不是 SD cross-attention 空间残差写入器。** 原版 W+ 是 FFC style modulation，不存在当前项目
   16 个冻结 SD attention 层上的独立可训练 `to_out_wplus` 空间残差出口。

### 2.3 原版的局限不能照搬

原版的 `hybrid` 使用 EG3D full RGB，会带来 EG3D 平滑材质、塑料感和油画感。当前方案不能把 EG3D
恢复成高频 appearance target，而应把 EG3D 降级为：

```text
depth / silhouette / visibility / low-pass layout / low-frequency geometry scaffold
```

换言之，EG3D 不再当“画师”，只当“几何传感器”。

---

## 3. 当前代码中已有的可复用先验

### 3.1 BrushNet：通用 inpainting 与空间条件先验

当前项目通过 `BrushNetModel.from_pretrained(...)` 加载 BrushNet，并将 `masked_image_latents + mask_latents`
作为 BrushNet condition。BrushNet 原项目定位是 plug-and-play image inpainting，它擅长利用 masked image
和 mask 产生 down/mid/up residual，辅助 SD UNet 做自然补洞与边界融合。

BrushNet 可以承担：

- hole 区自然补全；
- visible/hole 边界融合；
- 根据 mask 和低频 condition 生成照片级局部纹理；
- 利用 SD prior 避免黑洞与硬拼接。

BrushNet 不应单独承担：

- 当前身份记忆；
- target-view 几何推断；
- EG3D 高频材质重建。

### 3.2 ReferenceNet：source-specific appearance 先验

当前 ReferenceNet 来自 SD 1.5 UNet 权重，作为冻结 feature extractor 使用。它输入完整 source photo 的
VAE latent，提取多尺度特征。当前 `ReferenceAttentionProcessor` 在 self-attention 层注入：

- global reference attention residual：`to_k_reference / to_v_reference / to_out_reference / reference_scale`；
- local target-aligned residual：`reference_local_scale` 和 3D warp 后的 aligned reference features。

ReferenceNet 可以承担：

- 当前 source photo 的肤色、头发、眼睛、嘴唇、光照、高频统计；
- 真实照片域的 source-specific appearance；
- 通过真实照片 self-reconstruction 训练“如何使用 reference feature”。

ReferenceNet 不应单独承担：

- 不可见区域的 target-view 几何；
- 真实 target-view 高频 GT；
- 由 EG3D pseudo RGB 驱动的 appearance 学习。

### 3.3 W+ Adapter：身份先验，但必须降级

仓库中的外部 W+ Adapter 原项目目标是将 StyleGAN W+ 和 SD 对齐，实现身份保持与编辑。原项目的
W+ adapter 在 cross-attention 中使用 W+ K/V/Q residual，并强调 `residual_att_scale` 控制 identity 和
text alignment 的平衡。当前项目自定义 W+ RCA 更强：使用 16 个 cross-attention 层的独立 Q/K/V/O，
尤其有独立可训练 `to_out_wplus`。

因此 W+ 应该保留为：

- identity；
- 脸型；
- 五官比例；
- 低频人脸结构先验。

但必须禁止或强限制它承担：

- hole 高频纹理；
- pseudo RGB / x0 高频重建；
- 单像素空间残差；
- target-view 局部 appearance decoder。

---

## 4. 新方案的职责分工

| 模块 | 应承担 | 不应承担 |
|---|---|---|
| EG3D | target-view depth、silhouette、visibility、low-frequency geometry | 高频 RGB appearance target、LPIPS/GAN/x0 高频监督 |
| W+ | identity、脸型、低频结构、全局身份条件 | 空间高频纹理、单像素残差、pseudo RGB 重建 |
| ReferenceNet | source-specific 高频 appearance、真实照片纹理统计 | target-view 几何、不可见区域真实逐像素 GT |
| BrushNet / SD | 自然图像 inpainting prior、hole 融合、照片级补全 | 身份记忆、无几何约束的新视角结构猜测 |
| Cycle | 用真实 source photo 监督可回投区域、约束身份与纹理一致性 | 提供不可见区域唯一真实 GT |

一句话版本：

```text
EG3D tells where.
W+ tells who.
Reference tells texture.
BrushNet/SD makes it photorealistic.
Cycle says do not violate the real source photo.
```

---

## 5. 训练方案草案

### 5.1 阶段 A：真实照片 self-inpainting 外观能力预训练

目标：先让 Reference/BrushNet 学会从完整 source photo 中取真实高频，并在真实域补洞。

数据构造：

```text
target = source photo x
condition = x masked by novel-like hole mask
reference = full x
mask = 来自随机 target camera 的 forward-warp hole mask，或真实 novel mask 分布
W+ = disabled 或只作为冻结弱身份条件
```

监督：

```text
L1 / LPIPS / PatchGAN / ID / color consistency
```

参数建议：

```text
train: Reference adapter, optional BrushNet lightweight adapter / scale
freeze: SD UNet backbone, VAE, ReferenceNet backbone, BrushNet backbone initially, W+ RCA
```

这一步不使用 EG3D RGB 作为 target。它复用当前 REAL-REF 成功经验，但 mask 分布要尽可能模拟 novel hole。

### 5.2 阶段 B：真实域 multi-view cycle 训练

目标：训练真正的 novel-view inpainting prior。

对每张真实图片采样多个 target camera：

```text
x_source, c_source, W+, depth_source
target camera c_novel_1 ... c_novel_k
```

构造输入：

```text
warp_img = forward_warp(x_source, depth_source, c_source -> c_novel)
mask = 1 - visibility
geometry = EG3D depth / silhouette / low-pass target render
reference = x_source
wplus = W+
```

生成：

```text
pred_novel = diffusion_inpaint(warp_img, mask, geometry, reference, wplus)
pred_back = forward_warp(pred_novel, depth_novel or generated geometry, c_novel -> c_source)
```

监督：

```text
L_cycle_pixel / L_cycle_LPIPS / L_cycle_ID：pred_back vs x_source
L_boundary：visible/hole 边界一致性
L_lowfreq_geometry：pred_novel 的低频结构不违背 EG3D depth/silhouette/low-pass geometry
L_realism：PatchGAN 或已有真实域 adversarial，目标是真实照片域，不是 EG3D 域
```

关键约束：

```text
EG3D RGB 不作为高频 target。
NOVEL pseudo RGB/x0 loss 不直接训练 W+ 高频 RCA。
W+ 只接受 identity / low-frequency / contrastive / gated 条件训练。
Reference/BrushNet 接受真实域 cycle 与 realism 监督。
```

### 5.3 阶段 C：短程联合微调

仅当阶段 A/B 各自成功后，做低学习率联合微调以消除模块边界缝隙。

原则：

```text
不开 EG3D 高频 target。
不重新开放高容量 W+ 高频出口。
不全量微调 SD/ReferenceNet/BrushNet 主干。
```

---

## 6. 代码修改分支建议

### 分支 1：统一 geometry condition contract

现状审计发现，多个评估脚本调用 `build_geometry_condition`，但当前 `utils/diffusion_inpainting.py` 读取到的
生产函数只接受 `warp_image + mask`，没有统一的 `geometry_image` 参数。需要先做工程统一。

建议新增或恢复：

```python
build_geometry_condition(warp_image, mask, geometry_image, mode, lowpass_kernel)
```

支持模式：

```text
raw_warp
lowpass_rgb
depth_as_condition
silhouette
depth_silhouette
```

`sample_brushnet_inpainting` 增加可选参数：

```python
geometry_image=None
geometry_mode="none"
geometry_lowpass_kernel=31
```

默认行为保持生产兼容：没有 geometry 时仍使用 raw warp。

### 分支 2：新增真实域 cycle batch 构造

在 `training/coach_inpainting_static.py` 中新增 batch kind：

```text
real_cycle_novel
```

它与当前 real novel batch 不同：

- target 不再是 EG3D novel RGB；
- target supervision 来自 inverse warp back source；
- 可采样多个 target camera；
- EG3D 只产生 depth/silhouette/lowpass geometry condition。

### 分支 3：实现 differentiable 或 teacher-forced cycle loss

当前训练 loss 是单步 diffusion noise/x0 训练，不直接产生 50-step RGB，因此 cycle 可以分两层实现：

1. **低成本训练版**：在低噪声 timestep 预测 x0，decode 到 RGB，inverse warp 回 source，计算 cycle loss；
2. **验证版**：50-step DPM++ 采样后 inverse warp，计算真实 cycle 指标和肉眼图。

训练版要避免高噪声 x0 爆炸，只在低噪声 timestep 或 SNR 权重下启用。

### 分支 4：W+ 降级与梯度路由

必须避免重蹈当前问题。建议从最保守开始：

```text
W+ mapper / W+ RCA 不接收 EG3D pseudo RGB/x0 高频梯度。
W+ 可作为 frozen condition 参与前向。
若训练 W+，只训练低频/identity 相关目标，或只训练 mapper/gate，不训练独立 O。
```

可选实现：

- 冻结 `to_out_wplus`；
- 限制 W+ 只注入低分辨率 attention 层；
- W+ residual 做 layer/time scale schedule；
- 借鉴外部 W+ Adapter 的 scale 思路，但不要回到已失败的 shared contract 作为生产候选；
- 使用 stop-gradient：appearance/cycle high-frequency loss 不更新 W+ RCA。

### 分支 5：Reference/BrushNet 真实域能力训练

复用当前 REAL-REF 成功链路，但将 mask 分布升级为 novel-like 多视角 hole。优先训练：

```text
Reference K/V/O adapter
reference_scale / reference_local_scale
可选 BrushNet lightweight scale / adapter
```

避免：

```text
全量微调 ReferenceNet backbone
全量微调 BrushNet backbone
使用 EG3D RGB 当高频 target
```

### 分支 6：验证与指标

必须同时保留自动指标和肉眼检查：

- NOVEL hole pinhole count；
- target-relative dark defect；
- hole gradient；
- visible L1 / dirty texture；
- inverse-warp cycle L1/LPIPS/ID；
- real_ref 质量不退化；
- W+ off / production / new contract 三方对比；
- 原尺寸 hole focus 图。

成功标准不是 EG3D hole L1 最低，而是：

```text
单像素黑孔显著减少；
不出现 shared_trained 式全局脏化；
不出现 wplus_off 式雾蒙和身份弱化；
NOVEL 角度与轮廓保留；
REAL-REF 不退化；
cycle 回源视角真实一致性提升。
```

---

## 7. 创新性判断

该方案不丢失创新性。它不是简单复刻原版 WarpGAN，而是将原版 cycle 第一性原则升级到扩散先验框架：

| 维度 | warpgan_orig | 新方案 |
|---|---|---|
| 生成器 | FFC CNN | SD + BrushNet |
| 外观来源 | CNN + source warp + EG3D full hybrid | ReferenceNet source features + SD/BrushNet prior |
| EG3D 用法 | full RGB hybrid，可能泄漏材质 | depth/silhouette/low-frequency geometry only |
| W+ 用法 | CNN style modulation | 降级 identity/low-frequency condition |
| 监督 | inverse warp cycle + pixel/GAN/ID | real-domain multi-view cycle + reference-guided inpainting |
| 目标 | 补 hole | 解耦 geometry、identity、appearance 的真实域 novel-view inpainting prior |

可写成方法贡献：

1. **Geometry-only EG3D teacher**：将 EG3D 从 pseudo appearance target 降级为低频几何 scaffold；
2. **Reference-guided real-domain cycle training**：无配对 novel GT 时，用真实 source view cycle 监督训练高频补全；
3. **W+ responsibility downgrading**：W+ 只提供身份和低频结构，避免作为高容量空间外观 decoder；
4. **BrushNet/SD prior integration**：用强 inpainting prior 替代原版 FFC 的弱生成能力；
5. **多视角自监督样本扩增**：一张真实照片可采样多个 target camera 构造训练输入。

---

## 8. 近期最小执行计划

1. **文档冻结本方案**：停止 mask threshold、shared contract、Q/K/V/O 无目标搜索。
2. **工程审计**：统一 `build_geometry_condition` 与 `sample_brushnet_inpainting` 的 geometry 参数。
3. **实现 Stage A**：novel-like mask 的 real self-inpainting appearance pretrain。
4. **实现 Stage B prototype**：低噪声 x0 decode + inverse warp cycle loss，不更新 W+ 高频 RCA。
5. **50-step validation**：固定 3 个 identity、同 seed、生成 overview 与 hole focus。
6. **短程 250-step 单变量实验**：只检验新 contract 是否减少碎孔且不引入 wplus_off/shared_trained 式退化。

---

## 9. 需要避免的表述和实现

避免说：

```text
模型恢复了不可见区域真实逐像素高频。
```

应说：

```text
模型在 target-view geometry、source identity 和 reference appearance 约束下生成 photo-realistic、identity-consistent 的合理高频补全。
```

避免实现：

```text
EG3D RGB high-frequency target
NOVEL pseudo RGB/x0 loss 更新 W+ RCA
全量微调 SD/ReferenceNet/BrushNet backbone
只用自动 dark ratio 忽略肉眼 dirty texture
```

---

## 10. 总结

当前正确方向不是寻找一个更强的“EG3D 替代 RGB target”，而是让系统回到无配对 novel-view 任务的第一性原则：

```text
真实 source view 是唯一可信 appearance GT；
target-view geometry 可以由 3D/EG3D 低频信息提供；
新视角高频应由 ReferenceNet、BrushNet/SD prior 和真实域 cycle 共同推断；
W+ 只保留身份/低频职责，不能再成为 pseudo appearance 的空间写入器。
```

该方案与用户提出的设想一致，且保留并强化了项目创新性。

---

## 11. 当前实现的启动命令

所有命令从项目根目录运行，并使用项目 Conda 环境：

```bash
cd /data/xzy/warpgan20260803/20260803
conda activate warpgan
```

### 11.1 Stage A：真实照片 Reference 外观预训练

Stage A 不加载任何旧 WarpGAN adapter checkpoint。SD 1.5、VAE、ReferenceNet backbone
和 BrushNet 使用冻结预训练权重；WarpGAN Reference/W+ adapter 按零门控初始化，其中只训练
Reference appearance adapter。`WARP_GAN_MAX_STEPS` 表示实际优化更新次数。

```bash
WARP_GAN_MAX_STEPS=1000 \
CUDA_VISIBLE_DEVICES=0 python scripts/run_real_reference_pretrain.py
```

默认输出：

```text
experiments/_real_reference_pretrain/
```

### 11.2 Stage B：低频几何 + W+ 身份条件 + 真实域 cycle 微调

Stage B 从 Stage A checkpoint 只迁移模型权重，不恢复 optimizer、global step 或训练状态，
因此是一个从 step 0 开始的新目标训练，而不是 optimizer resume。每个真实样本执行两个梯度隔离更新：

1. `appearance_cycle`：完整 x0 decode 后回投 source，L1/LPIPS/ID 只更新 Reference adapter；
2. `identity_geometry`：EG3D 低频 target + W+ preservation/identity 只更新 W+ mapper、Q/K/V 和 gate；
3. 独立 `to_out_wplus` 始终冻结，避免重新引入已定位的 pinhole 高频出口。

默认从 Stage A 的 1000-step checkpoint 初始化：

```bash
WARP_GAN_STAGE_A_CHECKPOINT=experiments/_real_reference_pretrain/checkpoints/iteration_1000.pt \
WARP_GAN_MAX_STEPS=1000 \
CUDA_VISIBLE_DEVICES=0 python scripts/run_real_cycle_geometry_finetune.py
```

Stage B 不再默认或推荐加载旧 production checkpoint。若 Stage A 使用了不同步数，必须显式指定
实际 Stage A checkpoint 路径。

两个训练脚本默认拒绝覆盖已有实验目录。确实要清空同名目录重新训练时显式设置：

```bash
WARP_GAN_OVERWRITE=1 ...
```

默认输出：

```text
experiments/_real_cycle_geometry_finetune/
```

### 11.3 确定性 50-step 测试

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_real_cycle_geometry.py \
  --checkpoint experiments/_real_cycle_geometry_finetune/checkpoints/iteration_1000.pt \
  --steps 50 \
  --output experiments/_real_cycle_geometry_evaluation
```

重点查看：

```text
experiments/_real_cycle_geometry_evaluation/logs/images/val/overview_step_000000.png
experiments/_real_cycle_geometry_evaluation/logs/val_metrics.txt
```

评估同时输出：NOVEL/REAL-REF 指标、50-step 最终 NOVEL 输出回源后的 cycle
L1/LPIPS/ID/validity、target-relative dark defect，以及包含回源图的 overview。

### 11.4 已完成 smoke test

真实模型、真实数据和 production checkpoint 的单 batch 运行验证结果：

```text
Stage A:
  REFERENCE_SMOKE_OK
  loss = 0.022883
  Reference gradient tensors = 96

Stage B real model route smoke:
  REAL_STAGE_B_ROUTE_SMOKE_OK
  appearance_cycle only changes Reference parameters
  identity_geometry only changes W+ mapper/Q/K/V/gates
  frozen W+ output tensors = 32
```