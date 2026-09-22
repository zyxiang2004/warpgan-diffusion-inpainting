# NOVEL hole 碎裂纹理：根因诊断与消融实验全记录

> 起始问题：NOVEL SD output 的 hole 区域有"细碎小孔与碎裂纹理"，而 REAL SD output（含 hole）效果良好。
> 本文档汇总所有已完成消融的实测数据与代码依据，避免对话中断导致遗忘。
> 所有数值来自 `experiments/_ablation_*/logs/val_metrics.txt` 实测产物，非记忆。

---

## 0. 基线（best_model.pt，主训练 step≈31k）

novel_hole L1 长期停在 **0.103~0.112**，real_ref_hole 稳定在 **0.030~0.033**。
novel_hole 是 real_ref_hole 的约 3 倍，对应肉眼"只坏 NOVEL 的 hole"。

---

## 1. 代码确认的根因（逐条带行号，可复核）

### 成因①（主因）：NOVEL hole 仍被 EG3D 平滑 latent 监督

`training/coach_inpainting_static.py:604-616`（Plan D）：

```python
composite_rgb = batch['warp_img'] * (1.0 - batch['mask']) + batch['target'] * batch['mask']
#                                      ↑ visible=真实warp           ↑ hole=EG3D novel RGB
target_latents = self.vae.encode(composite_rgb * 2.0 - 1.0).latent_dist.sample() * 0.18215
```

该 `target_latents` 随后进入 hole 区的 noise loss（L653）和 x0 loss（L693-694, L731 → L733 总 loss）。
注释声称"不计算 EG3D 像素 loss"，但 hole 区的 latent 监督目标实际就是 EG3D RGB 的 VAE latent。
EG3D 是双刃剑：几何必需（消融A证移除则崩），但平滑材质被注入 hole 监督。

### 成因②：NOVEL 完全没有真实感像素监督

`configs/train_inpainting.yaml:98` + `coach:705-708`：

```yaml
pixel_sup:
  real_pair_only: True
```
```python
pixel_domain_ok = (not real_pair_only) or batch['batch_kind'] == 'real_ref'
# 只有 real_ref 进入 loss_pixel；novel 的 loss_pixel 恒为 0
```

REAL-REF 有 LPIPS+PatchGAN；NOVEL 只有 latent MSE（latent L1↓ ≠ 50步采样感知质量↑）。

### 成因③：NOVEL hole 内 local Reference 被 validity=0 硬屏蔽

`models/referencenet/attention_processor.py:186-188`：

```python
hidden_states = hidden_states + (local_output * validity_seq * torch.tanh(self.reference_local_scale))
#                                              ↑ novel hole 内 = 0，整项归零
```

REAL-REF 走 `validity=ones`（`coach:362-363`），NOVEL 走 3D warp 的 validity（`coach:365-367`），hole 内=0。

### 成因④（已被消融排除）：factorized 梯度隔离使 NOVEL 不能训练 Reference

`coach:984-1007` 参数分组 + `coach:1016-1017` 清梯度 + `coach:1068` novel step `forbidden=appearance_parameters`。
原假设：novel 不能更新 Reference 导致 Reference 在 novel hole 无效。
**J1 消融已排除**（见下）。

### 成因⑤（表层，已证治表不治本）：mask 的 erode/blur 配置从未接入

`configs/train_inpainting.yaml:69-70` 写了 `erode_kernel:3 / gaussian_blur_kernel:21`，
但 `coach:227` 用 `Warper()` 默认 `eks=-1,gks=-1`，配置从未传入。hole mask 是原始 splatting 碎片。
**用户已明确反对用腐蚀修 mask**（腐蚀后效果图完全不像人）。

---

## 2. 已完成消融全记录（实测数据）

所有消融均从 `best_model.pt` 初始化，独立实验目录，跑完后代码回退。

### 消融 A：移除 novel hole 监督（hole noise/x0 权重→0）

| step | novel_hole | novel_visible | real_ref_hole |
|---|---|---|---|
| 0 | 0.1083 | 0.0723 | 0.0301 |
| 500 | **0.2706** | 0.0717 | 0.0314 |
| 1000 | **0.2685** | 0.0701 | 0.0309 |
| 1500 | **0.2707** | 0.0741 | 0.0316 |

**结论：崩盘**。novel_hole 暴涨 2.5×（0.108→0.27），hole 不再被正确填充。但 visible 略改善、novel_edge 降（W+ 摆脱矛盾梯度）。
→ **EG3D hole 监督不能移除**，它是 hole 几何的唯一锚点。

### 消融 B1：cycle = 反投影 SD 输出 + 直接 LPIPS（硬截断 t<200）

| step | novel_hole | novel_edge |
|---|---|---|
| 0 | 0.1083 | 0.0266 |
| 500 | 0.1086 | 0.0275 |
| 1000 | 0.1054 | 0.0291 |
| 1500 | 0.1173 | 0.0288 |
| 2000 | 0.1186 | 0.0279 |


### 消融 B2：cycle = 反投影 SD 输出 + SNR 加权 LPIPS

| step | novel_hole | novel_edge |
|---|---|---|
| 0 | 0.1083 | 0.0266 |
| 500 | 0.1089 | 0.0279 |
| 1000 | 0.1058 | 0.0294 |
| 1500 | 0.1177 | 0.0293 |

**结论：与 B1 几乎相同**。SNR 加权修复了"64% step 不生效"的问题，但 loss 仍不收敛，novel_hole 仍在 0.105↔0.118 震荡。
→ 排除"加权方式问题"，确认 cycle 机制本身在 SD 下不收敛。

### 消融 G：cycle = 反投影 warp_img（几何精确）+ 二次 SD 补洞 + LPIPS

| step | novel_hole | novel_edge |
|---|---|---|
| 0 | 0.1083 | 0.0266 |
| 500 | 0.1190 | 0.0262 |
| 1000 | 0.1081 | 0.0277 |
| 1500 | 0.1074 | 0.0263 |

**结论：loss 仍不收敛（0.003-0.013 震荡）**。即使反投影几何精确的 warp_img，cycle 监督在 SD 下依然无效。

### 消融 G2：同 G 机制，跑长（用户肉眼判断）

| step | novel_hole | novel_edge |
|---|---|---|
| 0 | 0.1083 | 0.0266 |
| 500 | **0.1222** | **0.0183** |
| 1000 | 0.1137 | 0.0182 |
| 1500 | 0.1165 | 0.0189 |
| 2000 | 0.1095 | 0.0176 |

**结论：500步即重新引入油画感**。novel_edge 从 0.0266 暴跌到 0.018（-31%），全图高频被抹平。
用户肉眼确认："碎裂感消失，但变成统一的油画感"——hole 碎裂被平滑掉，代价是 visible 真实纹理也一起被抹平。
→ **任何基于 inverse warp + LPIPS 的 cycle 都会引入油画感**（inverse warp 重采样退化 + LPIPS 强制匹配 → 全局平滑化）。
→ 用户据此发现："hole 碎裂是高频不稳定，不是缺信息；visible 真实纹理宝贵不能被牺牲"。

### 消融 C：mask 腐蚀/模糊（纯推理，零训练）

| 指标 | 当前(无腐蚀) | erode3+blur21 |
|---|---|---|
| mask 碎点(<64px) | 659 | 77 |
| SD hole 高频 Laplacian | 0.0637 | 0.0258 |
| novel_hole L1 | 0.108 | **0.138（恶化）** |

**结论：mask 碎裂是"小孔形态"的部分来源，但腐蚀扩大 hole 导致 L1 恶化 + 肉眼不像人**。
→ 治表不治本。**用户明确反对用腐蚀修 mask。**

### 消融 J1：novel 分支解禁 Reference gate 更新（仍禁 K/V/O 投影）

| step | novel_hole | ref_global_abs | ref_local_abs |
|---|---|---|---|
| 0 | 0.1083 | 0.0319 | 0.0586 |
| 500 | 0.1087 | 0.0319 | 0.0588 |
| 1000 | 0.1054 | 0.0319 | 0.0589 |
| 1500 | **0.1171** | **0.0319** | **0.0590** |

**结论：gate 1500步几乎完全没动**（global 0.0319→0.0319，local 0.0586→0.0590）。
即使解禁，novel loss 对 gate 的梯度极弱——gate 不是"被禁止"，而是"novel 的 EG3D latent 监督根本不驱动 Reference"。
→ **排除成因④**（梯度隔离不是问题，解禁也无济于事）。

### 消融 G+ID：G 机制 + ID loss（填补原版 LPIPS+ID 缺口）

在 G 机制（反投影 warp_img + 二次 SD 去噪）上，cycle loss 从纯 LPIPS 升级为 **LPIPS + ID**。
ID loss 用 ArcFace 全局身份向量（对几何错位鲁棒），作用于 decode 后的整张人脸。

| step | novel_hole | novel_edge | cycle_lpips | cycle_id |
|---|---|---|---|---|
| 0 | 0.1083 | 0.0266 | 0.0956 | 0.3490 |
| 500 | 0.1319 | **0.0275** | 0.0352 | 0.1420 |
| 1000 | 0.1291 | **0.0265** | 0.0322 | 0.1125 |
| 1500 | 0.1238 | **0.0262** | 0.0584 | 0.1281 |
| 2000 | 0.1213 | **0.0265** | 0.0289 | 0.0401 |

**关键发现：ID loss 成功抑制了 G2 的油画感塌陷。**
- novel_edge 始终保持 0.0262~0.0275（G2 同期暴跌到 0.0176，油画感）。
- cycle_id 从 0.349 显著下降到 0.04~0.13（ID 在收敛，不是无效）。
- novel_hole 从 step500 的 0.1319 回落到 0.1213（有改善趋势但未低于基线 0.1083）。

**结论：cycle + ID 不会引入油画感（ID 阻止无差别模糊），但 hole L1 仍高于基线。**
→ cycle 方向**没有被完全证伪**——G+ID 证明了"加 ID 可防油画"，但 hole 改善有限，需更多步数或调权重。
→ 效果图在 `experiments/_ablation_g_id_cycle/logs/images/val/`，**需肉眼确认 hole 纹理是否比基线自然**。

---

## 3. 综合诊断（代码+消融双重佐证）

NOVEL hole 碎裂 = 三个机制叠加：

```
① hole 被迫贴 EG3D 平滑 latent（几何必需，但材质域错误）
② 无 LPIPS/GAN 真实感约束（只有 latent MSE）
③ local Reference 在 hole 被 validity=0 屏蔽（无 per-position 真实纹理）
```

REAL-REF 不碎裂 = 真实 target + local validity=1 + LPIPS/GAN + Reference 可更新。

### 消融 Plan B：hole 不设像素 target，改用 ID+LPIPS 约束

novel hole 的 noise/x0 权重→0（移除 EG3D latent target），改用可训练的 ID+LPIPS loss
（decode x0 → 与真实 source 照片算 ArcFace ID + LPIPS，仅 t<200）。
这是和消融 A 的关键区别：A 移除且不补偿→崩盘；B 移除+加身份/感知约束。

| step | novel_hole | novel_visible | novel_edge |
|---|---|---|---|
| 0 | 0.1083 | 0.0723 | 0.0266 |
| 500 | 0.1871 | 0.1061 | 0.0276 |
| 1000 | 0.2494 | 0.0952 | 0.0267 |
| 1500 | **0.2762** | 0.1037 | 0.0286 |
| 2000 | **0.2692** | 0.0998 | 0.0265 |

**结论：崩盘（和消融 A 一样）。** novel_hole 从 0.108 恶化到 0.27（A 是 0.27）。
ID/LPIPS 比 A 慢一些（500步 0.187 vs A 的 0.271），但最终同样崩溃。
→ **仅靠 ID+LPIPS 无法提供 hole 的几何/形状信息**——SD 在 hole 区失去形状锚点后，
生成内容逐渐偏离正确位置。ID 约束身份、LPIPS 约束感知，但都不告诉 SD "hole 该长什么形状"。
→ **hole 必须有某种形式的形状/几何 target**（EG3D 或其他），纯身份/感知约束不够。

---

### 消融 Plan C：hole target = EG3D 低频（形状）+ source 高频（真实纹理）

用拉普拉斯分解把 hole target 拆成：EG3D 的低频（形状/光照/轮廓，正确几何）+
source 照片的高频（皮肤毛孔/发丝，真实纹理），混合后作为 hole 的 latent target。
high_freq_ratio=0.5（平衡）。这是"保留几何 + 换纹理"的第一性原理方案。

| step | novel_hole | novel_visible | novel_edge |
|---|---|---|---|
| 0 | 0.1083 | 0.0723 | 0.0266 |
| 500 | 0.1192 | 0.0729 | 0.0299 |
| 1000 | **0.1112** | 0.0746 | 0.0287 |
| 1500 | 0.1163 | 0.0757 | 0.0306 |
| 2000 | **0.1132** | 0.0737 | 0.0283 |

**结论：第一个"不崩盘 + 不油画 + 稳定"的方案！**
- novel_hole 始终在 0.111~0.119（基线 0.1083），**从未崩盘**（对比 A/B 的 0.27）
- novel_edge 保持/略升（0.028~0.031），**无油画感**（对比 G2 的 0.018）
- hole L1 没有显著下降，但也**没有恶化**——拉普拉斯混合的 target 是稳定的
→ "保留几何 + 换纹理"方向成立：EG3D 低频提供了足够几何锚点（不崩），
source 高频提供了真实纹理域（不油画）。
→ **需肉眼确认：hole 区纹理质感是否比基线（EG3D 塑料）更真实。**

---

## 3. 综合诊断（代码+消融双重佐证）

**已排除的假说**：
- gate 被梯度隔离困住（J1 证伪）
- 移除 EG3D 监督（A 证伪，崩盘）
- 移除 EG3D + ID/LPIPS 补偿（Plan B 证伪，同样崩盘——ID/LPIPS 无法替代几何 target）
- 纯 LPIPS 的 cycle（B1/B2/G/G2 证伪，不收敛或引入油画感）
- mask 腐蚀（C 证伪，治表伤本，用户反对）

**部分验证、未完全定论**：
- cycle + ID（G+ID）：不引入油画感，但 hole L1 改善有限。
- **Plan C（拉普拉斯混合 target）**：不崩盘、不油画、稳定。方向成立但需肉眼确认纹理改善。

**关键教训**：hole 必须有几何 target（A/B 证明）。问题不是"要不要 target"，
而是"target 的纹理域要对"。Plan C 的"EG3D 低频几何 + source 高频纹理"是
目前唯一同时满足"有几何 + 真实纹理域"的方案。

**Plan C 肉眼反馈（用户）**：hole 区纹理没变塑料（好），但"mask 碎裂部分附近的
细小黑色孔洞没有被正确修复，还是碎裂感"。背景不重要处有扭曲。
用户希望"将碎裂的 mask 碎片联系起来，不产生碎裂感"。

**Plan C 局限性的根因**：Plan C 只改了 target（监督信号），没改 BrushNet 的
输入条件（碎裂 mask）。碎裂孔洞来自 mask 本身的碎片结构（Splatting.py:105 的
原始 splatting 掩码有 ~659 个小碎片），BrushNet 收到碎裂 mask → SD 沿碎片边缘
产生不连续黑色孔洞。无论 target 纹理多好，mask 碎片结构不变，碎裂感就不会消失。

**未尝试的方向（用户明确需求）**：mask 碎片连接（形态学闭运算 close=dilate+erode）。
- 与之前被否决的"腐蚀(erode)"不同：腐蚀扩大 hole（吃掉 visible）；
- 闭运算填合碎片间缝隙，连接邻近碎片成连续区域，最终边界几乎不变。
- 这正是用户说的"将碎裂的 mask 碎片联系起来"。
- 注意：只应用于 RGB/mask warper，不能用于多尺度 reference feature warper。

---

## 4. 尚未验证的方向（下一步候选）

### 方向 D1：hole-specific global Reference gate
global Reference 本就覆盖 hole（不依赖 validity），但只有一个统一 gate。
新增 hole-specific gate：`global_output × visible_gate + global_output × hole_gate`，只让 hole 能单独加强 global 注入。
- 优点：不依赖 local 几何对应，不动 visible，直接测"hole 是否需要更强 global Reference"
- 风险：需新增可训练参数，可能 OOM

### 方向 D2：hole 边界 local validity 软化
不把整个 hole validity 设 1（会注入零值 feature），只对 hole 边界向内扩张少量 latent 像素，传播最近的有效 aligned feature。
- 优点：稳定 visible/hole 边界纹理，不碰 hole 中心
- 风险：边界扩散范围需调，可能引入边界不一致

### 方向 H：W+ latent consistency（学习原版身份闭环）
原版有 `F.mse_loss(codes, gan.encoder_forward(pred))`（身份闭环），我们没有。
让 SD novel x0 → EG3D encoder → 必须还原 W+ codes。
- 优点：特征级闭环，不涉及 inverse warp，不会油画
- 风险：约束身份/结构，未必直接解决纹理碎裂；需 EG3D encoder 前向（显存+计算）

### 方向 E：novel 分支的 ID loss（从 monitor 升级为训练 loss）
当前 novel 的 ID loss 只是 detached monitor（`coach:724-726`）。
把它升级为可训练 loss（仅 novel，t<200）。ID loss 对几何错位鲁棒（全局身份向量）。
- 优点：最小改动，复用现有 id_loss_fn，不涉及 inverse warp
- 风险：ID loss 只约束身份，不约束纹理细节

---

## 5. 代码状态

- 所有实验代码已回退，仅保留项目原有 Plan D（`coach` 语法通过）
- 所有消融实验目录 + 效果图保留在 `experiments/_ablation_*/`，可供复查
- `best_model.pt`（step≈31k）未被任何实验污染

---

## 6. 2026-08-12 补充复核：mask 是触发器，NOVEL 条件缺口更接近主因

> 本节是在重新逐行对比当前工程、`warpgan_orig`、正式训练配置和固定验证产物后追加。
> 它修正第 3 节中“碎裂 mask 本身就是主因”的过强表述；此前消融数据仍然有效。

### 6.1 NOVEL 与 REAL-REF 实际使用同一张碎裂 mask

`training/coach_inpainting_static.py:324-332` 中，两分支都先执行同一个 source→novel
forward warp，并统一设置 `mask = 1 - visable_mask`。REAL-REF 只把 RGB condition 改为
`source * visable_mask`，没有更换 mask。因此：

- NOVEL 和 REAL-REF 的 mask 拓扑完全相同；
- REAL-REF 在同一碎裂 mask 下仍能良好修复；
- 碎裂 mask 不是“只坏 NOVEL”的充分原因，只能是空间高频触发器/放大器。

从固定验证总览反推的三个 mask 分别约有 1017、265、953 个连通分量，其中面积
不超过 16 像素的小分量约占 89%~93%，说明碎裂现象客观存在。但 Plan C step=2000
样本 0 的 NOVEL hole 内部梯度约 0.035，REAL-REF 仅约 0.011；相同 mask 下输出差异
仍接近 3 倍，必须由两分支独有条件解释。

### 6.2 与原版的关键缺口：diffusion 路径没有使用 hybrid hole geometry condition

原版 `warpgan_orig` 在 `warp.hybrid=True` 时执行：

```python
masked_img = warp_img * (1 - mask) + y_hat_novel * mask
inp = torch.cat([masked_img, y_hat_novel, mask], dim=1)
```

即原版生成器在 hole 内始终看到目标相机下的 EG3D novel render，获得逐像素五官、
轮廓和头发位置。当前 diffusion 配置虽然仍写 `warp.hybrid: True`，但训练中的 BrushNet
condition 实际为：

```python
masked_image_latents = VAE(raw_warp_img)
conditioning_latents = cat([masked_image_latents, mask_latents])
```

`warp.hybrid` 在 diffusion 路径没有被读取。正式推理
`scripts/infer.py:381-393` 还明确对 diffusion backend 跳过 `outs_novel`，因此推理 hole
只有 raw black warp、碎裂 mask、W+ 和 Reference，没有目标视角逐像素 geometry condition。

这与既有消融形成闭环：移除 EG3D hole target 后 novel_hole 从约 0.108 崩到约 0.27，
而 ID/LPIPS 无法替代。既有实验说明 hole 必须有目标视角形状锚点；新代码对比说明该
锚点目前只存在于训练 target/loss 端，不存在于推理 condition 端。

### 6.3 第二个分支差异：NOVEL hole 内 local Reference 被 validity=0 清零

REAL-REF 使用同视角 aligned feature，并令 `validity=ones`，所以 hole 内仍有逐位置真实
Reference 特征。NOVEL 使用 3D warp validity，在 hole 内为 0：

```python
local_output * validity_seq * tanh(reference_local_scale)
```

因此 NOVEL hole 的 local Reference 项恒为零。J1 只排除了“NOVEL 不允许更新 Reference
参数”这一假说，不能排除前向传播中 `validity=0` 对 local feature 的硬屏蔽。

### 6.4 修正后的根因优先级

1. **首要候选：缺少 target-view hole geometry condition。** 原版 hybrid 输入被 diffusion
   迁移遗漏；W+ 是全局 token，不能可靠替代逐像素空间草图。
2. **重要放大器：原始 splatting mask 极度碎片化。** BrushNet 在 latent 尺度收到椒盐式
   hole pattern，但相同 mask 在 REAL-REF 可修好，故不是唯一主因。
3. **重要分支差异：NOVEL hole local Reference 为零。** REAL-REF 则在 hole 内拥有强同位置
   Reference 信息。
4. **监督不对称：** REAL-REF 有真实 target、LPIPS、PatchGAN；NOVEL 只有 EG3D/composite
   latent 监督和 W+ 正则。

### 6.5 下一步实验：冻结权重的 Geometry-Condition 因果测试

不训练、同一 `best_model.pt`、固定验证 batch、相同 seed=42，比较：

- A `raw_warp`：当前条件；
- B `eg3d_full_hybrid`：hole 填完整 EG3D novel RGB；
- C `eg3d_low_hybrid`：hole 只填 EG3D 低频，避免塑料高频直接进入 condition；
- D `eg3d_low_hybrid_close3`：C + 64×64 latent mask 的 3×3 closing。

该实验只改变 frozen BrushNet 的输入条件，不修改 checkpoint。若 B/C 显著减少小黑孔，
即可支持“condition 端缺少空间几何”的因果判断；若只有 D 改善，则 mask 拓扑仍是主要
瓶颈；若均无改善，则进入 local Reference 边界传播实验。

---

## 7. Geometry-Condition 因果实验结果（冻结 best_model.pt，不训练）

实验入口：`scripts/eval_geometry_condition_causality.py`  
实验目录：`experiments/_geometry_condition_causality/`  
总览图：`experiments/_geometry_condition_causality/geometry_condition_overview.png`  
原始指标：`experiments/_geometry_condition_causality/metrics.json`  
完整日志：`experiments/_geometry_condition_causality/run.log`

### 7.1 实验有效性核验

- checkpoint：正式 `best_model.pt`，`architecture_version=5`，step=31000；
- 固定 validation batch，3 个 identity；
- 每个 variant 都使用 DPM-Solver++、50 steps、seed=42；
- 权重、W+、Reference features、scheduler 和初始 noise 完全相同；
- 唯一变量是 BrushNet RGB/mask condition；
- `raw_warp` 新输出与历史 Plan C step 0 NOVEL 输出逐样本像素 MAE 约
  0.00194~0.00197，最大差异不超过 3/255，成功复现视觉基线；
- 进程 exit code=0，12 次采样全部完成。

### 7.2 三个 identity 的均值

| variant | full L1 | hole L1 | visible L1 | hole edge error | hole dark ratio (<0.08) |
|---|---:|---:|---:|---:|---:|
| raw_warp | 0.081828 | 0.098501 | 0.072903 | 0.020728 | 0.088539 |
| eg3d_full_hybrid | 0.073863 | 0.080641 | 0.070485 | 0.012506 | 0.033421 |
| **eg3d_low_hybrid** | **0.073767** | **0.076870** | **0.070987** | **0.011706** | **0.035301** |
| eg3d_low_hybrid_close3 | 0.075050 | 0.077992 | 0.072276 | 0.011431 | 0.022371 |

相对 `raw_warp`：

| variant | full L1 | hole L1 | hole edge error | dark ratio |
|---|---:|---:|---:|---:|
| eg3d_full_hybrid | -9.73% | -18.13% | -39.67% | -62.25% |
| **eg3d_low_hybrid** | **-9.85%** | **-21.96%** | **-43.53%** | **-60.13%** |
| eg3d_low_hybrid_close3 | -8.28% | -20.82% | -44.85% | -74.73% |

### 7.3 分样本现象

- identity 0 是黑孔最明显的样本。`raw_warp` dark ratio=0.241、edge=0.0345；
  low hybrid 后分别降至 0.0985、0.0165。说明 geometry condition 能直接压制黑孔和
  hole 内碎裂梯度。close3 进一步把 dark ratio 降至 0.0602，但 hole L1 反而升高。
- identity 1 原本几乎没有黑孔。low hybrid 仍将 edge 从 0.0102 降至 0.0080，hole L1
  从 0.0671 降至 0.0646，但 full/visible 有轻微变化，需肉眼判断是否值得。
- identity 2 改善最强：hole L1 从 0.1351 降至 0.0658，edge 从 0.0175 降至 0.0107，
  dark ratio 从 0.0237 降至 0.0056。

### 7.4 对根因的支持程度

冻结现有 checkpoint、仅在推理时补回 target-view geometry condition，就能同时改善
full/hole L1、hole edge 和黑孔比例。因此以下判断获得强因果支持：

> 当前 diffusion 迁移遗漏了原版 hybrid 中的目标视角逐像素几何条件，这是 NOVEL hole
> 碎裂的首要原因；碎裂 mask 会放大该缺口，但不是唯一主因。

`eg3d_low_hybrid` 比 full hybrid 的 hole L1/edge 更好，且理论上更少携带 EG3D 高频
塑料材质，暂定为正式修复的首选条件。

### 7.5 closing 的副作用

3×3 closing 在 64×64 latent mask 上会显著扩大大 hole 样本：

- identity 0：面积 0.3706 → 0.4563；
- identity 2：面积 0.3479 → 0.4185。

它虽然进一步降低 dark ratio，但平均 full/visible L1 比不 closing 更差。当前实现过强，
不应直接进入正式训练。若肉眼确认低频 hybrid 仍残留少量碎片，再单独设计“只连接小间隙、
限制面积增长”的 component-aware closing，而不是普通 3×3 closing。

### 7.6 当前停止点：需要肉眼确认

请重点比较总览图每行后四列：

1. `raw_warp` 与 `eg3d_low_hybrid`：细小黑孔、皮肤/头发碎裂是否明显减少；
2. `eg3d_full_hybrid` 与 `eg3d_low_hybrid`：低频版本是否更少塑料感或 EG3D 纹理泄漏；
3. `eg3d_low_hybrid_close3`：是否出现脸形、头发边界、背景区域被吃掉或过度改写；
4. identity 0 和 identity 2 优先，identity 1 用于检查无明显黑孔样本是否被副作用伤害。

在肉眼结论返回前，不启动短程微调。若 low hybrid 肉眼通过，下一步应把同一 condition
同时接入训练、validation 和 `scripts/infer.py`，从 best_model.pt 做短程微调；若 low hybrid
仍不理想，则先做 NOVEL hole 边界的 local Reference validity 传播实验。

---

## 8. Geometry-Condition 肉眼反馈与第一性原理后续方向

### 8.1 用户肉眼反馈

- `eg3d_low_hybrid` 相比 `raw_warp` 没有完全抹除孔洞，但在不损坏整体效果的同时，
  确实削弱了孔洞导致的碎裂感；
- hole 中背景与原背景衔接仍不理想，希望后续能自然融合；
- low hybrid 没有油画感，但 hole 有“抹平”感，人脸边界不够清晰；
- full/low hybrid/close3 肉眼差异不大，close3 没有显示出值得承担其 mask 面积膨胀
  副作用的明显收益；
- 后续修正应遵循第一性原理，优先数学建模，不使用大量工程性 mask 补丁。

### 8.2 现象的频率解释

low hybrid 只提供 EG3D 的低频目标视角形状，因此它能解决“hole 中内容应出现在哪里”，
却不能自行恢复皮肤、发丝和人脸轮廓的高频细节。当前代码又在 NOVEL hole 中令 local
Reference validity=0，恰好截断了真实 source photo 的逐位置高频外观路径。

因此后续应分解为两个正交场：

1. **低频几何场**：EG3D low-pass condition 决定目标视角结构；
2. **高频外观场**：3D aligned Reference feature 决定边界与真实纹理。

不再优先尝试 closing、腐蚀、输出端粘贴或额外 hole gate。

### 8.3 连续 Reference 延拓的数学形式

将 aligned Reference feature 记为 `F`，3D warp validity 记为 `V`，使用 normalized
convolution 在可见域边界向 hole 做带置信度的连续延拓：

```text
F_hat = Gaussian_sigma(V * F) / (Gaussian_sigma(V) + eps)
V_hat = Gaussian_sigma(V)
```

实现时 `eps` 仅作为 `Gaussian_sigma(V)` 的支撑阈值/除法下限，不加到所有分母上，
从而保持 normalized convolution 对常数场的精确再现性。

原有效区域保持 `F,V` 不变，仅在 hole 中使用 `F_hat,V_hat`。`V_hat` 是局部有效样本的
高斯加权密度，会随离开可见边界的距离自然衰减，而不是将整个 hole 粗暴设为有效。

为保持频率职责分离，只延拓 64×64 与 32×32 的高分辨率 Reference feature；16×16
及以下不延拓，避免 Reference 改写由 EG3D low condition 提供的低频姿态和脸形。

下一冻结权重实验比较：

- `low_hybrid`；
- `low_hybrid + normalized convolution sigma64=1`；
- `low_hybrid + normalized convolution sigma64=2`；
- `raw_warp + normalized convolution sigma64=2`，用于分离 geometry 与 appearance 的贡献。

---

## 9. Reference 连续延拓实验结果：数学假设不成立，停止该方向

实验入口：`scripts/eval_reference_extension_causality.py`  
实验目录：`experiments/_reference_extension_causality/`  
总览图：`experiments/_reference_extension_causality/reference_extension_overview.png`  
指标：`experiments/_reference_extension_causality/metrics.json`  
日志：`experiments/_reference_extension_causality/run.log`

### 9.1 算子性质验证

在运行生成实验前，对 normalized convolution 做了独立数学性质测试：

- 原有效域 feature 严格不变；
- 常数 feature 场在有支撑位置精确再现；
- validity/confidence 始终位于 `[0,1]`；
- 置信度从 hole 边界向中心衰减（测试中边界约 0.40，中心为 0）；
- 无 NaN/Inf。

初版实现使用 `numerator/(support+eps)`，性质测试发现它会在低置信度区域系统性压低
feature 幅值，不满足常数再现性；已在实验前修正为仅用 `eps` 作支撑阈值和除法下限。

### 9.2 三个 identity 的均值

| variant | full L1 | hole L1 | edge error | dark ratio | hole gradient | boundary gradient |
|---|---:|---:|---:|---:|---:|---:|
| low_hybrid | 0.073766 | 0.076869 | 0.011706 | 0.035304 | 0.009648 | 0.028170 |
| low_hybrid + ref sigma1 | 0.073894 | 0.076245 | 0.011709 | 0.038103 | 0.009662 | 0.027827 |
| low_hybrid + ref sigma2 | 0.073770 | 0.075205 | 0.011721 | 0.038423 | 0.009659 | 0.027715 |
| raw_warp + ref sigma2 | 0.081545 | 0.097366 | 0.020600 | 0.093261 | 0.018631 | 0.049113 |

相对 low hybrid：

- sigma1：hole L1 -0.81%，但 edge +0.02%，dark ratio +7.93%；
- sigma2：hole L1 -2.16%，但 edge +0.13%，dark ratio +8.83%；
- boundary gradient 下降约 1.2%~1.6%，没有恢复清晰边界，反而更平滑；
- raw warp 单独增加 Reference 延拓仍远差于 low hybrid，说明 geometry condition 才是
  主导因果变量。

### 9.3 分样本方向不一致

- identity 0：延拓后 L1、edge 和 dark ratio 均轻微恶化；
- identity 1：hole L1 有改善，edge 略降，但 dark ratio 上升；
- identity 2：变化很小，无法稳定恢复边界细节。

同一个正规化算子在不同身份上方向不一致，不能通过继续调 sigma 解决。

### 9.4 第一性原理否决原因

normalized convolution 隐含假设：hole 邻域属于同一局部平稳 feature 场，可以由空间邻近
样本插值。但 novel-view disocclusion 恰好位于遮挡边界：邻近像素可能分别属于前景头发、
脸部和背景，不满足同一表面假设。各向同性高斯核会把错误侧的外观扩入新露出区域。

因此该方向虽然数学形式干净，但物理假设错误。不能作为正式修复，也不继续尝试更多 sigma、
closing 或各向同性局部插值。

### 9.5 下一步应采用的原则

已证实：

1. 低频 EG3D geometry condition 是必要且有效的空间锚点；
2. 高频细节不能通过 disocclusion 边界的二维邻域手工延拓；
3. 边界细节应由模型在“低频几何已知”的条件下学习真实图像高频残差。

因此下一步不再增加推理时补丁，而应把 low hybrid condition 统一接入训练与推理，从
`best_model.pt` 做短程一致性微调。其目标不是学习 EG3D 材质，而是学习条件分解：

```text
target image = low-frequency target-view geometry + learned high-frequency residual
```

REAL-REF 的真实照片监督负责约束高频残差的自然图像分布；NOVEL 的 EG3D/composite latent
监督继续约束目标视角结构。训练和推理使用同一 low-hybrid condition，消除当前 condition
端缺少 geometry 的结构性不一致。

---

## 10. Low-Geometry Condition 一致性微调：Stage 1（0→250 step）

### 10.1 正式实现

统一条件定义：

```text
C = (1 - M) * W + M * L31(I_target)
```

- `W`：真实 source 的 forward warp；
- `M`：原始 hole mask，不做 closing/腐蚀/膨胀；
- `L31`：固定 31×31 反射边界 box low-pass；
- NOVEL 的 `I_target`：EG3D novel render；
- REAL-REF 的 `I_target`：真实 target photo。

使用反射边界而不是零填充，避免在 frame 外人为假设黑色背景。低通对常数场的 FP32
最大误差约 `2.68e-6`，condition 在 visible/hole 分区上的复合误差严格为 0。

同一条件已接入：

- `training/coach_inpainting_static.py` 的 diffusion 训练；
- Coach validation；
- `utils/diffusion_inpainting.py` 共享采样；
- `models/diffusion_inpaintor.py` 正式 checkpoint loader；
- `scripts/infer.py` 正式推理。

checkpoint 新增：

```text
brushnet_condition_contract = visible_warp_plus_lowpass_target_geometry
```

旧 checkpoint 缺少该字段时自动解释为 `raw_warp_plus_mask`，不会被新条件意外改变；只有
新 contract checkpoint 才要求正式推理提供 EG3D target-view geometry image。

### 10.2 Step-0 冻结安全检查

独立目录：`experiments/_low_geometry_condition_step0/`

| domain | full | hole | visible | edge full | edge hole |
|---|---:|---:|---:|---:|---:|
| NOVEL | 0.074591 | 0.085458 | 0.070677 | 0.019836 | 0.012574 |
| REAL-REF | 0.039765 | 0.040588 | 0.039468 | 0.014436 | 0.007360 |

NOVEL 保留冻结 low-geometry 的主要收益。REAL-REF 相比原约 0.031 出现合理 condition
shift，但没有崩坏，因此允许进入短程训练。

### 10.3 Stage 1 设置

入口：`scripts/run_low_geometry_condition_finetune.py`  
目录：`experiments/_low_geometry_condition_finetune/`  
初始化：原正式 `best_model.pt`（step=31000）  
训练：factorized joint；identity LR=2e-5，appearance LR=5e-6  
停止：step 250，不自动继续到 2000。

### 10.4 Step 250 实测结果

| step | novel full | novel hole | novel visible | novel edge | real full | real hole | real visible | real edge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.074590 | 0.085454 | 0.070677 | 0.019835 | 0.039762 | 0.040587 | 0.039466 | 0.014437 |
| 250 | **0.062214** | **0.039342** | **0.070451** | **0.021681** | **0.033115** | **0.032817** | **0.033222** | **0.016211** |

变化：

- NOVEL hole 下降约 54%；
- NOVEL full 下降约 16.6%；
- NOVEL visible 基本不变，说明没有以破坏已知区换取 hole 指标；
- REAL-REF full 从 0.0398 恢复到 0.0331，接近原 checkpoint 的约 0.031；
- NOVEL edge 从 0.0198 升至 0.0217，没有继续变得更平，数值上恢复了部分高频变化；
- 训练无 NaN/OOM，step 100/200 梯度有限，进程 exit code=0。

### 10.5 当前风险与停止原因

NOVEL hole target 仍来自 EG3D/composite latent。hole L1 快速降至 0.0393 可能表示结构、
背景衔接和 hole 连续性真正改善，也可能意味着模型开始更强贴近 EG3D 平滑材质域。
仅凭 L1/edge 不能排除后者，因此不自动继续到 step 500/2000。

需要肉眼比较：

- `experiments/_low_geometry_condition_finetune/stage1_output_comparison.png`
- `experiments/_low_geometry_condition_finetune/logs/images/val/overview_step_000000.png`
- `experiments/_low_geometry_condition_finetune/logs/images/val/overview_step_000250.png`

重点检查：

1. NOVEL hole 黑孔/碎裂是否进一步消失；
2. low-hybrid 的“抹平感”是否恢复出清晰人脸/头发边界；
3. 是否出现 EG3D 塑料感、油画感或身份漂移；
4. hole 背景与 visible 背景是否更自然融合；
5. REAL-REF 是否保持真实照片质感。

### 10.6 Stage 1 肉眼反馈

- `NOVEL BrushNet condition` 整体效果很好，只有角度变换区域内残留非常细微碎纹；
- NOVEL step 0 的 low-geometry 输出较平滑，孔洞被压制；
- NOVEL step 250 随人脸边缘重新出现，孔洞/碎纹也开始重新浮现；
- REAL-REF step 250 比 step 0 恢复了更多真实细节。

这说明训练同时恢复了两类高频：真实边缘细节和碎孔伪影。由于 factorized joint 的
Identity 与 Appearance 参数严格分组，下一步不继续训练，而做 2×2 参数因果交换：

| Identity 参数 | Appearance 参数 | 目的 |
|---|---|---|
| step 0 | step 0 | low-geometry 冻结基线 |
| step 250 | step 250 | 当前 Stage 1 结果 |
| step 0 | step 250 | 判断 REAL-REF 学到的外观是否能单独恢复边缘且不带回孔洞 |
| step 250 | step 0 | 判断 NOVEL/W+ 更新是否是孔洞重新浮现的来源 |

Identity 参数包括 W mapper 与 cross-attention W+ Q/K/V/O/gate；Appearance 参数包括
self-attention Reference K/V/O、global gate 和 local gate。四组使用完全相同的 low-geometry
condition、Reference features、采样器与 seed，不再训练。

---

## 11. Stage 1 参数 2×2 因果交换：碎孔重现主要来自 Identity/W+ 更新

实验入口：`scripts/eval_low_geometry_parameter_factorization.py`  
实验目录：`experiments/_low_geometry_parameter_factorization/`  
总览图：`experiments/_low_geometry_parameter_factorization/parameter_factorization_overview.png`

真正的 step-0 参数取原正式 `best_model.pt`；训练目录的 `iteration_0.pt` 已完成一次优化，
不能作为冻结基线。step-250 参数取 `iteration_250.pt`。

### 11.1 NOVEL 结果

| combination | full L1 | hole L1 | visible L1 | hole edge error | dark ratio | hole gradient |
|---|---:|---:|---:|---:|---:|---:|
| I0_A0 | 0.074591 | 0.085458 | 0.070677 | 0.012585 | 0.046331 | 0.010539 |
| I0_A250 | 0.074062 | 0.083054 | 0.070823 | 0.012862 | 0.052732 | 0.010818 |
| I250_A0 | 0.062716 | 0.038497 | 0.071439 | 0.017507 | 0.108164 | 0.015973 |
| I250_A250 | 0.062214 | 0.039342 | 0.070452 | 0.017985 | 0.114219 | 0.016443 |

### 11.2 参数主效应

仅更新 Appearance（I0_A250）：

- REAL-REF full 从 0.039765 降至 0.033116，全部真实细节恢复来自 Appearance；
- NOVEL hole 只从 0.085458 降至 0.083054；
- NOVEL hole gradient 略升，dark ratio 仅从 0.0463 升到 0.0527。

仅更新 Identity（I250_A0）：

- NOVEL hole 从 0.085458 骤降至 0.038497，承担了几乎全部 target L1 改善；
- hole gradient 从 0.010539 升至 0.015973，脸部边缘被恢复；
- dark ratio 同时从 0.0463 升至 0.1082，碎孔也几乎全部由 Identity 更新带回；
- REAL-REF 完全不变，因为该分支禁用 W+，验证了参数交换的因果隔离。

### 11.3 第一性原理解释

当前 NOVEL 训练中：

```text
condition_hole = L(EG3D)
target_hole = EG3D
```

因此 Identity/W+ 分支被迫学习：

```text
H(EG3D) = EG3D - L(EG3D)
```

即 EG3D 的高频残差。该监督只在高度碎片化的 hole mask 内强作用，于是恢复脸部边缘时，
也重新刻画了与 mask 拓扑相关的细碎孔洞。Appearance 更新只贡献小部分该现象。

### 11.4 下一步数学修正

不冻结 Identity，不调 mask，也不继续沿用已污染的 step-250 Identity。重新从原正式
checkpoint 开始，使 NOVEL 的监督频率与 condition 职责一致：

```text
NOVEL condition_hole = L(EG3D)
NOVEL target_hole    = L(EG3D)
```

REAL-REF 仍保持：

```text
REAL condition_hole = L(real photo)
REAL target         = real photo
```

于是 Identity 只负责低频目标视角结构，Appearance 从真实照片学习自然高频残差。与消融 A
不同，NOVEL hole 仍有完整低频几何监督，不会失去形状锚点；删除的只是没有真实依据、且已被
因果实验确认会带回碎孔的 EG3D 高频监督。

---

## 12. 对话恢复审计与 Stage 2 实验状态（2026-08-12）

### 12.1 恢复时项目实际停点

本轮从中断对话恢复后，重新核对了代码、实验产物和进程，而不是依赖上轮文字记忆：

- `_low_geometry_condition_finetune` 已正常完成到 step 250，exit code=0；
- `_low_geometry_parameter_factorization` 已完成全部 2×2 参数交换与指标导出；
- 当前无 `train_inpainting`、`accelerate` 或 `torchrun` 进程；
- 恢复检查当时，`scripts/run_frequency_consistent_geometry_finetune.py` 与对应 coach 目标分支
  已经写好，但实验目录尚不存在；这是下文启动 Stage 2 前的历史状态；
- Stage 1 与参数交换的脚本、coach、正式推理路径均通过 `py_compile`。

因此暂停点不是“尚未定位原因”，而是已经完成如下因果链：

```text
low-geometry condition 在 step 0 压制碎孔
  -> step 250 恢复边缘时碎孔重现
  -> 2×2 参数交换定位到 Identity/W+ 更新
  -> 该分支唯一需要拟合的额外信号是 EG3D - L(EG3D)
  -> 下一步应删除 NOVEL 的 EG3D 高频监督，而非冻结 Identity 或修改 mask
```

### 12.2 Stage 2 唯一主变量

入口：`scripts/run_frequency_consistent_geometry_finetune.py`  
目录：`experiments/_frequency_consistent_geometry_finetune/`  
初始化：未触碰的原正式 `best_model.pt`，而不是 Stage 1 step-250 checkpoint  
计划停止：step 250，沿用相同 seed、LR、factorized 参数分组和验证 batch。

训练目标由 Stage 1 的：

```text
NOVEL condition_rgb = warp_visible + L31(EG3D)_hole
NOVEL target_rgb    = warp_visible + EG3D_hole
```

改为：

```text
NOVEL condition_rgb = warp_visible + L31(EG3D)_hole
NOVEL target_rgb    = condition_rgb
```

REAL-REF 不变，仍以完整真实照片为 target。这样保留 NOVEL hole 的低频形状、颜色和目标视角
空间锚点，只删除参数交换已确认会带回碎孔的 EG3D 高频残差。该实验与消融 A（完全移除
NOVEL hole 监督）不同，后者丢失了全部 hole 几何，因此不能用 A 的崩盘否定本实验。

### 12.3 实现审计与已知限制

`training/coach_inpainting_static.py` 中 `novel_hole_target=low_geometry` 会强制要求
`geometry_condition.enable=True`，并直接令 `composite_rgb = condition_rgb`；condition 与 target
在图像域使用完全相同的 mask 和 `L31` 低通算子。

当前 VAE 编码仍遵循既有训练方式：target latent 和 BrushNet condition latent 各自调用一次
`latent_dist.sample()`。因此二者在 latent 域不逐元素相等，但差异是 VAE posterior 随机采样噪声，
不包含 EG3D 高频纹理或碎裂 mask 拓扑。本阶段不同时改成 posterior mode，以保证唯一主变量仍是
“是否监督 EG3D 高频”。若 Stage 2 仍系统性重现碎孔，再把 deterministic VAE encode 作为下一项
独立、可证伪消融。

### 12.4 Stage 2 判定标准

到 step 250 后必须同时比较 step 0 / Stage 1 step 250 / Stage 2 step 250：

1. NOVEL：脸部与头发边缘是否恢复，但 dark pinhole/碎裂纹是否不再随 Identity 更新回升；
2. NOVEL：姿态、轮廓和背景连续性是否仍由低频 geometry 锚定，不能出现消融 A 式塌陷；
3. REAL-REF：是否保留 Stage 1 中由 Appearance 带来的真实细节恢复；
4. 数值：NOVEL L1 是对 EG3D full target 的诊断量，Stage 2 主动删除其高频监督后不要求优于
   Stage 1 的 0.0393；视觉碎孔、hole gradient/dark ratio 与结构稳定性优先；
5. 若边缘仍平而碎孔消失，说明目标频率一致但 Identity 缺少真实高频来源，下一阶段才考虑
   让 Appearance 的真实高频跨域迁移；若边缘和碎孔一起回来，则继续检查 VAE 随机编码或
   condition/mask 在 latent 尺度的耦合。

### 12.5 Stage 2 已完成结果

Stage 2 已运行到预定的 step 250，进程 exit code=0；无 NaN/OOM，step 0/100/200 的
Identity 与 Appearance 梯度均有限。产物：

- checkpoint：`experiments/_frequency_consistent_geometry_finetune/checkpoints/iteration_250.pt`；
- 训练总览：`experiments/_frequency_consistent_geometry_finetune/stage0_stage1_stage2_comparison.png`；
- step 250 原图：`experiments/_frequency_consistent_geometry_finetune/logs/images/val/overview_step_000250.png`。

训练验证指标：

| stage | novel full | novel hole | novel visible | novel edge | real full | real hole | real visible | real edge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage 0 frozen | 0.074590 | 0.085453 | 0.070677 | 0.019835 | 0.039763 | 0.040588 | 0.039466 | 0.014437 |
| Stage 1 full EG3D | 0.062214 | 0.039342 | 0.070451 | 0.021681 | 0.033115 | 0.032817 | 0.033222 | 0.016211 |
| Stage 2 frequency-consistent | **0.057738** | **0.031403** | **0.067223** | 0.021364 | **0.033113** | 0.032819 | **0.033219** | 0.016211 |

Stage 2 相比 Stage 1：NOVEL full 下降约 7.2%，hole 下降约 20.2%，visible 下降约 4.6%；
REAL-REF 几乎逐位相同。这与 factorized 设计一致：只改变 NOVEL/Identity 的监督频率，
Appearance 从真实照片学到的结果没有被扰动。

### 12.6 冻结统一评估：结论为“高频碎裂减弱，但暗孔代理未解决”

为避免训练日志与 checkpoint/采样条件不一致，新增冻结评估：

- 入口：`scripts/eval_frequency_consistent_geometry.py`；
- 目录：`experiments/_frequency_consistent_geometry_evaluation/`；
- 完整总览：`stage_comparison_overview.png`；
- NOVEL hole 聚焦总览：`novel_hole_focus_overview.png`；
- 原始指标：`metrics.json` / `metrics.csv`。

三个 checkpoint 使用完全相同的固定 validation batch、low-geometry condition、50-step DPM++
与 seed=42。NOVEL 统一结果：

| stage | hole L1 | hole edge error | dark ratio | hole gradient | dark components ≤4 | dark components ≤16 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 0 frozen | 0.085458 | 0.012586 | 0.046326 | 0.010539 | 223.7 | 264.0 |
| Stage 1 full EG3D | 0.039342 | 0.017986 | 0.114243 | 0.016444 | 325.3 | 380.3 |
| Stage 2 frequency-consistent | **0.031401** | **0.014130** | 0.113508 | **0.012233** | 346.0 | 397.7 |

Stage 2 相比 Stage 1：

- hole edge error 下降约 21.4%，hole gradient 下降约 25.6%，支持“EG3D 高频监督确实是
  碎裂高频的重要来源”；
- dark ratio 只下降约 0.6%，没有回到 Stage 0；
- 暗色小连通域代理没有下降，`≤4` 反而约增加 6.4%，`≤16` 约增加 4.6%。

暗组件统计会把黑发、眼睛和黑背景也计入，因此不能单独证明“小孔变多”；但它足以否定
“仅凭 L1/edge 数值即可宣布问题解决”。当前证据更准确的表述是：

```text
频率一致监督显著削弱了 Stage 1 带回的高频碎裂能量，
但 dark pinhole 是否肉眼消失仍未被自动指标确认。
```

### 12.7 当前停止点与下一阶段分支

现在不继续到 step 500/2000，也不立即增加新 loss。请优先肉眼查看：

1. `experiments/_frequency_consistent_geometry_evaluation/novel_hole_focus_overview.png`；
2. `experiments/_frequency_consistent_geometry_evaluation/stage_comparison_overview.png`；
3. 必要时查看 `images/id{0,1,2}_novel_stage{0,1,2}*.png` 原尺寸单图。

按观察结果进入唯一对应分支：

- **若 Stage 2 边缘保留且细碎孔明显少于 Stage 1**：方向成立，下一步才扩到 step 500，观察
  是否继续改善或重新反弹；
- **若碎纹变柔和但暗孔数量仍明显相同/更多**：下一项应做 deterministic VAE condition/target
  encode 消融，检查两次 posterior sample 是否在碎片 mask 上形成随机微结构；
- **若孔洞少了但脸部边缘仍过平**：不要恢复 EG3D 高频；应研究把 REAL Appearance 学到的真实
  高频迁移到 NOVEL hole，同时保持 Identity 只负责低频几何；
- **若几何、身份或背景连续性退化**：停止该方向，复查低通尺度，而不是回到消融 A 式无监督。

---

## 13. Stage 2 肉眼反馈与 Stage 3：确定性 VAE 对齐消融

### 13.1 用户肉眼反馈

对 Stage 0/1/2 的实际观察结论是：

- 碎孔有一些变化，但整体更像原地踏步；
- 仍存在大量单像素级碎孔；
- 视觉上呈现“没有对齐”的感觉；
- Stage 2 没有带来可感知的明显优化。

因此 Stage 2 不能判为成功，也不扩到 step 500。该反馈与自动评估中 dark ratio 基本不变、
暗色小连通域未下降一致：降低平均高频能量不等价于消除离散单像素孔。

### 13.2 尚未消除的训练/推理不一致

Stage 2 虽然令 NOVEL 的 target RGB 与 condition RGB 完全相同，但训练 latent 仍是：

```text
target_latent    = VAE(condition_rgb).posterior.sample_A
condition_latent = VAE(condition_rgb).posterior.sample_B
```

二者来自同一 RGB，却是两次独立 posterior sample。正式采样/推理的 BrushNet condition 使用
`posterior.mode()`。因此仍存在两个可合并为一个严格变量的问题：

1. target 与 condition latent 在训练中不逐元素对齐；
2. BrushNet condition 的训练编码是 sample，推理编码是 mode。

标准 diffusion target 使用 posterior sample 本身并非错误；本实验只检验它是否在当前高度碎片化
mask 与“target=condition”的特殊设定下，成为单像素随机微结构的来源。

### 13.3 Stage 3 唯一变量

入口：`scripts/run_deterministic_low_geometry_finetune.py`  
目录：`experiments/_deterministic_low_geometry_finetune/`  
初始化：原正式 `best_model.pt`  
停止：step 250。

其余 Stage 2 设置全部不变，仅对 `NOVEL + low_geometry` 执行：

```text
shared_latent = VAE(condition_rgb).posterior.mode()
target_latent = shared_latent
BrushNet condition image latent = shared_latent
```

REAL-REF 仍使用原有 posterior sample，不改变 Appearance 学习。checkpoint 额外记录：

```text
novel_low_geometry_deterministic_vae = true
```

### 13.4 判定标准

Stage 3 必须在相同 validation batch、condition、seed 和采样器下与 Stage 1/2 比较：

- 核心标准不是 L1，而是单像素碎孔是否肉眼明显减少；
- dark components `≤4`/`≤16` 应至少相对 Stage 2 明显下降，而不是只有 gradient 下降；
- REAL-REF 应继续与 Stage 1/2 一致；
- 若仍原地踏步，则 posterior 随机性被排除，下一根因应直接转向 **碎片化 mask 在 64×64 latent
  尺度的拓扑/边界耦合**。此时不再继续 lowpass 或长程训练，也不采用会破坏人脸的像素级 mask
  腐蚀；应测试 condition 与 loss 使用连续 reliability/距离场，而不是椒盐式二值边界。

### 13.5 首次 Stage 3 pilot 的随机轨迹混杂与修正

首次实现只做一次 posterior mode encode，不再调用 Stage 2 的两次 `posterior.sample()`。pilot 虽正常
结束到 step 250，但这会改变 CUDA RNG 的推进位置，使后续 diffusion noise、timestep 以及后续
REAL-REF batch 的随机轨迹均与 Stage 2 不同。其 REAL full 为 0.034534，而 Stage 2 为 0.033113，
证明确实存在实验轨迹混杂。因此该结果只保留为：

```text
experiments/_deterministic_low_geometry_confounded_pilot/
```

不能用于 posterior mode 的因果结论。

正式 Stage 3 修正为：

```text
target posterior.sample()     # 消费 RNG 后丢弃
condition posterior.sample()  # 消费 RNG 后丢弃
shared = target posterior.mode()
target_latent = condition_latent = shared
```

并仍执行两次 VAE encode，从而保持 Stage 2 在 diffusion noise/timestep 之前的编码次数、sample 次数、
sample shape 与 RNG 消耗顺序。正式 Stage 3 从原 checkpoint 在新目录重新运行。

### 13.6 严格 Stage 3 结果：posterior 随机错位被排除

正式 Stage 3 正常完成到 step 250，exit code=0，无 NaN/OOM。RNG 对齐有效：其训练轨迹与
Stage 2 基本逐位复现，例如：

| step | metric | Stage 2 | strict Stage 3 |
|---:|---|---:|---:|
| 0 | novel loss | 0.014569 | 0.014569 |
| 0 | identity grad | 0.106028 | 0.106030 |
| 100 | novel loss | 0.022031 | 0.022032 |
| 100 | identity grad | 0.084806 | 0.084837 |
| 200 | novel loss | 0.021905 | 0.021904 |
| 200 | identity grad | 0.054853 | 0.054846 |

step-250 训练验证同样近乎相同：

| stage | novel full | novel hole | novel visible | real full | real hole | real visible |
|---|---:|---:|---:|---:|---:|---:|
| Stage 2 | 0.057738 | 0.031403 | 0.067223 | 0.033113 | 0.032819 | 0.033219 |
| strict Stage 3 | 0.057739 | 0.031402 | 0.067224 | 0.033117 | 0.032819 | 0.033224 |

四阶段统一冻结评估：

| metric | Stage 2 | strict Stage 3 |
|---|---:|---:|
| NOVEL hole L1 | 0.031401873 | 0.031401351 |
| hole edge error | 0.014129082 | 0.014129480 |
| dark ratio | 0.113522857 | 0.113498844 |
| hole gradient | 0.012231757 | 0.012232236 |
| dark components ≤4 | 345.33 | 345.00 |
| dark components ≤16 | 397.00 | 396.67 |

两者在实验分辨率内等价；聚焦总览已更新为四列 checkpoint 对比：

- `experiments/_frequency_consistent_geometry_evaluation/novel_hole_focus_overview.png`；
- `experiments/_frequency_consistent_geometry_evaluation/stage_comparison_overview.png`。

**结论：NOVEL target 与 BrushNet condition 的独立 posterior sample 不是单像素碎孔主因。**
deterministic mode 对训练轨迹、最终指标和单像素暗组件均无实质影响，该方向停止。

### 13.7 根因收敛与下一阶段

目前已依次排除或限定：

1. 继续增加采样步数；
2. cycle / SNR weighting；
3. Reference 梯度隔离；
4. 缺少 low-frequency target-view geometry condition；
5. EG3D 高频监督（它放大碎裂能量，但删除后单像素孔仍在）；
6. target/condition VAE posterior 随机错位。

剩余最符合“单像素碎孔、没有对齐感”的机制是：

```text
高分辨率碎片化 binary visibility mask
  -> nearest downsample 到 64×64 latent
  -> 同时作为 BrushNet mask channel 与 noise/x0 loss 的硬分区
  -> Reference local validity 又使用另一尺度的硬可见性
  -> 多个空间场在边缘/细岛位置形成离散拓扑错配
```

这不等价于“腐蚀 mask”。用户已经实测像素级腐蚀会严重破坏人脸，不能重走该路线。下一阶段
应首先做 **冻结权重、无训练** 的因果测试，只改变 latent 空间中的控制场表达：

1. binary-nearest：当前基线；
2. area-downsample reliability：将高分辨率 hole occupancy 用 area/average 降到 64×64，保留
   `[0,1]` 连续覆盖率；
3. distance-field reliability：不改变最终像素 hole，只在 BrushNet mask channel 使用边界连续
   距离场；
4. loss mask 与 BrushNet condition mask 解耦：loss 仍以原 hole 监督，BrushNet 接收连续空间场。

固定 checkpoint、图像 condition、seed、采样器和最终输出语义，只比较单像素暗组件、边缘连续性
和人脸几何。若冻结测试没有肉眼改善，就不进入训练；若某个连续场明显改善，再做 250-step 单变量
训练。当前停在该下一阶段设计/肉眼检查点。

---

## 14. 冻结 latent reliability 因果实验：软场被否定，area-binary 拓扑有效

### 14.1 实验设计

入口：`scripts/eval_latent_reliability_causality.py`  
目录：`experiments/_latent_reliability_causality/`  
checkpoint：Stage 2 `iteration_250.pt`。

固定以下全部变量：

- 三个 validation identities；
- 512×512 原始 pixel hole mask 与 low-geometry RGB condition；
- W+、Reference features 和 Reference validity；
- 50-step DPM++、seed=42 和初始噪声；
- 最终像素输出语义，不做 compositing，也不改变人脸 pixel mask。

唯一变化是送入 BrushNet 的 64×64 第五通道：

| variant | latent mask 定义 |
|---|---|
| nearest_binary | 当前 `nearest` 硬二值基线 |
| area_soft | 512→64 area occupancy，保留 `[0,1]` 连续值 |
| area_majority_binary | area occupancy `≥0.5` 后二值化 |
| signed_distance_r2 | nearest 拓扑上两格宽的 signed-distance 软边界 |

四种 field 已通过独立张量检查：shape 均为 `[B,1,64,64]`，值域严格在 `[0,1]`，原始
512×512 mask 未被修改。

### 14.2 第一轮结果

| variant | dark ratio | dark comp ≤4 | dark comp ≤16 | hole L1 | visible L1 | latent components |
|---|---:|---:|---:|---:|---:|---:|
| nearest_binary | 0.08521 | 345.7 | 397.0 | 0.02936 | 0.06746 | 18.7 |
| area_soft | 0.08622 | 351.7 | 408.3 | 0.03010 | 0.06779 | 5.3 |
| area_majority_binary | **0.07847** | **341.7** | **385.3** | 0.02979 | 0.06891 | 5.3 |
| signed_distance_r2 | 0.09016 | 361.0 | 417.3 | 0.03025 | **0.06710** | 18.7 |

结论：

- `area_soft` 和 `signed_distance_r2` 都使暗组件增多，**连续软 mask 假设被否定**；
- area occupancy 本身能把 nearest 产生的碎片 latent 拓扑从约 18.7 个连通域降到 5.3 个；
- 只有 area 后重新二值化才降低暗孔代理，说明有效因素是更合理的 cell occupancy 判定，
  不是模糊/软化边界；
- 这与像素级腐蚀不同：pixel mask、RGB condition 和最终 hole 均保持原样。

主要可视化：

- `experiments/_latent_reliability_causality/novel_hole_focus_overview.png`；
- `experiments/_latent_reliability_causality/output_overview.png`；
- `experiments/_latent_reliability_causality/latent_field_overview.png`。

### 14.3 area threshold 冻结扫描

由于 `0.5` 可能是偶然阈值，新增：

- 入口：`scripts/eval_latent_area_threshold_sweep.py`；
- 目录：`experiments/_latent_area_threshold_sweep/`；
- 阈值：`0.25 / 0.375 / 0.5 / 0.625 / 0.75`。

即一个 64×64 cell 中至少多少比例的 512×512 像素属于 hole，才让 BrushNet 将该 cell 视为 hole。

| variant | dark ratio | dark comp ≤4 | dark comp ≤16 | hole L1 | visible L1 |
|---|---:|---:|---:|---:|---:|
| nearest | 0.08521 | 345.3 | 397.0 | 0.02936 | 0.06746 |
| area ≥0.375 | 0.08103 | 352.3 | 398.3 | 0.02973 | 0.06898 |
| area ≥0.5 | 0.07847 | 342.3 | 386.0 | 0.02979 | 0.06891 |
| area ≥0.625 | **0.07497** | 327.3 | 368.7 | 0.03035 | 0.06856 |
| area ≥0.75 | **0.06856** | **321.7** | **367.3** | 0.03108 | **0.06817** |

阈值升高时暗组件总体下降，但 hole L1 上升，说明“BrushNet 认为是 hole 的 latent cell 更少”
会以几何/覆盖代价换取更少暗孔。不能仅靠绝对暗色统计选择 0.75。

### 14.4 针对用户所述“单像素碎孔”的专用指标

绝对亮度 `<0.08` 会把正常黑发、眼睛和背景算入。为贴近视觉问题，新增只分析已保存输出、
不重新采样的脚本：`scripts/analyze_latent_threshold_pinhole.py`。

两个定义：

1. `local_pinhole`：hole 内像素亮度 `<0.20`，且比自身 5×5 邻域均值突然暗 `>0.08`；
2. `target_relative_dark_defect`：EG3D target 对应位置亮度 `>0.15`，但生成结果亮度 `<0.08`。

第二项更严格地排除正常黑发/背景，直接测“本应亮却生成黑点”的缺陷：

| variant | local pinhole ≤4 | target-relative defect ≤4 | target-relative ratio |
|---|---:|---:|---:|
| nearest | 360.0 | 184.3 | 0.004176 |
| area ≥0.5 | 359.3 | 166.0 | 0.003483 |
| area ≥0.625 | **328.3** | **142.3** | **0.002950** |
| area ≥0.75 | **320.7** | **135.7** | **0.002747** |

`area ≥0.625` 相比 nearest：

- local pinhole `≤4` 平均下降约 8.8%；
- target-relative dark defect `≤4` 平均下降约 22.8%；
- target-relative defect ratio 下降约 29.4%；
- hole L1 上升约 3.4%，visible L1 上升约 1.6%。

更重要的是 target-relative `≤4` 在三个 identity 上方向一致：

| identity | area ≥0.625 相对 nearest |
|---:|---:|
| 0 | -15.5% |
| 1 | -60.0% |
| 2 | -40.3% |

而 `local_pinhole` 在 identity 1 不稳定，说明邻域指标仍受正常纹理影响；因此候选选择以
target-relative defect 为主，不以单一平均暗色组件决定。

缺陷位置图：

- `experiments/_latent_area_threshold_sweep/local_pinhole_map_overview.png`；
- `experiments/_latent_area_threshold_sweep/target_relative_dark_defect_map_overview.png`。

### 14.5 结论与下一阶段候选

本轮得出三个可证伪结论：

1. **连续 reliability / distance field 无效，不能进入训练。**
2. **nearest 512→64 的单点取样确实制造了不合理 latent 拓扑，是单像素碎孔的一个真实来源。**
3. **`area occupancy ≥0.625` 是当前较保守 Pareto 候选。** 0.75 暗缺陷更少，但 hole L1 代价
   更大，且 identity 0/1 的 target-relative 收益没有明显超过 0.625，因此不选 0.75。

下一阶段可做 250-step 单变量训练：

```text
pixel mask / RGB condition / Reference validity / loss mask：全部保持原样
BrushNet latent mask channel：area(mask, 64×64) >= 0.625
```

训练必须从原正式 checkpoint 开始，并在训练/validation/正式 inference 中使用同一 checkpoint
contract。核心成功标准仍是肉眼单像素碎孔和 target-relative defect，而不是仅看 hole L1。
当前冻结因果证据已足够进入该训练阶段，但本轮停在 contract 变更前，避免未经独立记录直接把
新 mask 语义混入生产路径。

### 14.6 结论撤回（后续模块级因果证据覆盖本节训练建议）

用户肉眼比较 `nearest / 0.625 / 0.75` 后认为三者长得基本相同、均有孔洞，并质疑阈值扫描
是否已经退化为调参。该质疑成立。

本节只能证明：对**已经训练出碎孔的 Stage-2 checkpoint**，改变 BrushNet latent mask 会让若干
自动暗点指标变化。它不能证明 mask 是训练产生碎孔的主因，因为已经写入 W+ 参数的伪影不会因
冻结推理时替换 mask 而消失。`0.625/0.75` 是阈值搜索，不符合继续定位根因的第一性原理要求。

因此正式撤回：

- “nearest mask 是主因”的表述；
- “area ≥0.625 已足够进入训练”的建议。

第 14 节实验与产物保留为负结果/次要敏感性证据，不再推进任何 area-threshold 训练。

---

## 15. 最终模块级定位：碎孔由当前自定义 W+ RCA projection，尤其 `to_out_wplus` 带回

### 15.1 先纠正“原版和各模块单独都没问题”的比较逻辑

原版 WarpGAN、ReferenceNet、BrushNet、外部 W+ Adapter 并不等价于当前组合：

#### 原版 WarpGAN

原版没有 Stable Diffusion latent mask，也没有 W+ cross-attention。它执行：

```text
hole RGB = EG3D novel inversion（hybrid input）
W+       = FFC CNN style modulation
```

代码上，`inpaintor_forward()` 将 W+ 传给 `ffc_style_resnet`；W+ 作用于 bottleneck style resblocks
和逐级 upsampling blocks（`ffc_style.py:176-188`）。整个像素 CNN、style 分支与 RGB 输入联合训练，
不存在 16 个冻结 SD cross-attention 层上的独立 W+ 空间残差输出。因此原版没有碎孔不能证明
当前 W+ RCA 正确。

#### ReferenceNet / REAL-REF

REAL-REF 明确 `disable_wplus=True`。它只证明：

```text
SD + BrushNet + Reference appearance
```

能够工作；完全没有测试：

```text
SD + BrushNet + trainable W+ cross-attention RCA
```

所以 REAL 很好、NOVEL 出孔恰好指向两者唯一的 W+ 路径差异。

#### 官方 BrushNet

官方 BrushNet 使用 masked image latent + mask，并训练 BrushNet 本体、以普通 diffusion MSE
监督；没有当前额外的 W+ RCA residual branch。官方 mask 下采样存在不代表与当前 W+ projection
组合后仍不会出现伪影。

#### 外部 W+ Adapter

仓库中的外部 W+ Adapter 只有独立：

```text
to_q_wplus / to_k_wplus / to_v_wplus
```

W+ attention output 与 base attention 相加后，二者共同经过冻结 UNet `attn.to_out`。它**没有**
当前项目自行增加的独立 `to_out_wplus`。当前实现则是 base 已经过 `attn.to_out` 后，再通过一个
独立且可训练的 `to_out_wplus` 注入残差。这是项目特有结构。

### 15.2 Identity 子模块 2×2：不是 mapper，是 cross-attention RCA

入口：`scripts/eval_identity_submodule_factorization.py`  
目录：`experiments/_identity_submodule_factorization/`。

Appearance 固定为 Stage-1 step 250；Mapper 与 W+ RCA 分别取原 checkpoint 或 step 250：

| combination | hole L1 | dark ratio | hole gradient | dark comp ≤16 |
|---|---:|---:|---:|---:|
| M0_R0 | 0.08305 | 0.05273 | 0.01082 | 271.7 |
| M250_R0 | 0.07928 | 0.06444 | 0.01047 | 255.3 |
| M0_R250 | **0.03979** | **0.11515** | **0.01683** | **377.7** |
| M250_R250 | 0.03934 | 0.11425 | 0.01644 | 379.3 |

结论：

- Mapper 单独更新只带来很小 dark-ratio 变化，未恢复主要高频碎裂；
- W+ RCA 更新在 mapper 完全保持原值时，已经几乎完整复现 Stage 1 的 hole L1、dark ratio、
  gradient 和暗组件；
- 先前“Identity/W+ 导致碎孔”的结论进一步收敛为 **W+ cross-attention RCA 参数更新**。

总览：`experiments/_identity_submodule_factorization/novel_hole_focus_overview.png`。

### 15.3 RCA gate × projection：不是 gate，是 Q/K/V/O 权重

入口：`scripts/eval_wplus_rca_gate_projection_factorization.py`  
目录：`experiments/_wplus_rca_gate_projection_factorization/`。

Mapper 固定原值，Appearance 固定 step 250，拆分 projection 与 scalar gate：

| projection | gate | hole L1 | dark ratio | hole gradient |
|---|---|---:|---:|---:|
| old | old | 0.08305 | 0.05272 | 0.01082 |
| **new** | old | **0.03977** | **0.11478** | **0.01677** |
| old | new | 0.08333 | 0.05247 | 0.01082 |
| new | new | 0.03979 | 0.11516 | 0.01683 |

结论：

- 只替换新 gate 几乎完全没有作用；
- 只替换新 Q/K/V/O、保持旧 gate，已经完整复现碎孔；
- 所以不是门开得太大，而是 **projection 学到的残差内容本身有问题**。

### 15.4 Q/K/V/O 组件拆分：`to_out_wplus` 是最大主效应

入口：`scripts/eval_wplus_rca_projection_components.py`  
目录：`experiments/_wplus_rca_projection_components/`。

Mapper、gate、Appearance 固定，只分别替换一个 projection：

| projection update | hole L1 | dark ratio | 相对基线 | hole gradient | 相对基线 |
|---|---:|---:|---:|---:|---:|
| none | 0.08305 | 0.05273 | — | 0.01082 | — |
| Q only | 0.07192 | 0.07867 | +49.2% | 0.01116 | +3.2% |
| K only | 0.07368 | 0.07353 | +39.4% | 0.01105 | +2.1% |
| V only | 0.08014 | 0.05704 | +8.2% | 0.01087 | +0.5% |
| **O only** | **0.04382** | **0.09284** | **+76.1%** | **0.01526** | **+41.1%** |
| Q+K+V+O | 0.03977 | 0.11475 | +117.6% | 0.01677 | +55.0% |

`to_out_wplus` 单独更新已经恢复大部分边缘高频与暗孔现象；Q/K 改变 attention 空间选择后进一步
放大，V 单独作用很小。参数交换是冻结、同 batch、同 condition、同 seed 的因果实验，因此当前
能够确认的最小主因是：

```text
NOVEL 伪监督
  -> 训练 16 个 cross-attention 层的独立 W+ RCA projections
  -> 尤其 to_out_wplus 学成高频空间残差输出器
  -> Q/K 将该输出定位到目标视角边缘/hole 相关 query
  -> 脸部边缘与单像素碎孔一起被带回
```

这解释了用户观察到的“step 0 抹平孔洞，step 250 随脸部边缘一起浮现”。它不是 BrushNet
condition 本身突然变坏，而是训练后的 W+ RCA projection 覆盖/叠加了 BrushNet 的平滑结构先验。

总览：`experiments/_wplus_rca_projection_components/novel_hole_focus_overview.png`。

### 15.5 W+ contrast 不是当前主因

代码中存在 correct-vs-wrong W+ margin loss，但长期日志显示：

```text
correct x0 ≈ 0.0201
wrong-W+ x0 ≈ 0.1765
contrast loss = 0
```

即模型很早已满足 margin，contrast 项不再产生梯度。短程 Stage 1/2 日志也没有非零 contrast
证据。因此它在结构上可疑，但现有证据不支持将其认定为碎孔的实际驱动力。主梯度仍来自 NOVEL
noise/x0 pseudo supervision 对 W+ RCA projection 的更新。

### 15.6 当前确定程度与下一步

现在可以明确区分：

#### 已确定

1. 碎孔随 W+ Identity 更新而来，而非 Appearance；
2. Identity 中不是 Mapper 主导，而是 cross-attention RCA；
3. RCA 中不是 gate 主导，而是 Q/K/V/O projection；
4. projection 中 `to_out_wplus` 是最大单项主效应，Q/K 协同放大；
5. 当前项目额外可训练独立 O，与外部 W+ Adapter 和原版 WarpGAN 都不同。

#### 尚未声称

- 不能说所有碎孔 100% 只由 O 产生；Q/K+O 的协同仍贡献额外恶化；
- 不能说 mask 完全无影响；它改变输出敏感性，但不是主要训练根因；
- 尚未证明“冻结 O 后训练”一定同时保留足够身份/边缘质量，需要下一次 250-step 实验验证。

#### 下一步唯一合理实验

从原正式 checkpoint 重跑 low-geometry/frequency-consistent 250 steps，只改变：

```text
freeze to_out_wplus.weight
freeze to_out_wplus.bias
```

Mapper、Q/K/V、gate、Appearance、BrushNet condition、mask 和监督均保持不变。不同时调整阈值、
loss weight 或 mask。若冻结 O 后碎孔显著减少而身份/边缘仍能由 Q/K/V 学习恢复，则根因修复成立；
若 Q/K 仍把碎孔带回，再进一步冻结 Q/K 或回到外部 W+ Adapter 的共享 frozen `attn.to_out` 结构。

当前不再进行 mask threshold 训练或继续无目标消融。

---

## 16. Stage 4：冻结 `to_out_wplus` 的 250-step 因果训练

### 16.1 实验 contract

入口：`scripts/run_freeze_wplus_output_finetune.py`  
目录：`experiments/_freeze_wplus_output_finetune/`  
初始化：原正式 `best_model.pt`  
停止：step 250。

完全复现 Stage 2 的：

- seed=2107、batch 顺序、factorized joint；
- low-geometry condition 与 `novel_hole_target=low_geometry`；
- stochastic VAE posterior sample；
- identity/appearance LR、loss、mask、BrushNet、Reference 和 validation。

唯一变化：

```text
freeze all 16 cross-attention processors:
    to_out_wplus.weight
    to_out_wplus.bias
```

新增配置与 checkpoint contract：

```yaml
training:
  wplus_rca:
    freeze_output_projection: true
```

```text
wplus_rca_contract = frozen_output_projection
```

### 16.2 启动前参数审计

完整 Coach 初始化预检结果：

```text
W+ processors                  = 16
frozen O weight/bias tensors   = 32
active Q tensors               = 16
active K tensors               = 16
active V tensors               = 16
active gate tensors            = 16
active Appearance tensors      = 96
```

32 个 O tensor 全部：

- `requires_grad=False`；
- 不在 Adam optimizer parameter groups；
- 不出现在 factorized Identity active group。

Mapper、Q/K/V、gate 和 Appearance 仍可训练，Identity/Appearance 参数组无重叠。Identity 可训练
参数从 Stage 2 的 45.93M 降至 33.53M，减少约 12.40M，正好对应 16 个独立 O projection。

### 16.3 训练状态与验证指标

训练正常结束，exit code=0，无 NaN/OOM。step 0 基线精确复现 Stage 2。Appearance loss/gradient
轨迹基本一致，说明随机轨迹与 REAL 分支未被改变。

| stage | novel full | novel hole | novel visible | novel edge | real full | real hole | real visible |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage 0 | 0.074590 | 0.085453 | 0.070677 | 0.019835 | 0.039763 | 0.040587 | 0.039466 |
| Stage 2 | 0.057738 | 0.031403 | 0.067223 | 0.021364 | 0.033113 | 0.032819 | 0.033219 |
| freeze-O | 0.057937 | **0.029594** | 0.068144 | **0.020878** | **0.033113** | 0.032819 | **0.033219** |

冻结 O 没有导致 Identity 分支失去学习能力；NOVEL hole L1 反而比 Stage 2 再下降约 5.8%。REAL
指标与 Stage 2 一致，验证了 Appearance 隔离。

### 16.4 checkpoint 参数事实验证

统一评估入口：`scripts/eval_freeze_wplus_output_finetune.py`  
目录：`experiments/_freeze_wplus_output_evaluation/`。

相对原 checkpoint 的 projection 参数变化：

| group | Stage 2 mean relative change | freeze-O mean relative change |
|---|---:|---:|
| O | 0.031322 | **0.000000** |
| Q | 0.007127 | 0.009316 |
| K | 0.007057 | 0.009220 |
| V | 0.013926 | 0.020018 |
| gate absolute change | 0.000477 | 0.000527 |

O 的 mean/max change 都精确为 0，证明不是“名义冻结”。Q/K/V 和 gate 实际更新，并且比
Stage 2 变化更大，说明其余 projection 在补偿被冻结的 O 输出自由度。

### 16.5 同 seed 冻结采样结果

Stage 0、Stage 2、freeze-O 使用相同 validation batch、low-geometry condition、50-step DPM++
和 seed=42：

| metric | Stage 2 | freeze-O | relative change |
|---|---:|---:|---:|
| full L1 | 0.057738 | 0.057937 | +0.35% |
| hole L1 | 0.031402 | **0.029593** | **-5.76%** |
| visible L1 | 0.067223 | 0.068145 | +1.37% |
| hole edge error | 0.014130 | **0.013763** | **-2.60%** |
| dark ratio | 0.113532 | **0.101388** | **-10.70%** |
| hole gradient | 0.012232 | **0.011867** | **-2.98%** |
| dark components ≤4 | 345.3 | **320.0** | **-7.34%** |
| dark components ≤16 | 397.0 | **367.0** | **-7.56%** |
| local pinhole ≤4 | 363.7 | **326.7** | **-10.17%** |
| target-relative dark defect ≤4 | 183.7 | **160.7** | **-12.52%** |

所有碎孔代理指标方向一致下降，而不是只有 EG3D L1 变化。这支持第 15 节的因果定位：独立可训练
`to_out_wplus` 确实是碎孔的重要来源。

### 16.6 结论边界

本实验是**成功的方向验证，但不是问题已解决**：

- 改善幅度约 7%–13%，自动指标仍明显高于平滑 Stage 0；
- Q/K/V 在 O 冻结后发生更大参数变化，存在补偿路径；
- visible L1 轻微变差约 1.4%；
- 仅凭数值不能判断用户肉眼是否能感知单像素孔减少。

因此不自动扩到 500/2000 steps，也不立即同时冻结 Q/K。先肉眼比较：

- `experiments/_freeze_wplus_output_evaluation/novel_hole_focus_overview.png`；
- `experiments/_freeze_wplus_output_evaluation/overview.png`；
- 原尺寸：`experiments/_freeze_wplus_output_evaluation/images/id{0,1,2}_{stage2,freeze_O}.png`。

下一分支：

- 若肉眼有明确改善且边缘/身份可接受：继续做“冻结 O + 限制 Q/K”的最小因果实验；
- 若仍几乎相同：说明 Q/K/V 已通过旧 O 有效补偿，应直接测试回退到外部 W+ Adapter contract，
  即共享冻结 `attn.to_out`，而不是继续逐个调权重；
- 若明显过平或身份变弱：O 同时承担必要结构恢复，需要改成低频/层级受限输出，而非完全冻结。

---

## 17. Stage 5：共享冻结原生 `attn.to_out` 的 250-step 单变量训练

### 17.1 第一性原理 contract

入口：`scripts/run_shared_wplus_output_finetune.py`  
训练目录：`experiments/_shared_wplus_output_finetune/`  
统一评估：`scripts/eval_shared_wplus_output_finetune.py`  
评估目录：`experiments/_shared_wplus_output_evaluation/`。

本实验从未经过 Stage 1/2/4 微调的原正式 checkpoint 开始：

```text
experiments/train_inpaintor/
[20260809-104822]_rca_v5_equivariant_from_scratch/checkpoints/best_model.pt
```

与 Stage 2 保持相同：

- seed=`2107`、250 steps、factorized joint；
- Identity LR=`2e-5`、Appearance LR=`5e-6`；
- low-geometry BrushNet condition，kernel=`31`；
- NOVEL target=`low_geometry`；
- 原 pixel mask、latent mask、Reference validity、loss 和 validation batch；
- REAL-REF 的 LPIPS/PatchGAN 与初始化 discriminator。

唯一结构变量是 W+ 输出 contract：

```text
旧 independent:
    frozen_attn_to_out(base_attention)
    + gate * trainable_to_out_wplus(wplus_attention)

新 shared_frozen_to_out:
    frozen_attn_to_out(base_attention + gate * wplus_attention)
```

新配置项：

```yaml
training:
  wplus_rca:
    output_mode: shared_frozen_to_out
```

共享模式下，16 个 `to_out_wplus` 分支被结构性旁路；其 32 个 weight/bias tensor 全部冻结并从
Adam 参数组排除。Q/K/V、gate、Mapper 与 Appearance 仍可训练。checkpoint 保存：

```text
wplus_rca_contract = shared_frozen_to_out
wplus_output_mode   = shared_frozen_to_out
```

正式 `DiffusionInpaintor` 会读取 checkpoint metadata 并恢复同一前向 contract，避免训练、validation
和正式 inference 不一致。

### 17.2 启动前正确性验证

轻量 attention 单元测试通过：

1. gate=`0` 时 independent 与 shared 输出数值严格一致；
2. shared 模式下 Q/K/V 和 gate 均有梯度；
3. `to_out_wplus.weight/bias` 无梯度。

完整 Coach 预检通过：

```text
SHARED_OUTPUT_CONTRACT_AUDIT_OK processors=16 frozen_O_tensors=32
COACH_SHARED_PREFLIGHT_OK
```

### 17.3 训练状态

训练正常结束，exit code=`0`，无 NaN/OOM/safety stop。identity/Appearance 梯度均有限：

| step | appearance loss | novel loss | appearance grad | identity grad |
|---:|---:|---:|---:|---:|
| 0 | 0.036698 | 0.029943 | 0.297562 | 0.026068 |
| 100 | 0.014459 | 0.119176 | 0.087754 | 0.212774 |
| 200 | 0.022827 | 0.100572 | 0.155440 | 0.155247 |

训练 validation：

| stage | novel full | novel hole | novel visible | novel edge | real full | real hole | real visible |
|---|---:|---:|---:|---:|---:|---:|---:|
| shared step 0 | 0.076823 | 0.092088 | 0.071325 | 0.014918 | 0.039763 | 0.040586 | 0.039467 |
| shared step 250 | 0.077663 | **0.059772** | **0.084106** | 0.021766 | **0.033109** | 0.032811 | **0.033217** |

共享结构能在 250 steps 内把 hole L1 从 0.0921 降到 0.0598，证明 Identity 分支没有失去全部
几何学习能力；但 visible L1 上升到 0.0841，使 full L1 略差于 step 0。该现象提示共享投影限制了
高频暗孔，同时当前 Q/K/V 训练把几何修正扩散到可见区。

checkpoint：

```text
experiments/_shared_wplus_output_finetune/checkpoints/iteration_250.pt
```

### 17.4 checkpoint 参数事实

相对原正式 checkpoint 的全层聚合 relative change：

| projection | relative change |
|---|---:|
| Q | 0.013632 |
| K | 0.014720 |
| V | 0.050254 |
| O | **0.000000** |

O 精确不变，Q/K/V 实际更新；因此结果不是“名义共享但仍由独立 O 学习”。V 的变化最大，说明
在共享原生输出基底下，Identity 主要通过更新被读取的 W+ value 内容补偿几何，而不再拥有独立
高频输出基底。

### 17.5 同 batch、同 seed 的统一冻结评估

所有 checkpoint 使用同一 validation batch、low-geometry condition、50-step DPM++ 和 seed=`42`：

| metric | Stage 0 | Stage 2 independent | freeze-O | shared-trained |
|---|---:|---:|---:|---:|
| NOVEL full L1 | 0.074591 | **0.057738** | 0.057936 | 0.077661 |
| NOVEL hole L1 | 0.085458 | **0.031402** | 0.029593 | 0.059771 |
| NOVEL visible L1 | 0.070677 | **0.067223** | 0.068144 | 0.084104 |
| dark ratio `<0.08` | **0.04632** | 0.11351 | 0.10138 | 0.05583 |
| local pinhole `≤4` | **179.3** | 362.0 | 327.3 | 219.3 |
| target-relative defect `≤4` | **37.3** | 184.0 | 162.0 | 63.7 |
| target-relative defect ratio | **0.000798** | 0.004133 | 0.003431 | 0.002510 |

shared-trained 相比 Stage 2 independent：

- dark ratio 下降约 **50.8%**；
- local pinhole `≤4` 下降约 **39.4%**；
- target-relative defect `≤4` 下降约 **65.4%**；
- target-relative defect ratio 下降约 **39.3%**。

这远强于 freeze-O 的约 7%–13% 改善，确认独立输出 contract，而不只是 O 的某次数值更新，是
碎孔的重要结构来源。

但 shared-trained 仍没有回到 Stage 0：

- local pinhole `≤4` 比 Stage 0 高约 22.3%；
- target-relative defect `≤4` 比 Stage 0 高约 70.5%；
- visible L1 比 Stage 0 高约 19.0%；
- full L1 比 Stage 0 高约 4.1%。

因此当前结果是**根因结构得到强验证，但 250-step 共享 contract 尚未达到生产 Pareto**。

REAL-REF 的 Stage 2 / freeze-O / shared-trained 指标近乎逐位相同：shared-trained real full/hole/visible
为 `0.033110 / 0.032814 / 0.033217`。这符合 REAL 禁用 W+ 的设计，也验证了单变量隔离。

### 17.6 肉眼检查产物与停止点

优先查看：

1. `experiments/_shared_wplus_output_evaluation/novel_hole_focus_overview.png`；
2. `experiments/_shared_wplus_output_evaluation/overview.png`；
3. `experiments/_shared_wplus_output_evaluation/target_relative_dark_defect_map_overview.png`；
4. `experiments/_shared_wplus_output_evaluation/local_pinhole_map_overview.png`；
5. 原尺寸：`experiments/_shared_wplus_output_evaluation/images/id{0,1,2}_novel_{stage0,stage2_independent,freeze_O,shared_trained}.png`；
6. 训练 validation：`experiments/_shared_wplus_output_finetune/logs/images/val/overview_step_000250.png`。

本轮按单变量要求停止，不扩到 500/2000 steps，不改 mask、loss 或学习率，也不把 shared checkpoint
写入正式 `configs/infer.yaml`。需要用户肉眼判定三件事：

1. 单像素黑孔/碎裂是否相对 Stage 2 明显减少；
2. 脸部角度、轮廓和身份是否仍足够；
3. visible 区是否出现可感知的漂移、涂抹或背景变化。

若碎孔明显减少但 visible 漂移不可接受，下一步应保持 shared output contract，只限制 W+ 注入的
空间层级或低频范围；不能重新开放独立 `to_out_wplus`，也不应回到 mask threshold 调参。

### 17.7 用户肉眼最终判定：shared contract 明确失败，撤回候选修复建议

用户查看 `experiments/_shared_wplus_output_evaluation/overview.png` 后给出明确反馈：

> `shared_trained` 已经“脏得不行”，完全不像同一个人，油画感达到此前最严重程度。

该肉眼结论优先于 dark-ratio、pinhole count 和 EG3D L1。本实验正式判定为：

```text
FAILED — identity collapse + severe global oil-painting contamination
```

因此立即停止并撤回本节末尾的阶段性建议：

- 不再把 `shared_frozen_to_out` 视为生产候选；
- 不继续 shared contract 的 500/2000-step 训练；
- 不在 shared contract 上继续调层级、频率、学习率或 mask；
- 不把 shared checkpoint 写入正式 inference；
- 保留代码、checkpoint 和可视化仅作为负结果与因果证据。

生产状态复核：

- `configs/infer.yaml` 未指向 shared checkpoint；
- `configs/train_inpainting.yaml` 默认仍为 `output_mode: independent`；
- shared checkpoint 只存在于隔离实验目录；
- 当前没有训练/评估进程，GPU 空闲。

### 17.8 根因层级修正：O 决定碎孔形态，更深根因是 W+ 被迫承担伪 RGB 重建

此前第 15–17 节把因果链收敛到独立 `to_out_wplus`。Stage 5 肉眼失败要求进一步区分：

1. **故障载体/形态机制**：独立可训练 O 使错误 NOVEL 高频以局部边缘、暗脉冲和单像素碎孔的
   形式直接写入 SD hidden states；
2. **更深训练根因**：NOVEL Identity/W+ 分支仍由伪 RGB/latent noise+x0 重建目标驱动，却没有
   真实 novel-view 外观监督，也没有严格限制其只能表达身份与低频几何；
3. **职责错配结果**：只要 W+ attention 仍有足够表达能力，优化器就会让它承担 pseudo target 的
   图像生成/外观重建，而不是只提供 identity/geometry condition。

Stage 5 给出了关键反事实：

```text
移除独立 O
  -> 局部黑孔统计明显下降
  -> 伪监督压力转移到共享 Q/K/V（V relative change = 5.03%）
  -> 局部暗孔变成更大范围的脏纹理、油画化和身份重绘
```

因此，shared contract 并没有消除错误学习，只改变了错误学习的出口和视觉形态。自动暗孔指标下降
是因为故障从“稀疏极暗点”变成“稠密低频/中频脏化与身份崩坏”，不是整体感知质量改善。这也解释
了为何：

- target-relative defect `≤4` 大幅下降；
- 但 visible L1 上升约 25%，full L1 上升约 35%；
- 用户肉眼观察到历史最严重的油画感和 identity collapse。

当前更准确的第一性原理结论是：

> **不能通过更换 W+ attention 的输出投影来修复职责错配。只要 NOVEL 伪图像重建 loss 直接训练
> 一个高容量、空间可寻址的 W+ 分支，它就会学习伪目标外观；独立 O 时表现为碎孔，共享 O 时
> 表现为全局油画化和身份崩坏。**

这覆盖“独立 `to_out_wplus` 是完整根因”的过强表述，但不撤销参数交换事实：独立 O 仍是当前
生产 checkpoint 中单像素暗孔的最大单项实现载体。修正后的层级为：

```text
深层根因：Identity/W+ 的训练职责与 NOVEL pseudo-RGB 重建目标不匹配
    ↓
空间放大：Q/K 把伪目标误差定位到目标视角 query
    ↓
当前独立结构的主要表现载体：to_out_wplus
    ↓
当前可见症状：hole/轮廓单像素黑孔与碎裂

若改成 shared output：
同一职责错配经 V/Q/K 补偿
    ↓
症状转化为全局油画化、脏纹理和身份崩坏
```

本轮在此返回，不启动新实验。下一步若恢复研究，必须先重新定义“W+ 在训练目标中允许承担什么
信息”，而不是继续在 Q/K/V/O、mask threshold 或训练步数上搜索。

