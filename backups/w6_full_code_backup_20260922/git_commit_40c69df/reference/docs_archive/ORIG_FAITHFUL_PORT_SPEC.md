# 基于原版 WarpGAN 的 Diffusion 化改造规格（老师方案 · 稳妥定稿）

> 定稿日期：2026-08-23。本规格是老师意见 + 原版代码语义（`warpgan_orig/WarpGAN-main/`，
> 老师本人实现）的忠实收敛。**歧义一律按原版默认配置裁决，不引入任何未经原版
> 或本项目先验验证的新机制。**历史先验引用见 `docs/PROJECT_HISTORY.md`。
>
> 老师原话（三轮汇总）：
> ① "ReferenceNet 的输入是 warp 两次的结果和 gan inversion 的结果(SVINet 的结果)"；
> ② "latent code 通过叠块与文本特征对齐，然后做 cross，训练用于对齐的部分，
>    不行的话再解开 QKV 训练，训练框架模仿 brushnet"；
> ③ "Unet 输入：x 训练加噪；训练目标：加的噪声"；
> ④ "消融：code 用 cross attention 注入，x_mirror 也用 referencenet 的方式加入和
>    code 并行（做优化）。随后直接把图像 warp 一次，和原先做损失。用合成数据做训练"；
> ⑤ "我们预想的不是纯粹的 inpaint"。

---

## 0. 定位与总原则

**这不是一个新方法，是把原版 WarpGAN 的 SVINet（FFC-ResNet）忠实翻译成
BrushNet + SD1.5 diffusion，其余一切保持原版语义。**

五条总原则（每条都有出处）：

1. **原版怎么用 EG3D，我们就怎么用**：渲染只当输入条件 + 合成数据的 GT，
   真实数据的监督目标只有真实源照片（源视角）。原版 `warp_pred: False`、
   `with_mask: False`、synth 批次 `pred_novel vs target_img` 是裁决标准。
2. **不是纯 inpaint**：输出是网络生成的完整图像（原版 FFC sigmoid 全帧输出 +
   全图 L1 `weight_known=10, weight_missing=0, with_mask=False`；BrushNet 官方
   从纯噪声全帧采样）。warp 是条件，不是保留像素，不做任何拼接。
3. **训练框架模仿 BrushNet**：ε-prediction、masked image+mask 条件通道、
   全 timestep [0,1000)（项目 v3.1 实证：只训低 t → 高噪声段漂移到条件均值=雾）。
4. **参数解冻按最小梯度**：先只训"对齐投影"（mapper），不够再解 QKV，且只在
   合成数据上解（域内自洽无伪目标；真实数据上解 QKV = §17.8 碎孔/油画事故）。
5. **RefNet 只喂干净完整图**：gan inversion 重建图（y_hat / y_hat_novel），
   碎 warp 图只配进 BrushNet 条件通道。

---


## 1. 总流水线（一张图）

```
═══════ 批次 A：合成数据（EG3D 同身份双视角精确配对，与真实批次 1:1 交替）═══════

 W_src(采样) → EG3D render → src_img, src_hat, src_depth, c_src
 W_src → EG3D render(c_novel) → target_img, target_hat, target_depth, c_novel
 x_mirror = flip(src_img), c_mirror = mirror(c_src), depth_mirror = flip(src_depth)

 novel pass（唯一允许 EG3D 渲染当 target 的地方）:
   条件 = hybrid 图[warp(src_img)可见区 + target_hat 填洞] + mask  → BrushNet 条件通道
   RefNet 输入 = target_hat（gan inversion 结果，干净完整图） ∥ x_mirror（并行）
   W+ tokens = mapper(W_src) → cross-attention
   noisy = add_noise(VAE(target_img), t), t∈[0,1000)        ← 老师③
   loss: ε-MSE vs ε ＋ 低 t x0 解码像素域损失 vs target_img ＋ latent 闭环(×0.1)

═══════ 批次 B：真实数据（FFHQ，无 novel GT）═══════

 pass 1（novel，结构同上，RefNet 输入=y_hat_novel）:
   条件 = hybrid[warp(x) + y_hat_novel 填洞] + mask；输出 pred_novel（full-frame）
   监督：弱 ε 锚（visible=inv_warp@1.0, hole=y_hat_novel@0.1）——diffusion 必需
   （原版 FFC 无此自由度；权重取自项目消融 A / v2 契约，见 §5 注）

 pass 2（source，真实监督的唯一落点，忠实原版 warp_pred=False 语义）:
   warp_warp_img = forward_warp(hybrid_novel, c_novel→c)        ← "warp 两次的结果"
   条件 = hybrid[warp_warp_img + y_hat 填洞] + mask(源视角) → BrushNet
   RefNet 输入 = y_hat（源视角 inversion 结果）∥ x_mirror
   noisy = add_noise(VAE(x), t)
   loss: ε-MSE（全图无分区，原版 with_mask=False）
       ＋ t<200: x0 单步解码(≈FFC 1-step 忠实翻译) 像素域损失 vs 真实 x
       ＋ latent 闭环: pred→256→encoder→还原 W+ (×0.1)

 推理 = 只跑 pass 1（RefNet 输入 y_hat_novel，训练/推理严格一致）
```

## 2. 与原版的逐项对照（改什么、不改什么）

| 原版组件（行号已核实） | 本方案 | 备注 |
|---|---|---|
| `ffc_style_resnet` 生成器，全量训练 | BrushNet(618M, 可训) + 冻结 SD1.5 UNet + 注入 adapter | 唯一的核心替换 |
| FFC 输入：hybrid 图 + inversion 通道 + mask + mirror 条件 + W+ style 调制 | hybrid 图+mask → BrushNet 条件通道；inversion 图 → RefNet；mirror → RefNet 并行；W+ → cross-attention tokens | 语义一一对应，载体更换 |
| `warp.hybrid=True`（hole 填 y_hat_novel） | 同：hole 填完整 EG3D 渲染 | 条件端无毒（v3.3 实证） |
| synth 1:1 交替，`pred_novel vs target_img` 全损失 | 同结构：ε-MSE + 低 t 像素域 + latent 闭环 | 唯一 EG3D 当 target 处，域内自洽 |
| 真实批次 pass1 无 loss（`pred_novel` 不进 `cal_inpaintor_loss`） | 弱 ε 锚（§5 注），可消融"完全无监督"档 | diffusion 加噪训练必需 target |
| pass2 输入 = `warp_warp_img`（`warp_pred: False` 默认） | 同：warp hybrid 图回源，**不 warp 模型输出** | 老师④"直接把图像 warp 一次"= 此 |
| pass2 输出 `pred_inv_warp` vs x 全损失 | x0 单步解码 → 像素域损失 vs x | 单步 x0 = FFC 1-step 的忠实翻译 |
| L1×10 / GAN×10 / FM×100 / PL×30 / ID×0.5 / latent×0.1 | L1×10 / PL×30 / ID×0.5 / latent×0.1 / LPIPS(可选)；**GAN/FM 默认 0** | GAN/FM 在 diffusion x0 残噪上两次发散，标注可选实验 |
| `with_mask=False` 全图 L1，不分区 | 同：全帧生成，无硬拼接 | 老师⑤"不是纯粹 inpaint" |
| `input_mirror='condition'`（非损失，mirror loss 注释关闭） | x_mirror 走 RefNet 与 code 并行 | 老师④原意 = 此 diffusion 化 |
| mask = **像素域 erode3 + blur21 软化后**才进生成器（`process_mask` L186-198，config 值生效，每次 forward 执行） | ❌ 未搬（曾误判为"死配置"，2026-08-27 核验推翻：见 §8.26） | 待补：条件端软化（v10 候选） |
| 判别器 r1×10 | PatchGAN 仅低 t 干净 x0，小权重起步+止损线 | real_ref 先例峰值 17.98GB |

## 3. EG3D 使用三纪律（违者必踩已记录的坑）

1. 渲染永远可作**输入条件**（hybrid 填洞、RefNet 参考图、W+ 本身）；
2. 渲染只可在**合成批次**当监督 target（`pred_novel vs target_img`，域内自洽）；
3. 真实批次的 target 只有**真实源照片**（源视角 pass 2）。

禁止：真实 novel 输出对 EG3D 渲染求高权重像素/latent 损失（油画感来源，v3 300k）；
禁止 hole 零监督（崩盘 0.108→0.27，消融 A）；禁止纯 depth 填洞条件（蓝灰，v3.1）。

---

## 4. 架构规格（四个模块）

### 4.1 生成主干（替换 FFC）
- 冻结 SD 1.5 UNet + VAE；BrushNet（官方预训练权重起步，`conditioning_channels=5`：
  masked-image latent 4 + mask 1）**解冻全量训练**，8-bit AdamW lr=1e-5，
  gradient checkpointing（24GB @512, bs=1 先例 15-17GB）。
- 条件图 = hybrid 图（hole 填完整 EG3D 渲染，原版语义），**不加** depth 第 6 通道、
  **不加**低通/噪声变体（均非原版；低通/噪声/纯 depth 的教训见 PROJECT_HISTORY §4）。

### 4.2 W+ 身份注入（替换 FFC style 调制，老师②）
- `w_mapper`：W+ [14,512] 分 4 段叠块投影 + 全局均值 token → [18,768]，
  LayerNorm 对齐 CLIP 文本域 → 与空 prompt 并行进 16 层 cross-attention（attn2）。
- gate 固定 tanh(3)≈0.995（w-plus-adapter 原配方，常数注入）。
- **训练阶梯（老师的最小梯度）**：
  - 阶段 S1：只训 mapper（对齐投影），其余 W+ 路径全部冻结；
  - 阶段 S2（S1 不够再上）：解开 Q/K/V，O 保持冻结——**只在合成批次上解**
    （域内自洽无伪目标；真实批次上解 QKV/O = §17.8 碎孔与油画事故的复现条件）；
  - 真实批次阶段 W+ 分支整体冻结（权重可用 Stage C `iteration_50000.pt` 或按
    本方案 S1/S2 重训）。

### 4.3 ReferenceNet 外观分支（原版无对应，老师①④的新增量）
- backbone：SD-UNet 副本，冻结，多尺度特征 {320,640,1280}。
- **输入 = 干净完整图**：novel pass 喂 `y_hat_novel`，source pass 喂 `y_hat`
  （gan inversion 结果，训练/推理都可得，严格镜像）；**x_mirror（水平翻转源图）
  与之并行**（batch 维拼接，一次前向），特征经独立 K/V adapter 注入 attn1
  （gate 初始化 0→0.5，K/V 热启动自 SD 自注意力权重——先验 16 教训）。
- mirror 的几何不精确无害（attention 不要求空间对齐）；它为 disocclusion 区域
  提供"对侧脸真实纹理"的先验，即原版 `input_mirror='condition'` 的 diffusion 化。
- 碎 warp 图、带洞图一律不进 RefNet（只进 BrushNet 条件）。
- 已知边界：RefNet 特征 warp 到 hole 内恒为零（splat 几何事实，Phase-2 证伪），
  **不做**特征 warp 注入 hole 的尝试。

### 4.4 判别器（可选，默认关闭）
- PatchGAN + lazy R1，仅看低 t（t<200）干净 x0 解码图与 cycle 回投图；
  GAN/FM 在 x0 残噪上两次发散（grad ×100、白爆），默认权重 0，
  仅在基线稳定后作为单变量可选实验，预注册止损线（grad >10 或指标连续恶化即停）。

## 5. 数据与监督规格

### 5.1 批次结构（忠实原版 1:1 交替）
每个 iteration：合成批次一次 G 更新 ＋ 真实批次一次 G 更新（原版 coach L489-560 模式）。
合成数据管线复用 `scripts/gen_synthimg.py` / `gen_novelview.py` 离线产物。

### 5.2 损失表（原版权重 → diffusion 化落点）

| 原版 | 权重 | 本方案落点 |
|---|---|---|
| L1（全图） | 10 | pass2 低 t x0 解码 vs 真实 x；合成批次 vs target_img（全图无分区） |
| ResNet_PL | 30 | 同上位置 |
| ID | 0.5 | 同上位置 |
| latent 闭环 | 0.1 | x0 解码图 → 256 → `gan.encoder_forward` → MSE vs W+ codes（原版 L1195-1200 忠实翻译，本项目首次补上） |
| LPIPS | 0（原版关） | 不加（尊重原版；如需与历史对齐可 ≤10 可选） |
| GAN / FM | 10 / 100 | **默认 0**（diffusion x0 残噪上发散两次）；§4.4 可选实验 |
| —（diffusion 必需） | — | 全批次 ε-MSE（主训练信号，全 timestep） |

### 5.3 真实批次 pass1 的弱 ε 锚（唯一超出原版的自由度，注明理由）
原版 pass1 无监督（FFC 不需要 target）。diffusion 的 ε 训练必须给 noisy 一个来源，
稳妥取项目已验证的 v2 契约：**visible 区 target = `Warper.inverse_warp(x)`**（干净
重投影，内部 lap 0.028≈源照片 0.037，非碎裂 splat 也非 EG3D），**hole 区 target =
y_hat_novel @ 权重 0.1**（消融 A：hole 必须有锚；0.1 为历史稳定值）。
消融留一档"pass1 完全无监督"（100% 忠实原版）作对照。

## 6. 消融矩阵（老师④，全部同 seed / 固定验证集 / 肉眼终审）

| # | 变量 | 基线 | 实验组 |
|---|---|---|---|
| 1 | x_mirror（RefNet 并行） | 关 | 开 ←老师钦点，第一个做 |
| 2 | RefNet 输入 | 仅源图 x | y_hat/y_hat_novel（inversion 图） |
| 3 | W+ 注入 | 冻结（Stage C 权重） | S1 只训 mapper →（不够）S2 合成数据解 QKV |
| 4 | pass1 监督 | 弱 ε 锚 | 完全无监督（原版语义） |
| 5 | GAN/FM | 0 | 小权重（10/100 原版值，带止损线） |

---


## 7. 已规避的历史坑（本设计 vs PROJECT_HISTORY §4 证伪清单）

| 证伪项 | 本方案如何规避 |
|---|---|
| 真实 novel 对 EG3D 高权重监督 → 油画 | 真实批次 target 只有源照片（pass2）+ 弱锚 hole@0.1 |
| hole 零监督 → 崩盘 | 合成批次全监督 + 真实弱锚 0.1 |
| 纯 depth 条件 → 蓝灰 | 条件 = 原版 hybrid（完整渲染填洞） |
| visible target=碎 warp → 玻璃感/雾 | visible=inv_warp（v2 契约） |
| W+ 解 QKV/O 吃伪目标 → 碎孔/油画/身份崩坏 | QKV 只在合成批次解；真实批次 W+ 冻结 |
| GAN/FM 在 x0 残噪上 → 白爆发散 | 默认 0，可选实验带止损 |
| timestep 钳制 → 雾蒙蒙 | 全范围 [0,1000)（v3.1 修复版） |
| RefNet 特征 warp 进 hole → 结构不可能 | 不做特征 warp，mirror 走全局 attention |
| mask 腐蚀/闭运算 → 不像人（先验 4，监督端教训） | 原始 splatting mask（**但条件端欠原版的 blur 软化**，2026-08-27 核验） |
| 512 双 DDIM 全链 → 24GB 放不下 | pass2 用 x0 单步（= FFC 1-step 忠实翻译），显存 15-17GB 先例 |
| 从油画域旧 checkpoint 初始化 → 跳不出 | 从零（BrushNet 官方预训练起步），mapper 零初始化 |

## 8. 实施顺序与预注册判据

**Step 0（冒烟，1 天内）**：搭 pass1+pass2 数据流；2 步冒烟验证：
ε-MSE 非零、pass2 像素损失非零、latent 闭环非零、W+ 无梯度（若用冻结档）、
RefNet 输入确为 inversion 图、峰值显存 <22GB。**打印有效配置逐项核对**
（事故录 #10/#11 教训：配置写了≠生效）。

**Step 1（基线，无 mirror，25K 步）**：
- 5K：合成批次 novel_hole 显著下降（结构学习成立），无蓝灰/崩盘；
- 10-15K：真实 pass2 cycle_l1 持续下降；novel_edge 进入 0.022-0.028 区间或更低；
- 25K：肉眼裁决——对照原版 FFC checkpoint 的输出与 `_real_reference_pretrain`
  （real_ref_hole 0.027）水平；固定 3 身份 + seed=42 总览图交用户终审。

**Step 2（消融 #1：+x_mirror，单变量）**：从 Step 1 checkpoint 同步起步，
预注册：5K 不劣化、10K novel_hole/edge 改善、大 yaw 样本 hole 纹理肉眼更接近源照片。

**Step 3（按需）**：消融 #3（S1 mapper → S2 合成解 QKV）；#5（GAN/FM 可选）。

**止损规则**：任一判据失败 → 停止，回到本 spec 与 PROJECT_HISTORY 追加分析，
不带病长跑；save_interval ≤2000；实验目录独立，`WARP_GAN_OVERWRITE=1` 才覆盖。

## 9. 待老师确认清单（拿本 spec 对齐，不阻塞 Step 0/1）

1. 真实批次 pass2 的 diffusion 化是否认可"x0 单步"（原版 FFC 1-step 的翻译）？
   若坚持完整二次采样，24GB 需降到 256 或换 48GB 卡（先验 17）。
2. RefNet 输入取"inversion 图（y_hat/y_hat_novel）"是否即老师本意？
   （"warp 两次的结果"在本 spec 中承担 pass2 的 BrushNet 条件图；如老师指它进
   RefNet，则 pass2 RefNet 输入换成填洞后的 warp_warp_img，一行改动。）
3. GAN/FM（原版 10/100）在 diffusion 下是否要求尽力恢复？（当前按发散先验默认 0。）
4. latent 闭环按原版 ×0.1 补上，确认无异议。

---

## 附：本 spec 与老师五句话的映射索引

| 老师原话 | 本 spec 落点 |
|---|---|
| ① RefNet 输入 = warp 两次的结果 + gan inversion 结果 | §4.3（inversion 图进 RefNet）、§1 pass2（warp_warp_img 进条件）、§9-2 |
| ② 叠块对齐 → cross → 先训对齐再解 QKV → 框架模仿 BrushNet | §4.2（mapper/阶梯）、§0 原则 3、§4.1 |
| ③ UNet 输入加噪、目标是噪声 | §1（ε-prediction 全 timestep） |
| ④ mirror 走 RefNet 并行 + warp 一次回源损失 + 合成数据 | §4.3（x_mirror）、§1 pass2、§5.1（1:1 交替） |
| ⑤ 不是纯粹 inpaint | §0 原则 2（full-frame，无拼接） |

