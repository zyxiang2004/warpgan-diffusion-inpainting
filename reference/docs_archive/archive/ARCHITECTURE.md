# WarpGAN-Diffusion 网络架构与代码实现审计（v3.1，2026-08-22）

> 逐模块审计：每条注入路径的理论角色 + 代码实现 + 审计结论。
> 入口：`training/coach_inpainting_static.py`（下称 coach）、
> `models/referencenet/attention_processor.py`（下称 processor）、
> `models/BrushNet-main/src/diffusers/models/brushnet.py`。

## 0. 总架构（一图）

```
输入: 源照片 x, 相机 c, 新视角相机 c_novel, W+ codes, depth, depth_novel
  │
  ├─ 3D forward warp (splat) ──→ warp_img (碎裂可见区) + hole mask
  ├─ inverse warp (grid)   ──→ inv_warp (干净可见区, 仅作 target)      [v2]
  │
  ├─ BrushNet 条件 (5ch latent):
  │     cond_rgb = warp×(1-mask) + depth灰度×3×mask                   [v3]
  │     brushnet_cond = cat([VAE.mode(cond_rgb), mask_down]) → [B,5,64,64]
  │
  ├─ ReferenceNet (SD-UNet 副本, 冻结): VAE(x)@t=0 → 多尺度特征 {320,640,1280}
  │     └─ 3D 对齐副本 aligned_feat + validity (warp 到目标视角)
  │
  ├─ w_mapper (WProjModel, 冻结): W+ [B,14,512] → tokens [B,18,768]
  │
  └─ 训练样本: noisy = add_noise(VAE(target), t), t ∈ [0,1000) 全范围   [v3.1 修复]
        │
        ▼
  BrushNet(noisy, t, brushnet_cond) → down/mid/up 残差 (618M, 可训)
        │  加法注入
        ▼
  SD UNet(noisy, t, empty_prompt, add_res, cross_attn_kwargs) → ε_pred
        │  残差注入点:
        │   ① attn2 (cross-attn, 16 层): W+ tokens 分支 (冻结, gate≈0.995)
        │   ② attn1 (self-attn, 16 层): ReferenceNet 全局 + 对齐局部分支 (可训, gate≈0.46)
        ▼
  损失: noise MSE(hole→lowpass EG3D@0.1, visible→inv_warp@1.0)
      + t<200: cycle L1(10)+LPIPS(10)+ResNet_PL(30)+ID(0.5)
      + real_ref 批次(25%): MSE@1.0 + pixel_sup(LPIPS+PatchGAN+R1)
```

## 1. W+ 注入（attn2, 16 个 cross-attention 层）— ✓ 正确

**理论角色**：身份条件。orig 中 W+ 做 FFC style 调制；我们用 attention 注入
（w-plus-adapter 范式），Stage-1 配对训练后冻结（§17.8 教训：伪目标梯度进 W+
必然学坏）。

**代码**（processor L127-151）：`hidden += to_out_wplus(Attn(Q_w(img), K_w(W+), V_w(W+))) × tanh(wplus_scale)`
- Q_w 来自图像 hidden states，K_w/V_w 来自 W+ tokens——标准 adapter 形式 ✓
- Q/K/V/O 热启动自 SD 文本注意力权重（`initialize_from_attention` L64-68）✓
- gate 固定 tanh(3)=0.995 ≈ 全强度（w-plus-adapter 原配方：常数注入）✓
- w_mapper：W+ [14,512] 分 4 段投影 + 全局均值→4 个 CLIP 域 token，拼成
  [18,768]，LayerNorm 对齐 CLIP 分布（`models/mapper/w_proj.py`）✓
- joint 模式全冻结（mapper + Q/K/V/O + gate），权重来自 Stage-1
  `iteration_50000.pt`（id_score=0.218）✓

**已知边界**：W+ 学的是"W+→EG3D 渲染"映射，非与生成器协同塑造——冻结的代价，
RefNet adapters 联合训练部分补偿。

## 2. ReferenceNet 注入（attn1, 16 个 self-attention 层）— ✓ 正确，两个已知边界

**理论角色**：源照片真实纹理条件（本项目增量，orig 无）。

**代码**（processor L152-205），两条路：
- **全局路**（L153-178）：RefNet 特征（按通道数 320/640/1280 匹配，空间不符时
  bilinear 对齐）→ to_k/to_v_reference → 与图像 query 做标准 attention →
  to_out_reference → `× tanh(reference_scale)`。门 hot-open 0.5→tanh=0.462 ✓
- **对齐局部路**（L180-205）：3D warp 到目标视角的 aligned 特征，**不做 attention**
  直接 to_v→to_out 逐位置注入，`× validity × tanh(reference_local_scale)`。
  K/V 热启动自 SD 自注意力权重 ✓

**已知边界**（记录在案，非 bug）：
1. hole 内 validity=0 → 局部路在 hole 内恒为零（消融日志成因③；设计使然——hole
   没有源图对应位置）。hole 外观靠：depth 几何 + RefNet 全局 attention + W+
2. real_ref 批次走 `validity=ones`（同视角），供它学到"如何用参考特征"

## 3. BrushNet（控制分支，618M，可训）— ✓ 正确

**理论角色**：hole 内容决策者 + ControlNet 式条件翻译器。

**代码**：
- 条件：`conditioning_latents = cat([VAE.mode(cond_rgb)×0.18215, mask_nearest])`
  → 5 通道（brushnet.py L141 `conditioning_channels: 5`），与 BrushNet 官方
  masked-image+mask 形式一致 ✓
- 前向（coach L1371-1378）：`brushnet(noisy_latents, t, brushnet_cond)` →
  down/mid/up 残差 → SD UNet 的 `down/mid/up_block_add_samples` 加法注入
  （与 ControlNet 同范式，代码验证 L740-790 conv 栈）✓
- joint 模式 `brushnet.training=True` → 梯度使能（L1368 的 no_grad 判断正确）✓
- v3 起 hole 填归一化逆深度（ControlNet-depth 约定，2-98 分位、近亮、3ch）——
  BrushNet 控制分支天生翻译 OOD 条件分布 ✓

## 4. 本次审计发现的 BUG（已修复，v3.1）

**训练 timestep 范围钳制**（coach 原 L1356-1362）：`real_cycle` 批次
`timestep_limit=min(1000, 200)` —— novel 分支只在 t∈[0,200) 训练，
**高噪声段从未训练**。而生产采样从纯噪声出发，50 步中约 40 步在 t>200 的
未训练区域运行 → 未训练的高 t 行为漂移到条件均值 = 雾蒙蒙存活 v1→v2.1 的
真正结构根源。修复：恢复全范围采样（cycle loss 内部本就有 t<200 自门控，
外层钳制是早期 x0-cycle 设计的残留事故）。BrushNet 官方训练即全范围。

**连带影响**：cycle loss 现在只在 ~20% 的样本上生效（t<200 命中率）——这是
正确的分工（高 t 由几何 MSE 管，低 t 由真实域感知损失管），非退化。

## 5. 运行配置（本次更新）

| 项 | 旧 | 新 |
|---|---|---|
| max_steps | 50,000 | **300,000**（~4 天 @0.9 step/s） |
| save_interval | 5,000（60 个 ×4.2GB=252GB，不可持续） | **25,000**（12 个 ≈50GB，磁盘 586GB 余量内；env `WARP_GAN_SAVE_INTERVAL` 可调） |
| 训练 timestep | novel 限 t<200（bug） | 全范围 [0,1000) |
| 验证/安全停机 | 每 1000 步；novel_full>0.09 连续 4 次且已武装才停 | 不变（未达标前不武装，不会误杀 depth 模式冷启动） |

checkpoint 含权重+优化器状态（~4.2GB/个，resume 必需）；手动删除旧 checkpoint
不影响后续 resume（只需保留最新一个）。
