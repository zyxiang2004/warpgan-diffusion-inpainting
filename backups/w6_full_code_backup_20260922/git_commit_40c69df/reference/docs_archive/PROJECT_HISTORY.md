# WarpGAN Diffusion-SVINet 项目全史（2026-05 ~ 2026-08-23）

> **本文档是项目唯一权威历史文档**，2026-08-23 由 docs/ 下此前全部 10 份文档
> （PROJECT_SUMMARY / TRAINING_AUDIT_REPORT / DIFFUSION_SVINET_EVOLUTION_REPORT /
> CODE_MODIFICATION_REPORT / NOVEL_HOLE_ABLATION_LOG / METHOD_INTRODUCTION /
> REAL_DOMAIN_CYCLE_GEOMETRY_PRIOR_PLAN / DUAL_DDIM_CYCLE_512 / ARCHITECTURE /
> JOINT_TRAINING_PLAN）合并归纳而成，并补充了 v6/v7 结局与当前 run 的实际日志核查。
> 旧文档已移入 `docs/archive/`，仅作考古用途，**内容以本文档为准**。
>
> 项目历时约 4 个月。坦率地说：经历了大量试错、多次 AI 分析错误导致的弯路，
> 至今没有达到"肉眼可接受的新视角补全"。本文档的第一目标就是让新 agent
> **不再重复任何一次已证伪的实验、不再犯一次已记录的分析错误**。

---

## 0. 给新 agent 的必读摘要（30 秒版）

1. **任务**：单张真实人脸照片 → 3D warp 到新视角 → 填补 disocclusion 空洞 →
   输出真实感新视角图像。用 SD1.5+BrushNet+ReferenceNet+W+ 注入替换原版 WarpGAN
   的 FFC-ResNet 修补器（原版输出"油画感"）。
2. **当前状态（2026-08-23 09:52 起）**：一个**从零、300K 步、整合全部已验证决策**
   的 `mode=joint` run 正在运行：`experiments/_joint_train_orig/`（日志
   `train_logs/joint_train_orig.log`，入口 `scripts/run_joint_train.py`，GPU 24GB）。
   此前的 v5/v6/v7 续训全部终止归档。
3. **三个月反复出现的核心矛盾**（理解了它就读懂了 90% 的历史）：
   - hole 内**没有真实 GT**（新视角本来就看不见）；
   - EG3D 渲染当 **loss 目标** → 塑料/油画感；完全移除 → 几何崩盘（消融 A）；
   - 高容量 W+ 分支被伪目标训练 → 碎孔/油画/身份崩坏（§17.8 终结论：冻结 W+）；
   - 监督权重的"纸面配置"与"实际生效"多次不一致（AI 幻觉重灾区，见 §6）。
4. **先读 §4（已证伪清单）和 §6（事故录）再动手**，这两节是本项目最贵的资产。
5. 方法论铁律：**单变量实验、冻结权重因果消融、预注册判据、肉眼终审优先于自动指标**。

---

## 1. 项目目标与背景

### 1.1 原版 WarpGAN（NeurIPS 2025）是什么

- 代码基线：`/data/xzy/warpgan_orig/WarpGAN-main`（本仓库 `warpgan_orig/` 有副本）；
  本工程 `/data/xzy/warpgan20260803/20260803` 是在其上的改造。
- 流程：真实照片 → GOAE(EG3D encoder) 反演得 W+ codes [B,14,512] → 按
  (c_src→c_novel) 做 3D forward warp（splatting）→ 破碎 warp 图 + hole mask →
  **SVINet** 补洞 → 新视角图像。
- 原版 SVINet = FFC-ResNet（LaMa 系）前馈修补器；`warp.hybrid=True` 时 hole 直接
  填 EG3D novel render 作逐像素几何草图；真实监督通过 **cycle**：novel 输出
  inverse warp 回源视角与真实源照片算 L1/GAN/FM/PL/ID。原版同样没有真实
  novel-view GT——这是任务本质，不是缺陷。
- 原版输出：结构与身份正确，材质稳定平滑 = **油画感**（FFC 容量小 + EG3D 材质泄漏）。

### 1.2 本项目的改造目标

保留 3D warp/相机/身份优势，把修补核心换成 diffusion：

| 原版组件 | 替换为 | 理由 |
|---|---|---|
| FFC-ResNet generator | 冻结 SD1.5 UNet + BrushNet | diffusion 生成先验远强于前馈 CNN |
| EG3D novel render 作 RGB 条件 | （最终回归 orig：作条件 + 弱权重作目标） | 见 §3.5 的 v3.3/v4 摇摆史 |
| 无 ReferenceNet | 冻结 RefNet + 对齐特征注入 | 从真实源照片注入皮肤/头发高频 |
| W+ FFC style modulation | WProjModel + W+ cross-attention RCA | 3D 身份潜码 → SD token |
| `generated*mask + warp*(1-mask)` 硬拼接 | SD full-frame 输出 | warp 可见区也有拉伸/重影，不可当最终像素 |

**输出契约（2026-07-29 定，一直有效）**：`FINAL = SD full-frame decode`，
warp/mask 只是条件，不做任何像素拼接。

### 1.3 任务数据的基本事实（贯穿全部讨论）

- 数据：FFHQ（+可选 LPFF）真实照片 + EG3D 渲染产物（depth、novel render、W+ codes
  离线生成）；512×512。
- 两个训练任务族：
  - **Task A / real_ref（自重建）**：用 novel warp 的 mask 形状在源照片上挖洞，
    target = 同一张源照片（精确像素配对）。W+ 禁用，逼 Reference 分支学真实纹理。
    长期收敛良好（hole L1 最好 0.026），是"完整监督链有效"的对照组。
  - **Task B / novel（新视角）**：target 只能是 EG3D 渲染（伪目标）或 cycle 回投
    真实源照片（间接监督）。**全部主战场**。
- 指标：`novel/real_ref × full/hole/visible L1`、edge error、dark ratio
  （<0.08 暗像素比）、dark components ≤4/≤16（单像素碎孔计数）、cycle_l1/lpips/id。

---

## 2. 当前系统架构（2026-08-23 审计版 v3.1）

入口：`training/coach_inpainting_static.py`（下称 coach）、
`models/referencenet/attention_processor.py`（processor）、
`models/BrushNet-main/src/diffusers/models/brushnet.py`。

### 2.1 总架构图

```
输入: 源照片 x, 相机 c, 新视角相机 c_novel, W+ codes, depth, depth_novel
  │
  ├─ 3D forward warp (splat) ──→ warp_img (碎裂可见区) + hole mask
  ├─ inverse warp (grid)   ──→ inv_warp (干净可见区, 仅作 target)     [v2]
  │
  ├─ BrushNet 条件 (5ch latent):
  │     cond_rgb = warp×(1-mask) + hole填充×mask   （hole 填充方式随版本变，见 §3.5）
  │     brushnet_cond = cat([VAE.mode(cond_rgb), mask_down]) → [B,5,64,64]
  │     (+可选第6通道: 全图归一化逆深度, conv_in_condition 零初始化扩展)
  │
  ├─ ReferenceNet (SD-UNet 副本, 冻结): VAE(x)@t=0 → 多尺度特征 {320,640,1280}
  │     └─ 3D 对齐副本 aligned_feat + validity (warp 到目标视角)
  │
  ├─ w_mapper (WProjModel): W+ [B,14,512] → tokens [B,18,768]
  │
  └─ 训练样本: noisy = add_noise(VAE(target), t), t ∈ [0,1000) 全范围  [v3.1 修复]
        │
        ▼
  BrushNet(noisy, t, brushnet_cond) → down/mid/up 残差 (618M, joint 模式可训)
        │  加法注入 (ControlNet 范式)
        ▼
  SD UNet(noisy, t, empty_prompt, add_res, cross_attn_kwargs) → ε_pred
        │  残差注入点:
        │   ① attn2 (cross-attn, 16层): W+ tokens 分支 (joint 模式冻结, gate=tanh(3)≈0.995)
        │   ② attn1 (self-attn, 16层): RefNet 全局 + 对齐局部分支 (可训, gate≈0.46)
        ▼
  损失: noise MSE(hole→EG3D低频@权重, visible→inv_warp@1.0)
      + t<200: cycle L1(10)+LPIPS(10)+ResNet_PL(30)+ID(0.5) @512
      + real_ref 批次(25%): MSE@1.0 + pixel_sup(LPIPS+PatchGAN+R1)
```

### 2.2 模块职责（一句话版，多轮讨论收敛）

```
BrushNet：hole 内容决策者；目标视角哪里已有观测、哪里需要生成
depth  ：目标视角几何边界在哪（唯一无歧义几何契约）
W+     ：这个人是谁（只许身份/低频，不许外观——§3.3-17.8 教训）
RefNet ：这个人的真实外观是什么（高频纹理来源）
SD     ：如何生成自然图像
```

### 2.3 已知设计边界（记录在案，非 bug）

1. hole 内 validity=0 → RefNet 局部路在 hole 内恒为零（hole=源照片看不到的区域，
   几何事实）；hole 外观靠 depth + RefNet 全局 attention + W+ + SD 先验。
2. real_ref 批次走 `validity=ones`（同视角），学"如何使用参考特征"。
3. W+ 学的是"W+→EG3D 渲染"映射（Stage-1 配对训练后冻结的代价）。
4. `warp.hybrid` 配置项在 diffusion 路径历史上长期未被读取（§3.3），后续版本已把
   hole 填充显式实现在 `build_geometry_condition()`。

### 2.4 当前正在运行的 run（2026-08-23 09:52 启动）

`experiments/_joint_train_orig/`，**从零**（checkpoint_path=null,
initialization_path=null），300K 步，入口 `scripts/run_joint_train.py`，
日志 `train_logs/joint_train_orig.log`。关键配置（实测日志头）：

```yaml
training:
  mode: joint                      # 单 run 联合
  align_reference_features: true   # RefNet 对齐特征路开启
  hole_target_mode: rgb            # 条件端 hole 填完整 EG3D 渲染（orig hybrid 忠实回归）
  geometry_condition: {enable: true, mode: rgb}
  depth_condition: {enable: true}  # BrushNet 第 6 通道 = 全图归一化逆深度
  novel_hole_target: inv_warp_visible   # visible 目标 = 源照片逆向 warp（干净重投影）
  real_cycle:                      # cycle 主监督 @512
    {resolution: 512, max_timestep: 200,
     l1_weight: 10.0, lpips_weight: 10.0, id_weight: 0.5,
     gan_weight: 0.0, fm_weight: 0.0,          # GAN/FM 在 x0 残噪上发散，禁用
     x0_clip: 3.0,
     geometry_noise_weight: 1.0,               # ★ hole 有监督（v4-v5 零监督教训修正）
     geometry_x0_weight: 1.0}
  edge_band: {enable: false}       # v6 机制未启用（保留代码）
  brushnet_cycle: {lr: 1.0e-05, use_adapter: false, use_8bit_adam: true}
  validation: {max_novel_full_l1: 0.09, violation_patience: 4, arm_after_first_pass: true}
data: {batch_size: 1, synth: {able: false}}     # 不再使用 synthetic 配对数据
```

语义：把 §3.5 中逐条验证过的有效决策（全 timestep 训练、inv_warp visible target、
hole 有目标监督、depth 第 6 通道、cycle@512、GAN/FM=0、8-bit AdamW 训 BrushNet）
整合成一次忠实 orig 翻译的全新长跑。停止线：novel_full 连续 4 次 >0.09（armed 后）。

---



## 3. 四个月时间线总览

| 时间 | 阶段 | 主线 | 结局 |
|---|---|---|---|
| ~2026-07-14 | ① diffusion 迁移 | BrushNet+SD 替换 FFC；W+/Ref 注入 | 目标视角错配 bug 使 0-74k 全无效 |
| 07-15~07-22 | ② v2→v3 | mirror-ref→real-ref；多尺度 RefNet；LPIPS/GAN | v3 300k 稳定基线 hole L1 0.101，但油画感 |
| 07-28~07-30 | ② v4→v5 | 独立 K/V/O；factorized joint；52k 事故；full-frame 契约 | 指标改善但油画感不变 → 从零重启 |
| 08-03~08-13 | ③ 碎孔攻坚 | novel hole 碎裂纹理 20+ 轮消融 | 定位到 W+ RCA to_out_wplus → 深层=W+ 职责错配 |
| 08-13~08-21 | ④ 真实域 cycle | Stage A/B/C 课程式 + 512 双 DDIM cycle | ref_cycle 两次发散；24GB 放不下 512 双 DDIM |
| 08-21~08-23 | ⑤ joint mode v1→v6 | 单 run 联合；条件/目标契约 v1→v6 迭代 | v1 雾、v3.1 蓝灰、Phase-2 证伪、v5 证伪、v6 归档 |
| 08-23 09:52~ | ⑥ 当前 | `_joint_train_orig` 从零 300K 整合 run | 运行中 |

### 3.1 阶段①（~07-14）：diffusion 迁移与致命 bug

- 完成 SD1.5+BrushNet 底座、WProjModel（W+→token）、旧版 W+/Reference 注意力注入、
  real/synthetic/novel 三类任务、共享 DPM-Solver++ 采样。
- **最贵的一次 bug——目标视角错配**：warp 是按 c_novel 投影的新视角图，训练目标
  却是源视角原图 `batch['x']`。"保持新视角几何"与"回归源视角"目标矛盾，
  0–74k 步验证图逐帧变化仅 1.8/255，**完全无效**。修复：目标改 `y_hat_novel`。
- 其他早期修复：旧 RCA 只有独立 K/V（Q/O 走冻结底座）→ 补全独立 Q/K/V/O；
  验证采样曾每步硬回填干净背景 latent（偏离 BrushNet 官方）→ 改纯噪声 50 步；
  推理断链（infer.py 只会加载 LaMa）→ `models/diffusion_inpaintor.py` 双后端。

### 3.2 阶段②（07-15~07-30）：v2→v3→v4→v5，架构成熟与油画感定位

**v2/mirror-ref（07-15 审计对象）**：曾用"源图水平翻转+镜像相机"造跨视角监督，
但水平翻转≠非对称人脸的镜像视角，LPIPS 有几何标签噪声 → 放弃。

**v3（07-22，300k 完整训练，历史稳定基线）**：
- mirror-ref 改 **real-ref self-reconstruction**（25%）：mask 形状来自 novel warp，
  RGB/target 都是源照片本身；W+ 禁用。real_ref_hole 收敛到 0.030-0.033。
- ReferenceNet 改**多尺度**特征（320/640/1280）。早期因果消融发现旧版 Reference
  贡献几乎为零（full 0.152 vs no-Ref 0.153），W+ 才是主信号（no-W+ 0.376）。
- **SNR 平衡 x0 loss**：raw x0 MSE 在 t=999 放大 214 倍导致 loss 0.07~3.5 震荡，
  改 `ᾱ×(x0_err/hole_area)` 等价 `(1-ᾱ)×ε_err`。
- **原生 latent patch LPIPS+PatchGAN+R1**：像素监督最初 OOM（+6GB 超 24GB）；
  第一次修复（先解码 512 再 resize 256）无效——计算图已建。最终：取空洞覆盖率
  最高 的原生 32×32 latent 窗 +4px 上下文解码成 256 patch 再算损失，峰值 17.98GB。
- 确定性验证：test loader shuffle=False、`fixed_novel_view=1`。
- **v3 300k 终局：hole L1 0.101（warp hole 0.335），结构/身份正确，油画感稳定**。
  已排除：训练步数不足、DPM++/DDIM 选择、SD VAE、checkpoint 未加载、mask 方向等。
  油画感根因排序：75% 监督是 EG3D 平滑 latent + Ref 跨视角不对齐 + 主干冻结自由度不足。

**v4（07-28）**：Reference 自注意力升级独立可训练 K/V/O residual adapter（v3→v4
function-preserving 初始化，数学等价 max err=0）；appearance pretrain（v4-A）、
同 batch 梯度隔离 factorized joint（v4-B）、连续 alpha-bar 权重（v4-C）。
双任务 Pareto：v4-B step200 最优（real-ref hole 0.157 / novel 0.105）。

**52k 恐怖谷事故（07-28）**：`[20260728-102109]_rca_v4a_reference_pretrain` 把本应
1-2k 步的 capability warm-up 误配成 300k appearance-only 长训：novel L1 从 0.1016
单调恶化到 0.2109，视觉"局部像真人、整体关系不自然"。**禁止从 52k+ checkpoint
续训**。处置：appearance-only 模式加 `appearance_max_steps=2000` 硬保护。

**full-frame 输出契约（07-29）**：废除 `generated*mask + warp*(1-mask)` 硬拼接。
理由：warp 可见区仍有拉伸/双影/色偏，不能当最终像素；任务要求 SD 重新生成整幅。

**v4-fullframe 10k 失败 → v5（07-30）**：real_ref_full 0.099→0.081 但用户肉眼确认
油画感与之前完全一致。根因：**所有旧 checkpoint 的条件模块已收敛进 EG3D 平滑域，
从旧权重初始化无法跳出油画域吸引子**。决策：从零门控适配器重训 300k。
同时 v5 引入等变局部 Reference（real-ref 恒等对应 / novel 3D warp 对齐）、
可靠性软数据一致性（仅高置信核心区 0.85 软投影）、图像梯度误差指标。
此前审计还发现 target-aligned RefNet 特征是"结构存在但功能死链"（训练不可训、
验证不传参、推理无接口）——v5 修复为同一 adapter 的等变训练。

---


### 3.3 阶段③（08-03~08-13）：novel hole 碎孔攻坚战（20+ 轮消融）

从零重训到 ~31k 步后：novel_hole L1 卡死 0.103-0.112（real_ref 0.030），
hole 区肉眼"细碎小黑孔与碎裂纹理"。诊断（代码+行号级）：novel hole 被 EG3D
平滑 latent 监督（成因①）、novel 无 LPIPS/GAN 像素监督（成因②）、hole 内 local
Reference 被 validity=0 硬屏蔽（成因③）、splatting mask 约 659-1017 个连通碎片（触发器）。

**消融全记录（全部从 best_model.pt 独立目录起步，跑完回退）**：

| 消融 | 内容 | 结果 |
|---|---|---|
| A | 移除 novel hole 监督（权重→0） | **崩盘** 0.108→0.27。hole 必须有几何 target |
| B1/B2 | cycle=反投影 SD 输出+LPIPS（直接/SNR 加权） | 不收敛，0.105↔0.118 震荡 |
| G/G2 | cycle=反投影 warp+二次 SD 补洞 | hole L1 降但**油画化**（edge 0.026→0.018 抹平） |
| C | mask 腐蚀/模糊（纯推理） | 治表伤本，用户反对（效果图不像人） |
| J1 | novel 解禁 Reference gate 更新 | 无变化。排除"梯度隔离困住 gate"假说 |
| Plan B | hole 无像素 target，仅 ID+LPIPS | **崩盘** 0.27（同 A）。ID/LPIPS 给不了形状 |
| Plan C | hole target=EG3D 低频+source 高频（拉普拉斯分解） | 不崩不油画（0.111-0.119），但细碎黑孔仍在 |

**关键转折（08-12 §6 复核）**：NOVEL 与 REAL-REF 用**同一张碎裂 mask**，REAL-REF
却能修好 → mask 只是触发器。真正的缺口：**原版 hybrid hole geometry condition 在
diffusion 迁移时被遗漏**——原版生成器 hole 内始终看到 EG3D novel render（逐像素
五官/轮廓草图），diffusion 路径的 `warp.hybrid` 根本没被读取，推理端 hole 只有
黑 warp+碎 mask。

**冻结权重几何条件因果实验（§7，不训练）**：hole 填 lowpass EG3D → hole L1 -22%、
dark ratio -62%（0.089→0.035）；填完整 EG3D 与 lowpass 差异小。**低频几何条件
是主导因果变量**。

**Reference 连续延拓（§9，normalized convolution）证伪**：数学性质全对（常数再现、
置信度衰减），但 disocclusion 边界两侧分属前景/背景/头发不同表面，各向同性高斯
把错误侧外观扩进来；三 identity 方向不一致。**遮挡边界的高频不能靠二维邻域插值**。

**Stage 1-5（low-geometry 条件接入训练后的因果链）**：
- Stage 1（低频几何条件微调 250 步）：hole 0.085→0.039（-54%），但训练带回高频碎孔。
- 2×2 参数交换（§11）：碎孔重现 100% 来自 **Identity/W+ 更新**，非 Appearance。
- Stage 2（频率一致 target）：hole 0.031、碎裂能量 -21%，但单像素暗孔未降。
- Stage 3（确定性 VAE mode 替代 posterior sample）：RNG 严格对齐后逐位等价 →
  **posterior 随机错位被排除**（首次 pilot 因改变 RNG 消耗顺序而混杂作废——对照
  实验必须复现随机轨迹）。
- latent reliability 软场（§14）：distance-field/area 软 mask 无肉眼改善，
  结论后被模块级证据撤回。
- **模块级定位（§15，冻结参数交换）**：不是 mapper、不是 gate，是 W+ RCA 的
  Q/K/V/O projection，其中 **`to_out_wplus` 是最大单项主效应**（dark +76%）。
- Stage 4（冻结 O 训练）：碎孔指标 -7~13%，方向验证成功但幅度不足。
- Stage 5（共享冻结原生 `attn.to_out`）：自动暗孔指标大降（dark -51%）但用户肉眼
  终审**"脏得不行，完全不像同一个人，油画感历史最重"——判定 FAILED**。
- **§17.8 终结论（本项目最重要的一条）**：O 只是故障**载体**；深层根因是
  **W+ 高容量分支被伪 RGB 重建目标驱动，必然学会伪目标外观**——独立 O 时表现为
  碎孔，共享 O 时伪监督压力转移到 Q/K/V 表现为全局油画化+身份崩坏。
  换输出投影不能修复职责错配。唯一解：冻结/严格限制 W+ 分支，重新定义它只提供身份。

### 3.4 阶段④（08-13~08-21）：真实域 cycle 第一性原则与三阶段课程式

**REAL_DOMAIN_CYCLE 方案（08-13）**：承认任务本质——真实 source view 是唯一可信
appearance GT；EG3D 只当"几何传感器"（depth/silhouette/low-pass），不当"画师"；
W+ 只保留身份/低频；高频由 RefNet+BrushNet/SD 先验+真实域 cycle 推断。

**三阶段课程式实施**：
- Stage A（`run_real_reference_pretrain.py`，10k）：零门控从零、只训 Reference
  appearance adapter，real_ref 自重建。稳定（real_ref_hole≈0.027）。
- Stage B（ref_cycle）：**两次发散**——(1) GAN(10)+FM(100) 加在 x0 解码上：
  grad 0.5→50.7、cycle_gan 0.7→3.6、白爆（归档 `_ref_cycle_pretrain_diverged_ganfm`）；
  (2) 去掉 GAN/FM 后仍单调恶化（novel 0.083→0.145）：`geometry_noise_weight=1.0`
  全强度把 EG3D lowpass 域拉进所有 timestep → visible 泛白、人种漂移；RefNet gate
  8000 步仅 0.016（W+ gate 常开 tanh(3)=0.995 挤压 RefNet，而 W+ 是 EG3D 域训的）。
  根本矛盾：hole 内容决策者 BrushNet(618M) 全程冻结，在训模块既无内容来源也无容量。
- Stage C（`run_wplus_identity_pretrain.py`，纯配对 synth 任务 50k-100k）：
  id_score=0.218，是 W+ 冻结权重的来源（`iteration_50000.pt`）。
- Stage D（DUAL_DDIM_CYCLE_512，08-20）：orig 忠实双 DDIM cycle（novel 采样→
  forward warp 回源→二次 DDIM 补洞→全部 orig 损失作用于 pred_inv_warp）。
  显存账 512/40 步 ≈32GB，**24GB 卡无法运行**（这是放弃它转 joint mode 的直接原因）。

---

### 3.5 阶段⑤（08-21~08-23）：joint mode 单 run 与契约 v1→v6 迭代

放弃三阶段课程式（Stage B 两次发散实证 + 24GB 放不下 512 双 DDIM），改为
**joint mode 单 run**（`scripts/run_joint_train.py`）。核心决策表（每条有消融出处）：
BrushNet 解冻全量（8-bit AdamW lr=1e-5）、W+ 全冻结（gate 固定 tanh(3)，权重来自
Stage C 50k）、RefNet adapters 可训（gate 初始化 0→0.5）、SD/VAE/RefNet backbone
冻结（bf16）、cycle 感知损失 GAN/FM=0（x0 残噪发散实证）、real_ref 25% 批次带
pixel_sup。随后 48 小时内契约迭代 6 版：

| 版本 | 核心改动 | 结局 |
|---|---|---|
| v1 | target=condition（visible 监督=碎 warp 本身） | 17K：自重建好但 novel 雾蒙蒙。终诊：模型权重无恙（单步 x0 最清晰），**纯噪声链下 novel 无干净 visible 吸引子 → 高噪声段均值化=雾**；condition 锚定链则玻璃感 |
| v2/v2.1 | **target contract v2**：visible target=源照片逆向 warp（`inverse_warp`，干净重投影，lap 0.028≈源照片）；v2.1 visible 权重 0.025→1.0 | 彩眼睑消退、碎裂略改善；但瞳孔始终不圆、细节粗糙——感知损失全在 256 分辨率，监督侧有锐度天花板 |
| v3/v3.1 | hole 条件改纯 depth（ControlNet-depth 范式，零颜色） | **证伪**：hole 卡死 0.34-0.37，hole 内≈弱条件生成→收敛到 SD1.5 无条件先验均值=**蓝灰色块**。hole 需要颜色基底。意外收获：real_ref 0.0256 历史最好 → 牵出 timestep bug（见 §6-7） |
| v3.2 | lowpass hole 条件 + 全 timestep 修复 | 600 步即中止（分析指向条件侧而非深度侧） |
| v3.3 | **orig hybrid 回归**：条件 hole=完整 EG3D 渲染（清晰轮廓+颜色），目标 hole=lowpass@0.1（防塑料感）。step-0 基线 0.0498（历史最好起点） | 600 步即中止——用户对方法论的根本否定（见下） |
| v4 | **用户方案"诚实条件"**：hole 填噪声（诚实声明无信息）+ 第 6 通道全图 depth + hole 像素监督=0 + cycle@512 主监督 | 演化为 Phase-1/2 插件阶梯（见下） |

**用户对 v3.x 低通条件的方法论否定（原话精神）**：低通条件=“手搓面糊糊再让网络
优化边缘”——内容是手工的，网络从未学过生成；v3.3 把糊糊换成塑料同属此类。
佐证：real_ref 的 hole 从来是无内容黑块，网络照样补到 0.026——**失败在监督
（弱/伪）而非条件缺内容**。

**Phase-1（08-22 晚）**：SD UNet 100% 原生冻结（RCA 全拆）、BrushNet 618M 唯一
可训件、条件 6ch=[noisy4|masked图3|mask1|depth1]、Task A/B 50/50、hole 像素
监督=0、cycle@512 主监督。修复两个 bug（纯模式 trainable_parameters 为空导致
优化器未构建；validate 无 ref_feat KeyError）。

**Phase-2（08-23）RefNet 融合插件**：触发证据=Phase-1 25K 自重建 hole 0.099 vs
历史 0.026（RefNet 对齐路缺失）。`models/refnet_fusion.py` 4 尺度零初始化 1×1 conv
（3.79M）把对齐后源照片特征加进 BrushNet 残差流，从 Phase-1 25K 热插拔。
- **REAL 面板雾蒙蒙事故（08-23 下午）**：real_ref 指标翻倍恶化全图蒙纱。三因叠加：
  ①验证采样器漏接插件（训练见过"BrushNet+插件"，验证凭空抽掉 → 仪表盘坏，训练健康）；
  ②Task A 的恒等对齐特征=GT 本身，插件 1K 步学出直通捷径（loss 坍缩 500 倍），
  BrushNet 通路躺平；③save_interval=25000 中途崩溃全丢。修复：采样器镜像训练路径、
  融合插件只对 Task B 生效、save_interval 默认 2000。代价 5K 步作废。
- **Phase-2 证伪（08-23 晚）**：25K 步零指标变化 + **架构级不可能**——
  `Splatting.forward_warp` 的 hole 区域输出恒为零，对齐后 RefNet 特征在 hole 内
  本来就是零，插件在 hole 里只能输出常数，**结构上无法给 hole 送源内容**；
  scale3 权重严格为零（梯度从未流动）。
- **重大发现——depth 通道萎缩**：conv_in_condition 逐通道范数，depth 仅 0.09-0.12
  （其他条件 3.5+，弱 10-30 倍），25K 步几乎不增长。因果消融（同种子抹零 depth vs
  换种子）：抹 depth 影响=换种子的 1/7-1/10，**模型事实上没在读 depth**。
  机制：Task B 的 hole target 是高斯低通（模糊），跟着 depth 画锐边反而增大 MSE
  ——监督在主动惩罚 hole 内锐利结构。

**v5（edge target，08-23 深夜）**：hole target=lowpass(EG3D)+2.0×(depth 带符号
高通)，探针标定 hole 边缘能量≈源照片量级（guided filter 方案先被探针否决）。
**证伪（52K）**：depth 通道范数仅 0.1159（要求 >>0.5）；决定性对照——v5（target
带边）与 Phase-2（target 无边）的 depth 范数增长曲线**完全相同**（0.0899→0.116
vs 0.117）→ target 里的边对 depth 采用率边际贡献为零。两个探针排除了"VAE 瓶颈"
（latent-lap 0.355 vs 0.157，边在 latent 真实存在）和"边不在 target 里"。

**v6（edge band，08-24 记录）——真根因发现**：审计发现 v5 的根因分析也是错的——
读代码发现 `geometry_noise_weight=0.0`（v4 起如此），**Task B hole 的 ε-MSE 激励
是精确的 0**，v5 换的 target 从未进入 loss（唯一作用是低 t 时当泄漏旁路）！两个
run 收敛到同一终点因为都在"hole 零监督"下训练。v6 实施 `training.edge_band`：
从 novel depth latent 梯度取 top-10% 膨胀 2px ∧ hole mask 作监督带，带内
MSE@10+snr_x0@2，洞内其余保持 0 监督。冒烟梯度 0.7→37（激励从 0 到主导）。
**v6 实跑结局（docs 未记录，本次核查日志）**：从 Phase-1 25K 续训至 96K，
novel_hole 0.150→0.107、dark_ratio 0.0333→0.0069（-79%）、novel_full
0.091→0.079，确有系统性改善，但绝对水平仍不足（对照 v3.3 起点 0.0498、
历史 real_ref 0.026），于 08-23 09:13 归档为 `_joint_train_v6_edgeband_archived`。
随后 v7（`_joint_train_v7_orig`，09:33）从 v6 96K checkpoint 续训 2K 步即被终止
（同样水平）。

### 3.6 当前（08-23 09:52~）：`_joint_train_orig` 从零整合 run

放弃全部续训链，从零启动整合 run（配置见 §2.4）：hole 条件回到 orig hybrid
（完整渲染）、hole 目标恢复监督（geometry_noise/x0=1.0，吸收 v5/v6 零监督教训）、
visible 目标=inv_warp（v2 契约）、depth 第 6 通道、cycle@512、GAN/FM=0、
RefNet 对齐特征路开启、synthetic 配对数据弃用。**这是本项目第一次把"条件有内容、
目标有监督、W+ 冻结、BrushNet 可训、全 timestep"同时成立的 run。**

---


## 4. ⛔ 已证伪方向清单（动手前必查，禁止无新证据重试）

| # | 方向 | 证伪证据 | 出处 |
|---|---|---|---|
| 1 | 移除 novel hole 几何监督 | novel_hole 0.108→0.27 崩盘（两次独立复现：消融 A、Plan B） | 阶段③ |
| 2 | 用 ID+LPIPS 替代 hole 像素/几何 target | 同上崩盘——ID 管身份、LPIPS 管感知，都不告诉 SD"hole 该长什么形状" | 阶段③ Plan B |
| 3 | 纯 LPIPS cycle（直接/SNR 加权/反投影 warp+二次补洞） | 不收敛或引入油画感 | 阶段③ B/G |
| 4 | mask 像素级腐蚀/模糊 | 治表伤本，效果图不像人，用户明确反对 | 阶段③ C |
| 5 | mask 形态学闭运算（close3） | 无值得承担 mask 面积膨胀副作用的收益 | 阶段③ §7 |
| 6 | Reference 特征 normalized convolution 连续延拓 | 遮挡边界非同一表面，各向同性核扩入错误外观；三 identity 方向不一致 | 阶段③ §9 |
| 7 | target/condition VAE posterior 随机性 | RNG 严格对齐后逐位等价 | 阶段③ Stage 3 |
| 8 | latent 空间软 reliability 场（distance/area 软 mask） | 无肉眼改善，结论被模块级证据撤回 | 阶段③ §14 |
| 9 | 更换 W+ RCA 输出投影修碎孔 | 冻结 O：-7~13% 不足；共享 O：自动指标大降但肉眼油画化+身份崩坏（FAILED） | 阶段③ §16-17 |
| 10 | **让 W+ 分支承担 novel 伪目标重建（任何高容量形式）** | 独立 O→碎孔；共享 O→油画+身份崩坏。伪监督压力只会换出口 | 阶段③ §17.8 |
| 11 | mirror-ref（水平翻转）跨视角监督 | 翻转≠镜像视角，几何标签噪声 | 阶段② v2 |
| 12 | 从携带 EG3D 域的旧 checkpoint 初始化 | v4-fullframe 10k：指标改善、油画感纹丝不动（权重已固化进油画域吸引子） | 阶段② |
| 13 | appearance-only 长训（300k 误配） | 52k 恐怖谷：novel L1 0.10→0.21 单调恶化 | 阶段② |
| 14 | GAN/FM 加在 x0 单步解码上（cycle 内） | grad ×100、白爆、验证单调恶化，两次复现 | 阶段④ Stage B |
|     | 【2026-08-31 勘误（全量复盘）】：本行"两次复现"与 §3.4 原文矛盾——§3.4 记载的两次发散中，
|     | 第二次（novel 0.083→0.145 单调恶化）发生在**去掉 GAN/FM 之后**，元凶是 #15 的
|     | geometry_noise_weight（EG3D lowpass 全 timestep）。GAN/FM 自身事故 = **1 次白爆**，
|     | 且当时 BrushNet(618M 内容决策者) 全程冻结（§3.4"在训模块既无内容来源也无容量"）、
|     | geometry_noise 毒配置并行——三个混淆因素在 joint mode 后均已消除。本行不应再作为
|     | "GAN/FM 不可行"的充分证据引用。 |
| 15 | `geometry_noise_weight=1.0`（全 timestep 全强度 EG3D lowpass 目标） | visible 泛白、人种漂移、novel 0.083→0.145 单调恶化 | 阶段④ Stage B |
| 16 | RefNet gate 零初始化指望涨起来 | 8000 步仅 0.016；低通目标下 RefNet 特征降不了 loss（后改 0→0.5 初始化） | 阶段④ |
| 17 | 512 分辨率双 DDIM 全链 cycle（24GB 卡） | 显存账 ~32GB，物理不可行（A6000 48GB 可跑） | 阶段④ Stage D |
| 18 | hole 条件=纯 depth（零颜色） | hole 卡死 0.34-0.37，收敛到 SD 无条件先验蓝灰色均值。hole 需要颜色基底 | 阶段⑤ v3.1 |
| 19 | novel target 的 visible 区=splat warp 本身 | 碎裂图案被忠实学习（玻璃感）；高噪声链无干净吸引子→雾 | 阶段⑤ v1 |
| 20 | hole 零像素监督（指望 depth 条件+cycle 自发学几何） | depth 通道范数 0.09-0.12 几乎不涨；因果消融抹 depth=换种子 1/7 影响；target 带边也无用（激励精确为 0） | 阶段⑤ v4/v5 |
| 21 | RefNet 特征 warp 后在 hole 内注入源内容 | 架构级不可能：splat 的 hole 输出恒零，warp 对齐特征在 hole 内=零，插件只能加常数 | 阶段⑤ Phase-2 |
| 22 | 用大量工程性 mask 补丁/closing/输出端粘贴修碎孔 | 用户明确要求第一性原理优先，多项已实测无效 | 阶段③ §8 |

**长期有效但也多次被误读的边界**：
- EG3D 当 **loss 目标**有毒（塑料感）≠ 当**输入条件**有毒——orig 从第一天就用完整
  渲染填 hole（条件端），域污染只发生在目标端（阶段⑤ v3.3 复盘的关键纠偏）。
- 感知/对抗损失放在 **50 步采样链**上不可行（显存），放在 **x0 单步解码**上会发散
  （残噪）——目前唯一安全位置是 real_ref 低 t 干净解码 + cycle 回投图。

---

## 5. 已确立的结论与设计原则

### 5.1 监督契约（多轮血泪换来）

1. **hole 必须有几何/颜色 target**（EG3D lowpass 或 rgb），权重适中（历史稳定值
   noise 0.1~1.0 需按 run 观察）；完全移除=崩盘。
2. **novel visible 的干净 target = `Warper.inverse_warp`(源照片)**（grid_sample，
   每像素恰好采样一次，内部 lap 0.028≈源照片 0.037）。绝不能用 splat warp 本身
  （碎裂）或 EG3D（域污染）当 visible target。
3. **真实感监督的唯一安全来源**：cycle 回投真实源照片（L1 10+LPIPS 10+PL 30+ID 0.5
   @512, t<200, x0_clip 3.0, GAN/FM=0）+ real_ref pixel_sup（LPIPS+PatchGAN+R1）。
4. **训练 timestep 必须全范围 [0,1000)**：只训 t<200 会让生产采样的高噪声段行为
   漂移到条件均值=雾蒙蒙（v1→v2.1 的结构性根源，v3.1 修复后 real_ref 0.0256）。
5. cycle loss 内部自带 t<200 自门控，外层不要再钳制 timestep（曾是事故）。

### 5.2 参数职责路由

| 模块 | 唯一职责 | 训练策略 |
|---|---|---|
| W+ mapper+RCA | 身份/低频 | **冻结**（权重 Stage C `iteration_50000.pt`，id_score 0.218）。任何让伪目标梯度进 W+ 的配置都会学坏 |
| BrushNet 618M | hole 内容+条件翻译 | 解冻全量，8-bit AdamW lr=1e-5，grad checkpointing |
| RefNet adapters | 真实外观使用方式 | 可训 lr=5e-6，gate 0→0.5 初始化，K/V 热启动自 SD attn 权重 |
| RefNet/SD/VAE backbone | — | 冻结 bf16 |
| PatchGAN | real_ref 真实感 | 仅 real_ref 低 t |

### 5.3 方法论（防幻觉操作规程）

1. **单变量**：一次只改一个东西；短程 250-step 探针 → 肉眼裁决 → 再放大。
2. **冻结因果消融**：不训练，同 batch/seed/采样器，交换参数或条件定位根因
   （本项目最有效的手段，产出 §15/§17.8/depth 萎缩三大发现）。
3. **预注册判据**：启动前写死"多少步达到什么数值否则证伪回退"，不带病长跑。
4. **RNG 轨迹控制**：对照实验必须复现编码次数/sample 次数/消耗顺序（Stage 3 教训）。
5. **肉眼终审 > 自动指标**：dark ratio/pinhole/L1 多次与感知质量脱节甚至反向
  （Stage 5、v4-fullframe）；验证采样器必须镜像训练路径（Phase-2 雾蒙蒙教训）。
6. **配置必须验证生效**：读代码确认 loss 项真的非零、权重真的被读（v5 事故：
   target 换了但权重是 0，白跑 52K）。启动前打印有效配置+首步 loss 构成。
7. **实验目录纪律**：独立目录、跑完回退代码、`WARP_GAN_OVERWRITE=1` 才覆盖；
   save_interval ≤2000。

---


## 6. ⚠️ AI 幻觉与分析错误事故录（本项目最痛的一章，新 agent 引以为戒）

用户的原话总结："项目快四个月，每一轮都在 AI 幻觉中度过，没什么实质进展。"
以下每一条都实际消耗了数小时到数天 GPU 时间，且**多数在事后才被发现**：

| # | 事故 | 浪费 | 教训 |
|---|---|---|---|
| 1 | 目标视角错配（target=源视角）未被早期发现 | 0-74k 步全无效 | loss 在降≠任务自洽；先肉眼检查 target 本身 |
| 2 | 52k appearance-only 误配置（warm-up 配成 300k） | 54k 步 | 长跑前逐项核对 mode/max_steps 语义 |
| 3 | "碎裂 mask 是主因"的过强结论 | 数轮消融 | 同 mask 下 REAL-REF 能修好→mask 只是触发器。下因果结论前先找反例 |
| 4 | Stage 3 首次 pilot 改变 RNG 消耗顺序 | 一次实验作废 | 对照实验要保随机轨迹 |
| 5 | §14 area-binary"有效"结论被撤回 | 一轮 | 自动指标与视觉脱节时不要急着下结论 |
| 6 | "共享 O 是候选修复"建议被用户肉眼一票否决 | Stage 5 整轮 | 指标好≠好，先出图给人看 |
| 7 | 训练 timestep 钳制 t<200 残留 | v1→v2.1 症状全部被误诊 | v3.1 才发现。**v2.1 时代的全部症状（雾、瞳孔不圆、粗糙）是在未训练的高噪声段 bug 下观察的，相关结论须在修复后重新评判** |
| 8 | Phase-2 验证采样器漏接插件 | "雾蒙蒙"误诊为训练崩溃，5K 步作废 | 训练/验证路径必须镜像；先检查"仪表盘坏了"再怀疑"模型坏了" |
| 9 | Task A 恒等特征=GT 泄漏捷径 | 同上 | 给已有完美监督的任务喂额外条件=制造捷径 |
| 10 | **v5 根因分析本身是幻觉**："激励稀释~1-2%"——实际 `geometry_noise_weight=0.0`，激励是**精确的 0**，换的 target 根本没进 loss | 52K 步+一整轮分析 | 分析权重/梯度必须读代码+打日志验证；两个 run 同终点→先 diff 有效配置，别发明新机制 |
| 11 | `warp.hybrid: True` 写在配置里但 diffusion 路径从未读取 | 数周无人发现 | 配置项≠生效路径，逐行追数据流 |
| 12 | mask erode/blur 配置写了但 Warper() 默认值从未传入 | — | 同上 |
| 13 | 多轮"指标改善→肉眼不变/更差"（v4-fullframe、Stage 5、Phase-2） | 累计数万步 | L1/edge/dark 与感知质量弱相关；每轮都要有肉眼对照产物 |

**给新 agent 的硬性要求**：
- 任何"根因分析"必须给出**代码行号 + 可复现实测数据**，且优先考虑"我读漏了
  某个开关/权重/路径"这类平凡解释，再考虑深层理论；
- 两个实验收敛到同一终点 → 第一反应应是"它们的目标函数/有效配置其实相同"；
- 写入文档的每个结论标注"已证实（出处）/推测（待验证）"。

---

## 7. 代码地图与运行手册

### 7.1 关键文件

| 文件 | 作用 |
|---|---|
| `training/coach_inpainting_static.py` | 训练主循环、损失、验证（约 2252 行 diff vs 原版） |
| `scripts/run_joint_train.py` | **当前主入口**（env: `WARP_GAN_MAX_STEPS` / `WARP_GAN_EXP_DIR` / `WARP_GAN_RESUME_CHECKPOINT` / `WARP_GAN_SAVE_INTERVAL` / `WARP_GAN_OVERWRITE`） |
| `models/referencenet/attention_processor.py` | W+ RCA + RefNet 门控注意力处理器 |
| `models/mapper/w_proj.py` | W+ [14,512] → SD token [18,768] |
| `models/refnet_fusion.py` | Phase-2 融合插件（已证伪，保留未启用） |
| `utils/diffusion_inpainting.py` | 共享 BrushNet/DPM++ 采样（训练验证与推理共用） |
| `models/diffusion_inpaintor.py` | 生产级推理封装，加载 checkpoint |
| `datasets/dataset_inpainting_static.py` | 数据集（`fixed_novel_view` 支持） |
| `scripts/infer.py` | 双后端推理（diffusion / legacy_lama），输出 SD full-frame |
| `models/BrushNet-main/` | BrushNet 源码（conditioning_channels=5/6） |
| `warpgan_orig/` | 原版对照（cal_inpaintor_loss L1054-1074、rec loss L1118-1212） |

### 7.2 常用命令（从仓库根目录，conda env `warpgan`）

```bash
# 当前主训练（从零 300K）
CUDA_VISIBLE_DEVICES=0 python scripts/run_joint_train.py
# 冒烟（2 步）
WARP_GAN_MAX_STEPS=2 CUDA_VISIBLE_DEVICES=0 python scripts/run_joint_train.py
# 中间量可视化
python scripts/eval_phase2_visualization.py
# 正式推理（先改 configs/infer.yaml 的 ckpt_inpaintor）
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py
```

### 7.3 显存约束（24GB 是本项目硬边界）

- joint mode @512, bs=1, x0 单步 cycle ≈15-17GB（8-bit AdamW + grad ckpt）✓
- 512 双 DDIM 全链 ≈32GB ✗（需 A6000 48GB / A100）
- 历史峰值参照：v4 系 4-step 周期含验证 17.98GB；cycle@512 23.9GB（贴边）
- OOM 处理先例：原生 32×32 latent patch（+4px 上下文）替代整帧解码做像素监督

### 7.4 验证产物怎么看

- `experiments/<run>/logs/images/val/overview_step_*.png`：每身份 NOVEL 行
  （source→EG3D target→warp→SD 输出）+ REAL-REF 行；
- `experiments/<run>/logs/val_metrics.txt`：full/hole/visible L1、edge、dark、cycle；
- `debug_tensor_health.txt`：gate/通道范数健康度（盯 depth 通道范数是否 >0.2 上升）；
- 肉眼对照产物永远优先于 metrics（§6-6 教训）。

---


### 7.5 checkpoint 与目录索引（考古用）

| 目录 | 内容 |
|---|---|
| `experiments/_joint_train_orig/` | **当前 run**（从零 300K） |
| `experiments/_joint_train_v6_edgeband_archived/` | v6 边带监督（96K，novel_hole 0.107 / dark -79% 后归档） |
| `experiments/_joint_train_v7_orig/` | v6 续训 2K（98K，即终止） |
| `experiments/_joint_train_phase1/` | Phase-1 纯 BrushNet 基线 |
| `experiments/_joint_train_phase2/` | Phase-2 融合插件证伪证据（50K） |
| `experiments/_joint_train_phase3/` | v5 edge target 证伪证据 |
| `experiments/_joint_train_v1_target_eq_condition_20260821/` | v1 雾蒙蒙归档 |
| `experiments/_joint_train/` | v2.1 归档（15K） |
| `experiments/_v31_depth_falsified_67k/` | v3.1 蓝灰色证伪 |
| `experiments/_wplus_identity_pretrain/` | Stage C：W+ 冻结权重来源（iteration_50000.pt, id 0.218） |
| `experiments/_real_reference_pretrain/` | Stage A（real_ref_hole≈0.027 先例） |
| `experiments/_ref_cycle_pretrain/` | Stage B 两次发散证据 |
| `experiments/_brushnet_cycle_512_dual/` | 512 双 DDIM（24GB 不可行证据） |
| `experiments/_ablation_*/`、`_geometry_condition_causality/`、`_latent_reliability_causality/`、`_freeze_wplus_output_*`、`_shared_wplus_output_*`、`_identity_submodule_*`、`_wplus_rca_*`、`_depth_ablation/`、`_haze_diagnosis*/`、`_phase2_viz/`、`_low_geometry_*`、`_frequency_consistent_*`、`_deterministic_low_geometry_*`、`_reference_extension_causality/` | 阶段③⑤全部消融证据（metrics.json / overview.png / run.log 齐） |
| 历史 `experiments/train_inpaintor/[2026xxxx]*` | v2/v3/v4 系（v3 300k 基线、52k 恐怖谷等） |

---

## 8. 悬而未决的问题（新 agent 的工作清单）

1. **当前 run 的裁决**（预注册判据，沿用 v6 风格）：
   - 5K：hole 结构成形（novel_hole 显著低于 step-0 且不蓝灰/不崩）；
   - 10-15K：depth 通道范数 >0.2 且上升（v5/v6 的未竟目标）；novel_edge 低于
     Phase-1 区间（0.022-0.028）；
   - 25K：肉眼裁决质感（对照 `_real_reference_pretrain` 的 real_ref 0.027 水平）；
   - 任一失败 → 回到本文档追加分析，不带病长跑。
2. **极端视角"脸颊顺 hole 无限延长"**：监督盲区（cycle_valid≈0.73 区域无信号 +
   Task A 大洞样本稀少），depth 通道是盲区唯一真信息——已分析未修复。
3. **感知损失的分辨率天花板**：cycle 感知损失在 256 时瞳孔只有 ~4px；
   备选：cycle 感知损失 256→512（显存余量约 5GB）或 FM@t<50。
4. **novel 分支 hole 真实感的结构性缺口**：real_ref 有三重监督（真 GT+恒等对齐+
   LPIPS/GAN），novel hole 永远没有。当前答案是"orig 范式"（条件给足+cycle 翻译），
   若 `_joint_train_orig` 仍不达标，下一个第一性候选是 **v6 边缘带监督机制**
   （代码保留，enable 即用）与 orig 权重的混合。
5. 长期方向（历史文档提出、从未实施）：Relative Camera Encoder（hole 大时 UNet
   不知道该出左脸还是右脸，4-8 个 camera token 即可）、多视角真实数据、
   W+ 以外的身份注入形式（原版是 FFC AdaIN，非空间残差写入器）。

---

## 附：合并前的原始文档清单（已移入 docs/archive/）

| 文档 | 日期 | 主要内容 |
|---|---|---|
| PROJECT_SUMMARY.md | 07-21~07-30 | v2-v5 总览、full-frame 契约、OOM 修复 |
| TRAINING_AUDIT_REPORT.md | 07-15~07-28 | v2/v3/v4 逐版审计、52k 事故、模块分类 |
| DIFFUSION_SVINET_EVOLUTION_REPORT.md | 07-28 定稿 | 原版详解、修改时间线、DDIM 专项、油画感证据链、v4 方案 |
| CODE_MODIFICATION_REPORT.md | 08-09 | factorized 双分支代码溯源、novel 监督缺口诊断 |
| NOVEL_HOLE_ABLATION_LOG.md | 08-09~08-13 | 碎孔全部消融 + Stage1-5 + §17.8 终结论（最厚的一份） |
| METHOD_INTRODUCTION.md | 07-30~08-13 | 方法说明（对外口径） |
| REAL_DOMAIN_CYCLE_GEOMETRY_PRIOR_PLAN.md | 08-13 | 真实域 cycle 第一性原则、Stage A/B 命令 |
| DUAL_DDIM_CYCLE_512.md | 08-20 | 512 双 DDIM cycle 设计与显存账 |
| ARCHITECTURE.md | 08-22（v3.1） | 逐模块架构审计、timestep bug、运行配置 |
| JOINT_TRAINING_PLAN.md | 08-21~08-24 | joint mode 决策表、v1→v6 契约迭代、Phase-1/2 |

（根目录 `GROUP_MEETING_REPORT.md` 为 08-03 组会汇报文稿，不属于 docs/，未纳入合并。）

