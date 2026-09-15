# 联合训练方案（Joint Mode）：W+ 冻结身份 + BrushNet/RefNet 联合 + x0 cycle

> 2026-08-21 定稿。本文档是三阶段课程式训练（wplus_identity → ref_cycle → brushnet_cycle）
> 的替代方案：**单 run 联合训练**。每个决策都标注了出处（消融日志章节或实测），
> 不引入任何未经验证的新机制。

## 1. 为什么放弃三阶段课程式

1. **Stage 2 (ref_cycle) 两次发散的实证**：
   - GAN(10)+FM(100) 加在 x0 解码上 → grad 0.5→50.7、cycle_gan 0.7→3.6、验证单调恶化、白爆（已归档 `_ref_cycle_pretrain_diverged_ganfm_20260821`）
   - 去掉 GAN/FM 后仍单调恶化（novel 0.083→0.145）：`geometry_noise_weight=1.0` 全强度
     把 EG3D lowpass 域拉进所有 timestep → visible 泛白、人种漂移
   - RefNet gate 8000 步仅 0.016：低通目标下 RefNet 特征无法降 loss，gate 涨不动；
     W+ gate=tanh(3)≈0.995 常开挤压 RefNet（W+ 是 EG3D 域训的）→ hole 生成无信息
2. **根本矛盾**：hole 内容的真正决策者是 BrushNet（618M），但整个 Stage 2 它被冻结；
     在训的模块（RefNet adapters, gate≈0）既无内容条件来源也无足够容量。
3. **用户目标**：24GB 内联合训练，忠实 orig，最小增量。

## 2. 决策表（每条有出处）

| 决策 | 值 | 出处 |
|---|---|---|
| BrushNet 条件 mode | `lowpass_rgb` 不变 | 消融日志 §7（黑孔-62%/holeL1-22%）、§10 生产 contract |
| NOVEL target | `low_geometry`（频率一致 composite） | §12 Stage 2 实证：高频碎裂 -21%，无塌陷 |
| novel noise MSE 权重 | `geometry_noise_weight=0.1` | 消融 A：→0 崩盘（0.108→0.27）不能移除；=1.0 是本次泛白/恶化实证；0.1 为原始稳定值 |
| novel 外观监督 | x0 cycle t<200: L1(10)+ResNet_PL(30)+ID(0.5) | orig cal_inpaintor_loss 权重；x0 单步=orig FFC 1-step 的忠实翻译；GAN/FM=0（x0 残噪发散实证） |
| x0 clip | 3.0（VAE 潜空间保险丝） | 发散 run 的白爆实证 |
| real_ref 自重建 | 25% 批次，noise MSE 1.0 + pixel_sup(LPIPS+GAN+PL, hard_low_noise) | appearance_pretrain 稳定先例（real_ref_hole≈0.027）；消融日志 §15.1"SD+BrushNet+Reference 能工作" |
| W+ 分支 | **冻结**（mapper + RCA 全部，gate fixed tanh(3)） | §17.8："只要 NOVEL 伪图像重建 loss 直接训练高容量 W+ 分支，它就会学伪目标外观"（独立 O→碎孔；共享 O→油画化身份崩坏）。冻结是最严格执行。权重来自 Stage 1 `iteration_50000.pt`（id_score=0.218，纯配对 synth 任务无伪目标问题） |
| BrushNet | **解冻全量**，8-bit AdamW lr=1e-5，grad checkpointing | orig train_brushnet.py 全量训练形式；BrushNet 官方微调=masked 条件+MSE，与本方案 novel 伪配对形式一致；24GB 显存账（见 §3） |
| RefNet adapters | 可训 lr=5e-6，**gate 初始化 0→0.5** | gate 零初始化是 Stage 2 涨不动的实证根因；adapter K/V 热启动自 attn 权重（有效投影），打开安全 |
| RefNet backbone / SD / VAE | 冻结（bf16） | 一贯不变 |
| 判别器 | real_ref pixel_sup 内 PatchGAN（低 t 干净解码） | appearance_pretrain 稳定先例；cycle x0 上禁用（发散实证） |

## 3. 显存账（24GB @ 512, bs=1, x0 单步 cycle）

```text
冻结 bf16: SD UNet 1.7 + RefNet 1.7 + VAE 0.2            ≈ 3.6 GB
BrushNet 可训: 权重 fp32 2.5 + 梯度 2.5 + 8bit AdamW 1.2  ≈ 6.2 GB
RefNet adapters + gates fp32                             ≈ 0.3 GB
激活（bf16/fp32 autocast + grad ckpt, UNet+BrushNet+VAE） ≈ 3-5 GB
cycle 分支（decode 256 + warp + ResNet_PL/L1/ID）        ≈ 1.5 GB
合计 ≈ 15-17 GB < 24 GB ✓（smoke test 实测为准，峰值须 <22GB）
```

时间换空间：gradient checkpointing（BrushNet 原生 `_supports_gradient_checkpointing=True`）。

## 4. 与 orig 的对应关系（唯一增量 = inpaintor 换 diffusion）

| orig (FFC CNN) | 本方案 |
|---|---|
| hole 填 EG3D 渲染 (hybrid 条件) | lowpass(EG3D) 填 hole（§7 因果实测更优） |
| 1-step 前向生成 novel | BrushNet+SD 单步 x0 预测 |
| cycle 回源视角 L1+GAN+FM+PL+ID | 同左但 GAN/FM 置 0（x0 残噪发散），PL(30) 保留 |
| W+ style 注入 (FFC AdaIN) | W+ attention（Stage 1 配对训练后冻结） |
| 全模块联合训练 | BrushNet+RefNet adapters 联合（W+ 冻结，显存+防伪目标） |
| RefNet | 无对应（本项目增量，REAL-REF 已验证其价值） |

## 5. 监控与止损线

- smoke：2 步通过 + 峰值显存 <22GB + loss 构成正确（cycle_l1/resnet_pl 非零、W+ 无梯度）
- 2K 步：验证不得差于 step-0 基线（novel_full≈0.083, real_ref≈0.134）
- 5K 步：Ref gate 均值应 >0.1（对比 Stage 2 的 0.016）；monitor_lpips 应单调降
- 10K 步：cycle_lpips 趋势向下、肉眼无泛白/人种漂移
- 任一止损线失败：停止并回到本文档追加分析，不带病长跑

## 5.1 2026-08-21 补充：v1 终诊与 target contract v2

v1（target=condition，跑到 17K）：自重建良好，但 novel 输出雾蒙蒙；冻结消融
（`scripts/eval_haze_diagnosis.py`，产物 `experiments/_haze_diagnosis/`）证明：

1. **模型权重无恙**：单步 x0 直接输出是所有采样方式里最清晰的（visible lap 0.094
   vs 纯噪声 50 步 0.056），雾不在权重里；
2. **纯噪声链收敛到均值**（H2 步数/采样器被否定：100 步=50 步；随机采样更糊）；
3. **condition 锚定链（SDEdit t250）清晰但呈"玻璃拼凑"**——放大即见大量伪影。

根因（`novel_hole_target=low_geometry` → `batch['target']=condition_img`）：

```text
novel 分支 MSE 目标的 visible 区 = 前向 splat warp 本身
  实测：condition visible lap=0.0639 vs 源照片 0.0333（多出的≈条纹/边界伪影能量）
  → 纯噪声链：novel 分支不存在任何干净 visible 吸引子 → 高噪声段均值化 → 雾
  → condition 锚定链：模型忠实输出被监督教会的碎裂图案 → 玻璃感
```

**target contract v2（`novel_hole_target=inv_warp_visible`）**：novel 分支 target 的
visible 区 = 源照片的逆向 warp（`Warper.inverse_warp`，grid_sample，每像素恰好采样
一次，几何约定与 forward_warp 逐位一致）。实测：内部干净（lap 0.0281 vs 源 0.0368，
旧 splat 0.0264）、回投源视角 L1=0.0161（模型输出~0.06）、hole 精确保持 lowpass。
这是真实照片域的精确重投影——不是 EG3D（域污染教训不复发）、不是 splat（条纹教训
不复发）。历史注记：§12 Stage 2 的频率一致原则只覆盖了 hole 的 EG3D 域问题，
没意识到 visible 区的"condition"本身就是碎图。

v2 从 Stage-1 锚点全新 50K（用户选定，干净归因）。LPIPS(10) 保留（v1 10K→11K
已单独验证有效）。旧 run 归档于 `experiments/_joint_train_v1_target_eq_condition_20260821/`。

## 5.2 2026-08-22：v2.1 结果与 v3（depth 条件，ControlNet 范式）

v2.1（visible 监督 0.025→1.0，10K→15K）：眼部色度三 identity 全部回落到源照片水平
（彩眼睑数值面消退），碎裂感肉眼略改善但未消除。用户关键观察：**瞳孔始终不圆、
细节粗糙**。定性结论：hole 的 MSE 目标是 lowpass（模糊监督训出模糊），感知损失全在
256 分辨率（瞳孔 ~4px 梯度趋零）——细节锐度存在监督侧天花板，等不出来。

**v3（用户决策：从 Stage-1 锚全新 50K，不续训）**：hole 条件改为归一化逆深度图
（ControlNet-depth 约定，近亮远暗，鲁棒 2-98 分位归一化，3 通道；BrushNet 与
ControlNet 同范式，控制分支天生翻译 OOD 条件分布）。监督侧不变：hole MSE→lowpass
EG3D @0.1 + cycle，visible→inv_warp @1.0（v2.1 权重保留）。这正是 ControlNet 范式：
depth 进、RGB 出；无歧义几何，外观由 Reference/W+ 与 target 侧损失供给。

- step-0 基线 novel_hole=0.466（模型从未见过 hole=depth，初期照抄灰度图属预期），
  v3 的任务就是学会 depth→RGB 翻译
- 若 hole 颜色漂移（depth 无颜色信息）：退路 = depth+lowpass 混合填充
- 若细节仍粗糙：下一刀 = 感知损失分辨率 256→512（显存余量 ~5GB 足够）
- 历史注记：orig 用完整 RGB 渲染填 hole 也成功——"条件带外观"非毒药；lowpass 的
  歧义外观才是。depth 是条件端最干净的几何表达。

产物：`experiments/_joint_train_v3_depth/`，日志 `train_logs/joint_train_v3.log`。
v2.1 归档：`experiments/_joint_train/`（15K checkpoint 保留）。

## 5.3 2026-08-22：v3.1（纯 depth 填 hole）67K 步裁决——证伪，回退 v3.2

v3.1 数据（`experiments/_v31_depth_falsified_67k/`）：

| 指标 | v3.1 (depth hole) | v2.1 (lowpass hole) |
|---|---:|---:|
| novel_hole | **卡死 0.34-0.37**（67K 步，10K 后平台） | ~0.10 |
| novel_visible | 0.10（健康，不受影响） | 0.10 |
| real_ref_full | **0.0256（历史最好）** | 0.044 |

**蓝色纹理机理**（用户肉眼确认）：纯 depth 携带零颜色，而 hole 的其余颜色源全部
缺席（RefNet 局部路 validity=0、全局路只带风格、hole MSE@0.1 太弱）→ hole 内生成
≈弱条件生成 → 收敛到 SD 1.5 无条件先验均值 = 著名的蓝灰色调。孔洞为蓝灰色块与
边界的相互作用。

**裁决**：hole 需要颜色基底——orig 用完整 EG3D 渲染填 hole 早已说明；lowpass 是
频率安全折中（§7）；纯 depth 不行。depth 思路正确的一半（几何无歧义）由 visible
区的真实 warp 承担。

**v3.1 的意外收获**：全 timestep 修复被验证有效（real_ref 0.0256 历史最好）。
v2.1 时代的全部症状（雾、瞳孔不圆、粗糙）是在"高噪声段从未训练"bug 下观察的，
须在修复后的世界重新评判。

**v3.2 = v2.1 lowpass hole 条件 + 全 timestep 修复 + 300K/25K 存档**，
`experiments/_joint_train_v32/`。step-0 基线 novel_hole=0.059（depth 版为 0.466）。
若细节仍粗：备选①cycle 感知损失 256→512；②FM@t<50（orig 纹理压力）。

## 5.4 2026-08-22：v3.2 中止 → v3.3（orig hybrid 回归：hole 填完整渲染）

用户对架构的根本性批评（正确）：hole 的轮廓/颜色/质感全部押注在低通图上——RefNet
在 hole 内无逐位置通路（validity=0 是几何事实：hole=源照片看不到的区域）、W+ 是
全局 token、cycle 只有 256 回投。"网络结构干什么吃的"的答案：它被我们对 EG3D 域
污染的恐惧废掉了——把"需要的内容（清晰轮廓+颜色）"和"害怕的东西（塑料感）"一起
砍进低通里了。

**orig 的洞察读反了**：§17.8 的教训是"EG3D 当 **loss 目标**有毒"，从来没说"当
**输入条件**有毒"。orig 的 hole 从第一天就填完整清晰渲染，FFC CNN 的任务是质感
翻译（塑料→照片），由真实域 GAN/FM 监督。低通色块还有次生问题：非自然分布过
SD VAE 产生脏斑。

**v3.3 = 条件/目标解耦（核心改动）**：

```text
条件 hole = 完整 EG3D 渲染（清晰轮廓+颜色）   ← orig hybrid 忠实回归
目标 hole = lowpass(EG3D) @0.1               ← 塑料感防污染保持（§17.8 不复发）
质感翻译  = cycle(真实源照片 L1+LPIPS+PL+ID) + real_ref(pixel_sup GAN)
```

代码：coach forward() 的 inv_warp_visible 分支显式构造 hole_target=lowpass(render)
（不再继承 condition）；config `mode="rgb"`。step-0 基线 novel_hole=0.0498
（历史最好起点；lowpass 0.059、depth 0.466）。保留全 timestep 修复、300K、25K 存档。
v3.2 只跑了 600 步（中止原因：分析发现 v3.1 的教训指向条件侧而非深度侧，直接跳到
v3.3 避免 300K 白跑）。

## 5.5 2026-08-22：v3.3 中止 → v4（用户方案：诚实条件 + ControlNet 多条件范式）

用户对整个方法论的否定（正确）：低通条件=“手搓面糊糊再让网络优化边缘”——内容是手工的，
网络从未学过生成。v3.3（完整渲染填 hole）同属此类（只是把糊糊换成塑料），600 步即中止。
用户澄清此前一直以为低通是深度信息。

**v4 = 用户方案的忠实落地**：

```text
条件（全部真实信号，无手工内容）：
  hole 填充  = 噪声（诚实声明"此处无信息"）
  +第6条件通道 = 全图归一化逆深度（新视角几何边界在哪）——BrushNet conv_in_condition
                5→6ch，新通道零初始化（step-0 = 预训练行为），训练中学会使用
  RefNet     = 源照片外观（全局 attention）
  W+         = 身份（全局 token）
  visible    = 真实 warp 像素
监督（全部真实域，无伪目标）：
  hole 像素监督完全移除（geometry_noise/x0_weight=0）
  cycle 回投真实源照片 @512（主监督，L1+LPIPS+PL+ID）——瞳孔细节的分辨率瓶颈解除
  real_ref 自重建（pixel_sup GAN）
```

关键佐证：real_ref 的 hole 从来就是无内容黑块，网络照样补到 0.026——证明"无内容条件
+真实监督"可行；novel 分支此前的失败在监督（弱/伪）而非条件缺内容。orig 的 hole 监督
也从来只是 cycle（真实照片），从未有过像素 GT。

代码：utils（noise 模式、normalize_depth_latent、采样函数 depth 参数）；coach
（conv_in_condition 扩展、forward 产 depth_cond_image、loss cat 第6通道、validate 传参）；
脚本（mode=noise、depth_condition、hole 权重 0、cycle 512）。step-0 基线 novel_hole=0.27
（噪声 hole 的预期起点）。显存 23.9GB（512 cycle 代价，贴边可行）。

**预注册**：5K 检查 hole 结构是否成形（depth 通道是否被学会使用）；若结构错乱→回退
hole MSE@0.1 做 geometry 锚（消融A教训的现代化版本）。10K 肉眼裁决质感。

## 5.6 2026-08-22：Phase-1 启动（纯 BrushNet + 双任务，ControlNet-depth 范式）

经完整架构讨论（用户主导）定稿：**单一写入者 + 诚实条件 + 证据门控插件**。

```text
架构: SD UNet 100% 原生冻结（RCA 全拆，stock attention，空prompt）
      BrushNet 618M 唯一可训练件，从官方预训练权重起步
      条件 6ch: [noisy 4 | masked图 3 | mask 1 | depth 1]
      depth 通道 conv 零初始化（step-0=预训练行为），模型学会读

Task A（50%，real_ref 自重建）: hole=warp洞形状（照片自身mask），
      depth=源视角（与照片真配对），监督=扩散MSE vs 源照片（真像素GT）
      + pixel_sup(LPIPS+PatchGAN)——三个月来首次真收敛判定
Task B（50%，novel cycle）: hole=噪声，depth=novel视角（与推理同构），
      监督=cycle 回投真实源照片 L1(10)+LPIPS(10)+PL(30)+ID(0.5)@512
hole 像素监督=0（低通 target 全面退役；depth 条件+cycle 承担几何锚）
```

插件路线（证据门控，预注册）：10K hole 颜色漂移→+RefNet（零初始化 conv 进
BrushNet）；20K 大hole身份崩坏→+W+（token 拼进 BrushNet cross-attn context，
即"文本槽位=条件总线"方案）；30K 纹理不足→FM@t<50。5K hole 结构错乱→hole
MSE@0.1 低通锚回退（消融A的现代化检验）。

修复的 bug：纯模式下 trainable_parameters 为空导致优化器未构建（Grad=0）；
validate() 无 ref_feat 的 KeyError。step-0 基线：novel_hole=0.231（噪声 hole
预期起点）。`experiments/_joint_train_phase1/`，300K 步/25K 存档。

## 5.7 2026-08-23：Phase-2 启动（RefNet 融合插件 + 可视化升级）

触发证据（Phase-1 纯基线 25K 数据）：自重建 hole 0.099 vs 历史 0.026（带 RefNet
对齐路）——退化 100% 归因 RefNet（该分支 W+ 历史上一直禁用），预注册的 Phase-2
条件按预期触发。用户同时指出 novel 极端视角"脸颊顺 hole 无限延长"（监督盲区：
Task A 大洞样本稀少 + Task B cycle_valid≈0.73 区域无信号，depth 通道是盲区唯一
真信息、仍在学会重视的路上——已分析未改动，留观）。

**Phase-2 改动（单变量：+RefNet 融合插件）**：
- `models/refnet_fusion.py`：RefNetBrushnetFusion——4 尺度零初始化 1×1 conv
  （3.79M 参数），把**对齐后**的源照片特征加进 BrushNet down/up residual 流。
  零初始化 = step-0 输出与 Phase-1 完全一致（从 Phase-1 25K checkpoint 热插拔，
  权重零扰动）；单一写入者保持（一切仍经 BrushNet add_res 进 SD）
- RCA attention adapters 在插件模式下强制禁用（use_refnet_fusion 时
  rca_adapters_enabled=False——否则随机初始化的 gate 会注入噪声）
- optimizer 状态不兼容时自动重建（参数组拓扑变化，权重保留、动量重算）
- 监督不变（Task A 真GT MSE + Task B cycle@512，50/50）

**可视化升级**（`scripts/eval_phase2_visualization.py`，
`experiments/_phase2_viz/phase2_overview.png`）：8 面板×3 identity——源照片 /
novel depth 伪彩 / 源 depth 伪彩 / Task B 条件(噪声hole) / Task B 输出 /
cycle 回投 / Task A 条件 / Task A 输出。深度图等中间量直接可看。

### 事故记录（2026-08-23 下午）：REAL 面板"雾蒙蒙"的完整根因链

现象：25K→29K 期间 real_ref 验证指标翻倍恶化（full 0.047→0.095，visible 0.026→0.075），
REAL SD output 面板全图蒙纱；novel 分支不受影响；训练 real_ref loss 从 1.1 坍缩到 0.002。

三个叠加因素（全部已修复）：
1. **验证采样器漏接插件**：`sample_brushnet_inpainting` 里 BrushNet 前向后没有
   `refnet_fusion` —— 训练时 SD 看到的是"BrushNet+插件偏移"，验证时凭空抽掉插件，
   SD 收到从未训过的残差分布 → 全图（含 visible）蒙纱。**训练本身是健康的，坏的是仪表盘。**
2. **Task A GT 泄漏捷径**：Task A 的恒等对齐特征就是干净源照片 = GT 本身，插件在
   1K 步内学出直通捷径（loss 坍缩 500 倍 = Task A 监督名存实亡），同时 BrushNet 自身
   通路"躺平"——这解释了为何只有 REAL 面板崩（novel 特征是 warp 有损版，无法当 GT 用）。
3. **save_interval=25000**：续训后下一次存盘要到 50K，中途崩溃全丢。

修复：
- 采样器加 `refnet_fusion` 参数（镜像训练路径）；coach 验证 novel 调用传入插件
- **融合插件只对 Task B（novel）生效**（`batch_kind != 'real_ref'` 门控）：Task A
  本来就有真 GT 监督、不需要源先验，喂它只制造捷径。这是设计修正——插件的假设本来
  就是"帮监督盲区的 novel hole"，不是帮已有完美监督的自重建
- Task A 验证调用走纯路径（无插件无 reference kwargs），面板回归"诚实的自重建测量"
- save_interval 默认 2000
- `eval_phase2_visualization.py` 同步：加载插件权重、novel 走融合、Task A 纯路径
- 代价：25K→30K 的 5K 步作废（无 checkpoint 可救），从 25K 重启

### Phase-2 证伪记录（2026-08-23 晚）：RefNet 融合插件无效 + depth 通道萎缩的根因发现

**用户裁决确认 + 数据支持**。插件从 25K 训到 50K（25K 步，远超 35K 判决点）：

**证伪证据（三重）**：
1. **指标零变化**：novel_hole 全程 0.124-0.159 震荡无趋势（Phase-1 同步 0.135-0.167，
   噪声级重叠）；novel_edge（轮廓指标）0.025-0.031 vs Phase-1 0.022-0.028——轮廓无改善
2. **架构级不可能**：`Splatting.forward_warp` 的 hole 区域输出恰好全零（splat 定义），
   warp 对齐后的 RefNet 特征在 hole 内部本来就是零——插件在 hole 里只能加 `proj(0)=bias`
   常数，visible 区域又与 warp_img 条件冗余。**结构上无法给 hole 送源内容，训练多久都没用**
3. **权重证据**：scale 0-2 权重非零（0.67/1.14/1.55，模型部分采纳但无指标收益——冗余信号）；
   scale 3 严格为零（梯度从未流动）

处置：训练已停在 50K；checkpoint 保留在 `_joint_train_phase2/checkpoints/`（含 iteration_50000.pt
作为证伪证据）。插件代码保留但不启用。

**更重要的新发现——"轮廓不清"的根因（depth 通道萎缩）**：

`conv_in_condition`（10ch = 4 noisy_lat + 4 masked_lat + mask + depth）逐通道范数：

| 通道 | Phase-1 25K | Phase-2 50K |
|---|---|---|
| noisy latent | ~1.85 | ~1.85 |
| masked latent | ~3.4-3.7 | ~3.4-3.7 |
| mask | 0.995 | 0.946 |
| **depth** | **0.0899** | **0.1172** |

**depth——hole 轮廓的唯一几何契约——权重比其他条件弱 10-30 倍，且 25K 步几乎不增长**。
因果解释：hole 的训练 target 是高斯低通 EG3D（模糊），输出锐利跟随 depth 边缘反而增大
MSE——**监督在主动惩罚 hole 内锐利结构**。Task A（真 GT 锐利）与 Task B（低通模糊）梯度
冲突，模型取平均→软。这同时解释轮廓不清与极端视角脸颊延长（depth 不约束边界）。

**因果消融验证（2026-08-23，`scripts/eval_depth_channel_ablation.py`）**：
50K checkpoint，同一种子下把 depth 通道整体置零（归一化后常数 1.0，无任何空间结构），
对比换种子（无关扰动基线），3 个 identity：

| 指标 | 抹掉全部 depth 结构 | 仅换随机种子 |
|---|---|---|
| 全图 L1 | 0.0045-0.0088 | 0.0266-0.0900 |
| hole 内 L1 | 0.0115-0.0148 | 0.0800-0.1273 |

**抹掉 depth 的影响仅为换种子的 1/7-1/10**——输出由 warp/mask 条件+采样噪声主导，
depth 贡献在噪声量级。与权重范数证据（0.09-0.12 vs 3.5）互相印证：depth 通道
事实上未被读取。可视化：`experiments/_depth_ablation/depth_ablation.png`
（A 真depth / B 零depth / |A-B|x5 几乎全黑 / C 换种子 / |A-C|x5 大量结构）。

**为什么输入准却没被用**：网络只学"用它能降 loss 的东西"。Task A（源帧小洞）
上下文插值即可填洞，depth 增益小；Task B（novel hole）的 target 是高斯低通
（模糊）——跟着 depth 画锐利边缘反而增大 MSE。两条监督合力的最优策略就是
"不读 depth"。可视化里看到的是**输入**的准确性；权重和消融测的是**模型**
是否读它——两者互不矛盾。

**下一步候选（按推荐排序）**：
- A. 边缘保持 hole target：EG3D 渲染用双边滤波替代高斯低通——保留几何轮廓锐度、
  继续压制塑料质感。最小改动、直击根因。风险：EG3D 边缘统计可能与真实照片有差
  （但轮廓由几何决定，非外观）。判据：depth 通道范数应显著增长 + novel_edge 下降
- B. 结构-only 损失：hole 内 hinge 惩罚 |∇img| << w·|∇depth|（只要求"depth 有边的地方
  图像必须有边"，不规定外观）。避免动 target，但 x0 解码空间的损失历史上有发散前科
  （GAN/FM），需谨慎
- C. 冻结 Phase-1 为最终方案，接受当前质量，转推理/部署





## 5.8 2026-08-23 晚：depth 引导的 hole target（代码名 v5；**非插件阶梯的 Phase-3**）

> 编号勘误（2026-08-24）：分期约定为 Phase-1 纯 BrushNet → Phase-2 RefNet 插件 →
> **Phase-3 W+ 插件**（证据门控）。本节实验是监督契约修复，不属于插件阶梯，曾误用
> "Phase-3"编号——目录 `experiments/_joint_train_phase3/` 因训练进程在写而保留原名，
> 语义上应读作 `experiments/_joint_train_v5_edge_target/`。**Phase-3=W+ 插件仍在
> 路线图上**，待本实验出结果后按证据决定是否启动。阶梯暂停理由：Phase-2 验尸发现
> 监督契约惩罚锐利边缘（depth 通道废掉），地基不修，任何插件（含 W+）都无法解决
> 空间轮廓问题——W+ 是全局 token，无空间通路，先上只会重演 Phase-2。

第一性原理：条件去噪模型是最优预测器，读一个条件 ⟺ 该条件含其他条件不可预测、
且 loss 度量的结构。Phase-1/2 的 hole target（高斯低通 EG3D）把 depth 唯一能预测的
东西（锐利轮廓）销毁了——读 depth 反而增大 MSE。修复=把轮廓放回 target，且只放在
depth 知道的地方。

**新 hole target（`build_geometry_condition(mode="depth_guided_rgb")`）**：

```text
T_hole = lowpass(EG3D) + 2.0 · (g − box(g))      # g = 归一化逆深度（与第6条件通道同一变换）
```

- 颜色 = 低通 EG3D（防塑料感的 §17.8 防线原样保留）
- 边 = depth 的带符号高通：**带符号的台阶、精确落在 depth 不连续处、平滑面≈0**
- edge_gain=2.0 由探针标定（`scripts/probe_hole_target.py`）：hole 边缘能量
  0.0082-0.0149 ≈ 源照片量级（0.0099-0.0156）；旧低通只有 0.0058-0.0083
- **实现弯路（已证伪并记录）**：先试了经典 guided filter（He 2010），探针否决——
  它只能保留"图像与 guide 局部相关"的边，而 hole 恰是 EG3D 平坦、depth 有边的区域，
  guided 输出比高斯低通更糊（lap 0.0042-0.0062）。带符号高通构造无此依赖

**单变量原则**：vs Phase-1 只改 hole target 构造；RefNet/插件全关（Phase-2 证伪），
从 Phase-1 25K checkpoint 续训（optimizer 单参数组兼容）。其余监督完全不变。

### v5 证伪记录（2026-08-24，判据 1 失败）

预注册判据终审（52K）：
- **depth 通道范数 0.1159**（要求：10-15K 步内 >>0.5）——失败
- novel_edge 0.026-0.032 全程持平；用户视觉裁决"轮廓仍不清、极端角度与背景融合"
- **决定性对照**：v5（target 带边）27K 步 0.0899→0.1159；Phase-2（target 无边、纯低通）
  25K 步 0.0899→0.1172——**增长完全相同**。target 里的边对 depth 权重采用率的
  边际贡献为零。训练已停（52K checkpoint 保留于 `_joint_train_phase3/`）

**两个探针排除了两个候选根因**：
1. ~~VAE 瓶颈~~（`scripts/probe_latent_edge.py`）：guided target hole latent-lap=0.3553
   vs 低通 0.1573——边在 loss 计算的 latent 空间真实存在，能量翻倍
2. ~~边不在 target 里~~（`scripts/probe_hole_target.py`）：像素空间边缘能量
   0.008-0.015 ≈ 源照片量级

**真正根因（机制分析，待下一步实验检验）——优化器没有理由走 depth 这条路**：
1. **冷启动**：depth 通道是 conv_in_condition 零初始化扩展的第 6 通道（为保
   step-0=预训练行为）——初始权重≈0，梯度只能来自"与 depth 特征相关的残差误差"
2. **激励稀释**：depth 独有的信息（hole 内部深度边位置）只占 loss 像素的 ~1-2%；
   hole 边界环的边可由 mask 通道推断（disocclusion 边界≈mask 边界），内部软边可由
   SD 先验以极小 MSE 代价近似——不读 depth 只损失 ~1% loss
3. **L2 回归均值**：任何残余位置不确定下，ε-MSE 最优输出=模糊边；确信的锐边需要
   模型先学会读 depth——鸡生蛋困局。极端视角"与背景融合"正是条件均值的签名
4. 两 run 相同的 0.09→0.116 增长大概来自 Task A（源视角 depth 与真照片边是真对）

### v6（2026-08-24）：边缘带监督——真根因发现与修复实施

**审计发现 v5 证伪记录里的根因分析也是错的**：v5 归因"激励稀释（~1-2%）"，实际读代码发现
`geometry_noise_weight=0.0`（v4 起如此，run_joint_train.py 注释"v4: NO hole pixel target"）
——**Task B hole 的 ε-MSE 激励是精确的 0**，不是小。v5 换的 target 从未进入 loss；它唯一的
作用是作为加噪输入（低 t 时反而是泄漏旁路）。"两个 run 收敛到同一终点"的真正解释：
两者都在"hole 零监督"这同一个目标函数下训练，depth 权重的微小增长均来自 Task A。

**v6 实施（`training.edge_band`）**：
- `_build_edge_band`：从 novel depth（与第 6 条件通道同一归一化）latent 梯度取 top-10%，
  膨胀 2px，∧ hole mask——"监督带"与"可读条件"是同一信号
- 损失新增两项（real_cycle 分支）：`edge_band_weight·MSE(ε)` + `edge_band_x0_weight·
  snr_x0(带内)`；**洞内其余像素保持 0 监督**（防 EG3D 域污染的零原样保留，比 v4 之前
  的全局 0.1 更严格）
- v5 的带边 target 保留（有监督的带必须有带边的答案，否则不可满足）
- 参数：weight=10, x0_weight=2, quantile=0.90, dilate=2（env 可调）
- 冒烟证据：real_cycle 梯度 0.7→37（×50）——激励从 0 到主导的直接信号
- 从 Phase-1 25K 续训，目录 `experiments/_joint_train_v6_edgeband/`

**预注册判据（5K 步 = 30K 前后）**：
1. depth 通道范数 >0.2 且上升（否则证伪回退，一行配置）
2. 10-15K 步：novel_edge 低于 Phase-1 区间（0.022-0.028）
3. 消融：|A−B|（抹 depth）升至 |A−C|（换种子）同量级
4. 视觉：极端视角轮廓与背景分离（用户终审）





## 6. 入口

```bash
bash scripts/run_joint_train.sh          # 300K steps
WARP_GAN_MAX_STEPS=2 ... smoke           # 冒烟
python scripts/eval_phase2_visualization.py   # 中间量可视化
```





