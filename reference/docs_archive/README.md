# WarpGAN Diffusion 移植 —— 项目总入口（新 agent 必读）

> **更新：2026-09-08 深夜（v16@100K 训练进行中）**。本文是 `docs/` 的唯一导航入口，
> 整合了全部历史文档与本日之前的全部对话决策。新 agent 读完本文即可开工；
> 各历史文档按 §7 的地图按需深入。**规则：任何新决策/新实验/新教训都应
> 追加到 STEP0_SMOKE_REPORT.md（§8.x 流水账）并在必要时更新本文。**

---

## 0. 30 秒版

- **任务**：单张真实人脸照片 → 3D warp 到新视角 → 填补 disocclusion 空洞 → 输出
  真实感新视角图像。**hole 区域不存在真值照片**（那个视角没被拍过）——一切设计
  张力源于此。
- **做法**：用 冻结SD1.5 UNet + BrushNet（add-samples 控制）+ attn1 bank 检索（照片
  纹理注入，AnimateDiff 范式）+ attn2 W+ cross（3D 身份潜码注入）替换原版 WarpGAN
  的 FFC-ResNet 修补器（原版输出油画感）。
- **现状**：v16（v12 + pass2 真照片像素回流）续训至总 100K 步进行中；下一版
  **v17 = v16 + pass2 对齐 FM**（补特征层质感供给）已定义待起跑。
- **当前主要瓶颈**：盲区高频质感（沙沙感）——真值供给密度不足（§4 修正理论）。

## 1. 两句话的项目史

1. **前史纪元（2026-05 ~ 08-23，详见 PROJECT_HISTORY.md）**：自研 diffusion 方案
   （joint 训练、factorized 双分支、edge band、W+ 可训……）四个月大量试错未达标，
   沉淀出"已证伪清单 + 事故录"（该项目最贵的资产）与五大方法论铁律。
2. **忠实移植纪元（08-23 起）**：推倒重来——按老师钦点的 ORIG_FAITHFUL_PORT_SPEC
   从原版代码逐行忠实移植（教练 coach_inpainting_diffusion.py），然后在**单变量
   纪律**下一版一版做受控实验（v1→v17），每版预注册判据、肉眼终审。

## 2. 当前架构（60 秒版，详见 ARCHITECTURE_DIFFUSION.md）

```
入口 scripts/train_inpainting_diffusion.py (hydra)
 └─ Coach（training/coach_inpainting_diffusion.py）
     ├─ 冻结: VAE / SD1.5 UNet / CLIP空提示 / WplusNet(GOAE) / Warper(前向splat) / WarperExt(反投)
     ├─ 可训: BrushNet(官方inpainting权重) / 每层SD注意力的ReferenceAttentionProcessor / w_mapper
     └─ train(): real批与synth批 1:1 交替（同原版）
          real pass1(novel): 条件=warp可见+渲染填洞；监督=dual-band结构锚(照片证据全带/盲区低带)
          real pass2(source): 条件=pass1条件图warp回源视角(warp两次)；监督=ε-MSE vs x
                              + t<200的x0-patch L1×2 真照片像素回流(v16)
          synth批: EG3D双视角渲染对；只训ε结构段(质感段=0, 油漆红线) + W+ mapper唯一更新点
     注入: attn1 = bank检索(主位x_mirror照片/副位渲染, K/V注入)
           attn2 = W+ tokens(18×768)与空文本并行cross, 固定tanh(3)门
```

## 3. 版本谱系（v1→v17，一版一行；详情见 STEP0 对应 §8.x）

| 版本 | 单变量改动 | 裁决 | 教训/要点 |
|---|---|---|---|
| Step0 | 冒烟+探针A | ✅ | **像素损失族放 ε 位置=毒损失**（PL×30 低t ε 0.9→1.6）§8.5 |
| step1b | 纯ε基线 25K | ✅ best | hole 0.039-0.045；大角度油画/破损 §8.10-12 |
| Step2 | +x_mirror | ❌ 崩 | **根因是 resume 损失覆写丢失**（§8.16 配置事故） |
| v1-v3 | (事故代) | ❌ | 同上：v1-v3 全部无效运行——**resume 必须重申全部覆写** |
| v4/v5 | mirror 监督覆盖 | ❌ | v6 条件端几何对齐仍碎片化 |
| v6 | 条件+监督对齐 | ❌ | **二值 mask=碎片指纹（1800 连通域）→ 连续置信度场** |
| v7/v7a | 软检索+连续w | ❌趋稳 | softmax 份额不动 → **mirror 需独立 K/V**（源区分） |
| v8 | bank替代860M RefNet | ✅等价 | 像素级等价(0.7-1.6%)，省 860M/9G §8.21-22 |
| v9 | synth结构监督+mirror主位 | ✅微 | "监督结构决定纹理域"；**原版cond软化曾被误判死配置** §8.26 |
| v10 | 条件软化(bitwise=orig) | ❌ | **负结果：条件端不是接缝瓶颈**（对拍 3.58e-07）§8.27 |
| v11 | hole零监督+local窗口 | ❌ | local gate 走低；盲区高频漂移实证(0.041→0.094) |
| v12 | 盲区低频锚(dual-band)−local | ✅ **best结构** | 结构/接缝稳住（用户裁决）；**纹理仍糙**；顺带修 v4-v11 隐性 bug（权重场 latent 阶跃）§8.31-34 |
| SDEdit | 4采样对比探针 | ❌ | **推理端不可救**（三种初始化三种失败）§8.33 |
| v13 | 原版全配方恢复(latent闭环+身份层) | ❌ | v12 优于 v13；输出侧身份锚无收益+果冻 §8.35b/37 |
| v14 | GAN/FM(novel x0 vs x) | ❌崩 | **视角错位对**：D 先学视角差；FM×10 也炸 §8.39-40 |
| v15 | 盲区score保真teacher | ❌ | hole 不如 v12、色调差 §8.41-43 |
| v16 | **pass2 像素回流**(L1×2,t<200,32patch) | ✅ 进行中 | 同位重建原版唯一照片质感通路；50K"还不错仍沙沙" §8.45-46 |
| v17 | **pass2 对齐 FM**(计划) | 待跑 | 真=x照片/假=pass2 x0解码(t<200)；v14 基建开关即用 §8.51 |

## 4. 理论框架演化（为什么是 v17）

1. **证据分级原理**（最高原则）：源照片 > x_mirror 真照片 > EG3D 渲染（只有结构
   可信，纹理=塑料）> SD 先验。监督/注入强度必须与证据强度匹配。
2. **四要素**：形状（warp+渲染条件）/ 质感（照片分布）/ 身份（W+）/ 一致性（接缝）。
   v12 解决了形状与接缝；**质感是唯一剩余缺口**。
3. **拓扑理论（§8.47，已被部分修正）**：曾认为"监督拓扑搭反"（novel 位置该零监督
   全靠分布约束）——v14/v15 按它做反而崩。
4. **修正理论（§8.50，现行）**：真正瓶颈=**高频质感的真值供给不足**。原版把唯一
   照片质感教学全押在 pass2 对齐对上且密度极高（每步全帧 L1×10+PL×30+FM×100+
   adv×10）；v16 同位置只有 ε-MSE+patch L1×2（18% 命中）——**密度差约两个数量级**。
   v17 的 FM 就是在正确位置提密度的下一步（与原版 real pass2 的 FM 同位同参照）。

## 5. 难点与红线全集（违反任何一条 = 重踩已记录的坑）

**损失侧**：
- 像素/感知损失直接做 ε 目标 = 毒损失（x0 残噪梯度放大；两次白爆实证）——只许经
  t<200 门控的 x0 解码路径（v16 回流/未来 v17 都遵守）。
- 真实 novel 对 EG3D 渲染高权重监督 = 油漆（v3 300K 实证）；synth 质感段权重必须 0。
- 渲染高频不进任何监督目标；低通**图**不进输入/目标（与"误差低通=损失频段选择"
  严格区分——后者是 v12 资产，前者是红线）。

**结构侧**：
- 二值 mask 进监督 = 碎片指纹 → 一律连续置信度场。
- 纯 depth 填洞 = 蓝灰（hole 需颜色基底）。
- hole 长期零监督会高频漂移（v11 实证 0.041→0.094）→ 至少低频锚。
- 判别/特征对必须**视角对齐**（v14 教训：novel x0 vs source x 的错位对必炸）。

**W+ 侧**：
- W+ 分支（mapper/QKV）只在 synth 批更新（real 批梯度清零——§17.8 前史红线）。
- gate 固定 tanh(3) 常开；不解 QKV 除非 frozen/s1 都验证完。

**工程侧（事故录）**：
- **resume 必须重申全部 CLI 覆写 + 核对 EFFECTIVE CONFIG**（v1-v3 整代事故）。
- 训练 timestep 全范围 [0,1000)（只训低 t = 雾，v3.1）。
- full-frame 输出契约（不硬拼接）；不 mask 腐蚀/闭运算（条件端软化除外=orig 语义）。
- hydra 路径含方括号必须加引号；sed 改 eval 脚本的 /checkpoints 后缀模式已三次踩坑。
- NFS 会挂（已两次）：看门狗必须先探 `ls ./data/` 再 resume；/tmp 重启即清——守护
  脚本放 `scripts/`（持久位置）。
- 磁盘：每实验定期清中间 ckpt（单 ckpt 4.5G）。
- **配置写了 ≠ 生效**：EFFECTIVE CONFIG 打印逐项核对是每次启动的固定动作。

## 6. 当前状态与下一步（2026-09-08 深夜）

- **在跑**：v16 续训至总 100K（进程见 `pgrep -f train_inpainting_diffusion`；日志
  `train_logs/step3_v16_200k_v5.log`；实验目录 `experiments/train_inpainting_diffusion/v16_run/`）。
  守护（持久位置 `scripts/`）：`v16_resume.sh`（自动取最新 ckpt，目标 100K）→
  `v16_100k_queue.sh`（等 iteration_0099999.pt → 停训 → 自动跑 eval_step2_ab.py）→
  `v16_100k_watchdog.sh`（死进程自动复活，先探 NFS）。
- **100K 判读程序（§8.51）**：A/B 三方对照（v16@100K vs v12 vs step1b）+ lap ratio
  趋势（50K 末桶 1.065 是否续降）→ 持续改善=步数假说成立；平台=密度瓶颈证实 →
  无论哪个，**v17 起跑**。
- **v17 配方（§8.51 已定义）**：v16 + pass2 对齐 FM——真=x 照片、假=pass2 的 x0
  单步解码（t<200 门控，x0_clip=3.0），权重 10（原版 100 的 1/10 起步），FM-first
  不带 adv；基建（make_discriminator/_gan_fm_g_loss/_discriminator_step）v14 已留，
  CLI 开关即用。单变量纪律：其余一切不动。
- **理论哨兵**：novel 监督降权对照（若出现大正效果 → 拓扑理论复活，§8.50）。

## 7. 文档地图（本目录）

| 文档 | 时效 | 用途 |
|---|---|---|
| **README.md（本文）** | 持续更新 | 总入口：历程/理论/红线/现状/导航 |
| ARCHITECTURE_DIFFUSION.md | 当前有效 | 代码文件地图+调用图+v16 数据流（09-08 新写） |
| ORIG_FAITHFUL_PORT_SPEC.md | 长期有效 | 老师钦点的移植规格（08-23 定稿/08-27 修）——一切语义歧义回原版代码裁决 |
| STEP0_SMOKE_REPORT.md | 流水账（128K+） | **全部实验细节的原始记录**（§8.1-8.55）；本文的任何概括与它冲突时以它为准 |
| PROJECT_SNAPSHOT_20260829.md | 停在 v12 | v12 完成日全景快照（理论/证据库/候选路线——其中 §8"侵蚀说"已自我修正） |
| PROJECT_HISTORY.md | 前史权威 | 2026-05~08-23 全史：已证伪清单（§4）、事故录（§6）、监督契约（§5）、Stage A/B/C 考古索引（§7.5） |
| NEW_SESSION_START.md | 已过时(08-24) | 当时开工文档；红线一节仍可读，其余被本文覆盖 |
| archive/ | 考古 | 前史 10 份原始文档（PROJECT_HISTORY 附录有清单） |

## 8. 新 agent 开工清单

1. 读本文 → ARCHITECTURE_DIFFUSION.md → （按需）PROJECT_HISTORY §4/§6。
2. 查训练状态：`tail train_logs/step3_v16_200k_v5.log`；进程/看门狗：
   `pgrep -f train_inpainting_diffusion; pgrep -f v16_100k`。
3. 若 100K 已完成：先看 `train_logs/step3_v16_100k_ab.log` 与
   `v16_run/logs/images/val/`，按 §6 判读程序裁决，然后向用户汇报再动 v17。
4. 动手前过一遍 §5 红线；写代码前看 coach 模块头的 ARCHITECTURE MAP 注释。
5. 任何新实验：预注册判据 → 单变量 → EFFECTIVE CONFIG 核对 → 肉眼终审 →
   结果无论好坏都追加到 STEP0_SMOKE_REPORT.md。

