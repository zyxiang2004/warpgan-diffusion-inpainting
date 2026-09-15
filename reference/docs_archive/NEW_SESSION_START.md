# 新会话启动文档（IMPLEMENTATION BOOTSTRAP）

> **状态更新（2026-08-24 10:38 起）**：Step 1'（纯 ε 25K）已完成并通过肉眼终审（hole 0.039-0.045，
> 小/中角度好，大角度油画/破损——症状归因与质感消融见 STEP0_SMOKE_REPORT.md §8.10-8.12）。
> **Step 2（+x_mirror，老师钦点消融 #1）已启动**：从 24999 续训 15K 至 40000，单变量
> `reference.use_mirror=True`，实验目录 `experiments/train_inpainting_diffusion/step2_mirror_15k/`，
> 日志 `train_logs/step2_mirror.log`。训练结束后自动产出 baseline vs mirror 的 A/B 对照
> （`step2_mirror_15k/ab_review/`，7 联图：x|渲染|条件|mask|锚|基线生成|mirror生成）。
> 本文档以下内容写作于 Step 0 之前，§4 的"GPU 被占用"已过时，其余仍有效。

> 写于 2026-08-23。如果你是刚接手的新 agent，**按顺序读完本文 + 两份文档就可以开工，
> 不需要任何历史对话记忆。**本项目的全部知识已外部化到文件。

## 1. 任务

把原版 WarpGAN 的 SVINet（FFC-ResNet 修补器）忠实翻译成 BrushNet + SD1.5 diffusion，
**严格按 `docs/ORIG_FAITHFUL_PORT_SPEC.md` 执行**（那是老师方案 + 原版代码语义的定稿
spec，所有歧义已按原版默认值裁决，勿自行发明）。

## 2. 必读（按顺序）

1. `docs/ORIG_FAITHFUL_PORT_SPEC.md` —— 实施规格（改什么、怎么改、判据、坑）
2. `docs/PROJECT_HISTORY.md` §4（已证伪清单）、§5（监督契约）、§5.3（防幻觉规程）
   —— 其余章节按需查
3. 本文

## 3. 目录与角色

| 目录 | 角色 |
|---|---|
| `/data/xzy/warpgan20260803/20260803/warpgan_orig/WarpGAN-main/` | **工作底座（原版代码）**：在其上改造。关键文件：`training/coach_inpainting_static.py`（训练主循环，L489-560 synth/real 交替、L207-305 双 pass forward、L1054-1074 损失入口、L1118-1212 损失实现）、`configs/train_inpainting.yaml`（原版权重：L1 10/GAN 10/FM 100/PL 30/ID 0.5/latent 0.1，`with_mask:False`、`warp_pred:False`、`hybrid:True`、`synth.able:True`、`input_mirror:'condition'`）、`datasets/dataset_inpainting_static.py` + `dataset_inpainting_synth_static.py`（真实/合成数据）、`models/saicinpainting/`（FFC 生成器，被替换对象）、`utils/warp/Warper.py`（forward_warp/inverse_warp，保留复用） |
| `/data/xzy/warpgan20260803/20260803/models/` | **创新模块库（从主工程搬来的成品）**：`BrushNet-main/`（BrushNet 源码，conditioning_channels 5/6）、`referencenet/`（SD-UNet 副本 + `attention_processor.py` 注意力注入）、`mapper/w_proj.py`（W+ [14,512]→[18,768] 叠块对齐 mapper，老师②的现成实现）、`diffusion_inpaintor.py`（推理封装）、`w-plus-adapter-main/`（W+ adapter 参考）、`refnet_fusion.py`（已证伪插件，勿启用） |
| `/data/xzy/warpgan20260803/20260803/`（主工程根） | 历史工程。`training/coach_inpainting_static.py` 里有大量可参考的 diffusion 化实现（ε-MSE、cycle@512、x0 单步、像素监督 patch、全 timestep 采样），按需抄；`utils/diffusion_inpainting.py`（共享采样器）；`scripts/run_joint_train.py`（joint mode 入口参考）；`docs/`（历史文档+归档） |

**改造方式建议**：在原版工程内新增（如 `training/coach_inpainting_diffusion.py` +
`configs/train_inpainting_diffusion.yaml`），不动原版文件本身，方便随时 diff 对照
原版语义。原版工程缺的依赖（diffusers 等）从主工程环境拿：conda env `warpgan`
（`/home/xzy/miniconda3/envs/warpgan/bin/python`）已装齐。

## 4. 当前机器状态（开工前必查）

- **GPU 0 正被主工程 `_joint_train_orig` 训练占用（19.7/24.5GB，99% util，PID 见
  `ps aux | grep run_joint_train`）**。它是旧方案（teacher spec 定稿前）的 run。
  开工决策（用户已倾向按新 spec 重来）：**先征求用户是否停掉它**再跑冒烟；
  未停之前任何 GPU 测试都会 OOM。
- 数据/预训练权重沿用原版工程路径（`./pretrained_models/`、`./data/`，相对原版
  工程根解析）。

## 5. 实施顺序（照 spec §8，此处只列骨架）

1. **Step 0 冒烟（2 步）**：搭批次 A（合成）+ 批次 B（真实双 pass）数据流；
   验证 ε-MSE / pass2 像素损失 / latent 闭环 / W+ 无梯度（冻结档）/ RefNet 输入
   = inversion 图；**打印有效配置逐项核对**（历史事故：配置写了≠生效）；
   峰值显存 <22GB。
2. **Step 1 基线 25K**（无 mirror）：预注册判据见 spec §8。
3. **Step 2 单变量 +x_mirror**（老师钦点第一消融）。
4. 判据失败即停，回到 spec/PROJECT_HISTORY 追加分析，不带病长跑。

## 6. 红线（违反任何一条 = 重踩已记录的坑）

- 真实数据的 novel 输出不得对 EG3D 渲染求高权重监督（油画）；
- hole 不得零监督（崩盘）；hole 条件不得用纯 depth（蓝灰）；
- W+ 分支解 QKV 只许在合成批次上（真实批次上 = §17.8 碎孔/油画）；
- GAN/FM 默认 0（x0 残噪上两次白爆）；
- 训练 timestep 全范围 [0,1000)（只训低 t = 雾）；
- 不做硬拼接输出（full-frame 契约）；不做 mask 腐蚀/闭运算；
- 单变量实验 + 预注册判据 + 肉眼终审 > 自动指标。
