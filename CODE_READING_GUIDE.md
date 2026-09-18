# CODE_READING_GUIDE — 从零读懂当前代码（2026-09-18 编写）

> 目的：**不改任何代码**，用一条可核实的路径建立对当前系统的心智模型。
> 标注约定：【源】= 参照源头（原始论文/代码库），均给出可对照的文件与行号。
> 行号基于 2026-09-18 版本，会漂移——用各节的 grep 锚点重新定位。
> 本文档不重复 `reference/ARCHITECTURE_DIFFUSION.md`（文件级地图）的内容，
> 而是给出**阅读顺序 + 每一步该看什么 + 参照源头在哪**。

---

## 第 0 步：先读文档建立地图（1-2 小时，不要跳过）

| 顺序 | 文档 | 作用 |
|---|---|---|
| 1 | `PROJECT_STATUS.md` §1-§2 | 问题定义 + 术语表（hole/pass1/pass2/anchor/沙沙感…） |
| 2 | `PROJECT_STATUS.md` §6.3 | **技术红线**（7 条，违反即重蹈覆辙） |
| 3 | `PROJECT_STATUS.md` §7 | 当前三臂状态、方法论更正、根因分析（最新） |
| 4 | `reference/ORIG_FAITHFUL_PORT_SPEC.md` | 移植规范（2026-08-23 老师定稿） |
| 5 | `reference/ARCHITECTURE_DIFFUSION.md` | 文件级架构地图 |
| 6 | `../gpt.md` | 老师的监督解耦处方：§4 目标矛盾、§9-13 处方、§27 消融顺序 |
| 7 | `PREREG_V18W6.md` + 附录 | 当前 wave 的预注册与中期判定 |

**读代码前必须接受的一个事实**（§7.7）：当前系统的洞区质感问题的根因是
结构性的（监督目标分布错位 + D 打点偏离），读代码时请带着这个问题去核对。

---

## 第 1 步：启动链路（30 分钟）

```
scripts/v19_w6a.sh                          ← 配方、剂量、设计理由都在头部注释
  └→ scripts/train_inpainting_diffusion.py   ← Hydra 入口；SEED=2107（同原版）
       └→ training/coach_inpainting_diffusion.py  ← Coach 类（~2760 行，主战场）
            └→ configs/train_inpainting_diffusion.yaml  ← 全部配置键
```

- `v19_w6a.sh` 头注释写明了：gpt.md 消融第 5 步、orig WarpGAN adv×10、
  InvSR adv×0.1 的剂量谱系——**为什么是 ×1.0**。
- 验证方法：`ps -ef | grep train_inpainting` 看运行中进程的命令行 = sh 的展开。
- ⚠ 注意：coach 顶部 docstring 的 "v16 running recipe" 是历史配方，
  **当前配方以 sh 脚本 + 运行目录下保存的 `config.yaml` 为准**。

---

## 第 2 步：模块装配（`Coach.__init__`，grep `def __init__`）

顶部 docstring（L1-124）自带 ARCHITECTURE MAP，按冻结/可训练分类：
- 冻结：SD1.5 VAE/UNet、CLIP 空提示、GOAE(W+ 渲染)、warper
- 可训练：BrushNet（官方 inpainting ckpt 初始化，唯一大模块）、全部注意力
  processor、WProjModel(W+→token)、[InvSR 判别器]

关键方法：
- `_install_attention_processors`（L554）：两个注入位（attn1 参考纹理 /
  attn2 W+ 身份），门控残差设计——零门 = 原版 SD。
- `_route_trainable_parameters`（L595）+ `_build_optimizer`（L640）。

---

## 第 3 步：数据与几何（半天，理解 anchor 是关键）

- `_parse_real_batch` / `_parse_synth_batch`（L792/L807）：真实/合成批字段。
- **`_build_novel_view`（L1548）——全项目最重要的一段**：构造 pass1 的
  condition/mask/anchor。anchor = 照片证据与渲染按深度一致性混合
  （洞区以镜像重投影 x_mirror 为主）。W6 臂在此处有 `anchor_median=5`。
- `flip_yaw` / `get_mirror_c`（L775/L784）：镜像视角相机参数。
- 【源】`utils/warp/`（Splatting.py、splatting_ext.py、Warper.py 等）
  **全部来自原始 WarpGAN**，与 `reference/orig_WarpGAN-main/` 同名文件可逐行对照。
- ⚠ §7.7 根因 1：anchor 洞区统计 ≠ 照片统计——所有参照损失的靶。

---

## 第 4 步：前向图（`_diffusion_forward`，L906-999）

单一咽喉点，读它能看懂 80% 的数据流：
```
condition_img → VAE latents；+下采样 mask → 5ch BrushNet 条件
目标图 → VAE → 加噪(t~U[0,1000)) → noisy_latents
参考图 → 同一冻结 UNet 上 no-grad t=0 pass → 逐层 attn1 bank（_write_bank）
brushnet(noisy, cond) → down/mid/up add-samples
denoising_unet(noisy, add_samples, cak{wplus_features, reference_features}) → eps_pred
```
- 【源】参考特征 bank = AnimateDiff mutual-self-attention 范式；
  参照实现保存在 `models/referencenet/mutual_self_attention.py`（不 import，
  仅作范式源）。详见 `models/referencenet/attention_processor.py` 头注记。
- HF 高频通道（`_compute_hf_map` L1604，AnyDoor 式 sobel）W5 已验证无效，
  当前臂未启用（`hf_enable=False`）。

---

## 第 5 步：损失地图（1 天，最重要的一节）

| 损失 | 位置（grep 锚点） | 参照源头 | 当前状态 |
|---|---|---|---|
| ε-MSE | `_eps_mse` | SD 标准 | **开**（全 t，全帧） |
| dual-band ε | `_eps_mse_dual_band` | 项目自研（v6） | pass1 用 |
| 低 t x0 像素族 | `_low_t_losses` | v14 设计 | 关（weight=0） |
| x0 分布族(旧) | `_x0_losses` | v18 预 InvSR | **关**（W1 教训：全 t 毁采样） |
| **x0 InvSR 族** | `_x0_losses_invsr` | 【源】InvSR (CVPR'25, 2412.09013) `trainer.py` backward_step L1239-1312；t 域 [100,250]→我们 [0,300]（实测边界） | **开**（t≤300） |
| GAN/FM(旧) | `_gan_fm_g_loss` | 【源】orig WarpGAN coach L1169-1198 忠实移植 | 关（weight=0） |
| LatentLPIPS | `models/invsr_latent_lpips/` | 【源】Diffusion2GAN (ICLR'24) LatentLPIPS；InvSR 所用；ckpt=vgg16_sdturbo_lpips.pth | W6 臂开 |

读 `_x0_losses_invsr` 时注意三点（都有行内注释）：
1. `x0_hat` 单步估计 + ±latent_bound 截断（InvSR L1506-09）；
2. min-SNR 对齐（分布族乘 ab_t，v18 自研）；
3. D 配对入队 `_disc_queue_invsr.append`（L1349）——**§7.7 发现的缺口就在这**：
   配对只有 `real_p2`（real=照片，打在 pass2）和 `synth_x0`（real=渲染，
   违反红线 #3），没有 pass1 洞区配对。

---

## 第 6 步：判别器与更新

- `models/saicinpainting/training/modules/unet_discriminator.py`：
  【源】InvSR UNet2DConditionDiscriminator（106M，时间步条件），头注记含
  逐行对照。参照源在场：`/tmp/invsr/InvSR-master/`。
- `_discriminator_step_invsr`（L1437）：hinge（【源】InvSR trainer L1612+），
  每步更新 D，G 在 dis_warmup 后才见 ldis。
- `_discriminator_step`（L1401）：旧 v14 路径（R1 惩罚，orig 移植），当前关。

---

## 第 7 步：训练循环与审计

- `train()`（L1961）：real/synth 1:1 交替（【源】orig coach L489-560 形制）；
  D 队列在 G 步积累、独立 D 步消费。
- `_smoke_audit`（L2184）+ `_v10/_v12/_v13/_v15_audit`：历代事故的免疫系统，
  每个 wave 起跑必须过冒烟门（历史教训见 PROJECT_HISTORY）。
- `resume()`（L2558）：⚠ 有一个 v19 修复（未提交 diff，10 行）——InvSR 模式
  resume 时 D 加载门从 use_gan_fm 放宽到 `_invsr()`，此前 W3-P 90K 续训
  曾把 106M D 静默重置为随机初始化。

---

## 第 8 步：推理与验证

- `_sample_novel`（L2633）：50 步手写 DDIM（eta=0）、`seed=42` 固定；
  `t_start>0` 切换 SDEdit 模式（W4 已裁决弃用，当前臂 t_start=0 纯噪声全链）。
- `validate()`（L2724）：**6 格面板 `[x | y_hat_novel | cond | mask | anchor | gen]`**，
  最右格 = 生成输出。⚠ val 批次在 3 个固定样本间轮换（洞像素数
  112170/166416/204907），跨步比较必须按样本分组（§7.2）。

---

## 参照源头总表（机制 ↔ 我们的位置 ↔ 源）

| 我们的位置 | 机制 | 参照源（在场路径） |
|---|---|---|
| `utils/warp/*` | 前向/反向 splatting warp | `reference/orig_WarpGAN-main/` 同名文件 |
| `_build_novel_view` | anchor 混合构造 | orig WarpGAN 的 novel-view 训练组织 |
| `_gan_fm_g_loss` / `_discriminator_step` | FM+adv（关） | orig WarpGAN `training/coach_inpainting_static.py` L1169-1240 |
| `_x0_losses_invsr` / `_discriminator_step_invsr` | 潜空间低 t 分布监督 + hinge | `/tmp/invsr/InvSR-master/trainer.py` L1239-1312, L1612+ |
| `models/.../unet_discriminator.py` | UNet-D 106M | InvSR `src/diffusers/.../unet_2d_condition_discriminator.py` |
| `models/invsr_latent_lpips/` | LatentLPIPS | Diffusion2GAN（ICLR'24），InvSR 沿用 |
| `models/referencenet/attention_processor.py` | 参考纹理 attn1 + W+ attn2 | AnimateDiff mutual-self-attn 范式（`mutual_self_attention.py` 在场） |
| BrushNet 5ch 条件 | `_diffusion_forward` | BrushNet 官方 inpainting ckpt |
| `../gpt.md` | 监督解耦处方（§9-13, §27） | 处方本身；**§11-12 masked patch D 尚未实现（=W7 提案）** |

---

## 自检清单（读完应能不看代码回答）

1. pass1 和 pass2 的输入条件分别是什么？为什么 pass2 是"唯一照片监督"？
2. anchor 在洞区由什么构成？为什么它的统计≠照片？（§7.7 层1）
3. InvSR 族损失在哪些 tag 上生效？D 的配对各以什么为 real？（§7.7 层2）
4. 为什么 adv×0.1 vs ×1.0 对洞区脏度无差？
5. val 面板第 6 格是什么？为什么 val L1 会"平"？（§7.2-7.3）
6. 当前三臂配方差异是什么？（config diff 一行：`x0.w_ldis`）

## 已知雷区（读代码时容易误判）

- `configs/train_inpainting_diffusion.yaml` 里 `x0.d_real: paired_target`
  是**死键**（v18 P1 遗留，代码无消费者）——D 的 real 由代码路径决定。
- coach 顶部 docstring 的损失权重是 v16 历史配方，不是运行配方。
- `_x0_losses`（旧族）与 `_x0_losses_invsr`（现役）容易混淆——看 `_invsr()` 谓词。
- 日志里 `real_p2_hit=0` 表示该步 t>300 未命中分布监督（正常，非故障）。

