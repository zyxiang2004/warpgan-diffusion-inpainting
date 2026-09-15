# WarpGAN diffusion port — file map & call graph (v16, 2026-09-08)

本导览描述 Step-3 diffusion 训练管线的文件层次与互相调用。代码语义零改动的
注释整理产物；运行配方以 `scripts/v16_resume.sh` 的 CLI 覆盖为准。

## 1. 文件层次（训练链实际 import 的文件加 ★）

```
warpgan_orig/WarpGAN-main/
├── scripts/
│   ├── ★ train_inpainting_diffusion.py   # Hydra 入口：seed→exp_dir→Coach→train()
│   ├──   eval_step2_ab.py                # A/B 评测（100K 队列自动调用）
│   └──   eval_bank_gate.py / eval_frozen_ablation.py / ...  # 各代诊断脚本
├── configs/
│   └── ★ train_inpainting_diffusion.yaml # 默认值；v16 配方=CLI 覆盖
├── training/
│   └── ★ coach_inpainting_diffusion.py   # 核心 Coach（模块装配/双 pass/损失/守护）
├── models/
│   ├── ★ referencenet/attention_processor.py   # attn1 参考纹理 / attn2 W+ 注入
│   ├──   referencenet/mutual_self_attention.py # 范式参考（不被 import）
│   ├── ★ mapper/w_proj.py                 # WProjModel：W+ [14,512]→[18,768]
│   ├── ★ wplusnet.py                      # WplusNet（GOAE 冻结：codes/渲染/深度）
│   ├── ★ saicinpainting/...               # set_requires_grad / ResNetPL / D 构建
│   └──   BrushNet-main/                   # BrushNetModel（第三方，不动）
├── datasets/
│   ├── ★ dataset_inpainting_static.py         # real 批（FFHQ+EG3D 静态对）
│   └── ★ dataset_inpainting_synth_static.py   # synth 批（EG3D 双视角渲染对）
├── utils/
│   ├── ★ warp/Splatting.py          # Warper：前向 splat warp
│   ├── ★ warp/splatting_ext.py      # WarperExt：inverse_warp（反投）
│   └──   warp/{Warper,WarperPytorch,render_warp,rotate,warp_cam}.py  # 原版遗留，训练未用
└── docs/  （仓库根 docs/ 下为项目史与理论记录）
```

外层（仓库根）：`scripts/v16_resume.sh / v16_100k_queue.sh / v16_100k_watchdog.sh`
= 训练恢复 / 100K 停止+自动 A/B / 存活看门狗（持久位置，重启不丢）。

## 2. 调用图（一次训练步）

```
train_inpainting_diffusion.py (hydra)
  └─ Coach(opts)                       # coach_inpainting_diffusion.py
       ├─ 冻结装配: vae / denoising_unet / CLIP 空提示 / gan(WplusNet)
       │             warper(Splatting) + warper_ext(splatting_ext)
       ├─ 可训装配: brushnet(BrushNetModel) + w_mapper(WProjModel)
       │             _install_attention_processors()  ──► attention_processor.py
       │               （每个 SD attn 层换成 ReferenceAttentionProcessor）
       └─ train()
            ├─ real 批 ─► _forward_real
            │    ├─ _build_novel_view      （warper + warper_ext：cond1/anchor）
            │    ├─ _diffusion_forward(pass1)
            │    │    ├─ _extract_reference_features → _write_bank  (attn1 bank)
            │    │    ├─ _wplus_tokens → w_mapper(codes)            (attn2 注入)
            │    │    └─ brushnet + denoising_unet(+add_samples, cak) → eps
            │    ├─ _eps_mse_dual_band      （pass1 结构锚）
            │    ├─ warper.forward_warp(cond1) → pass2 条件（warp 两次）
            │    ├─ _diffusion_forward(pass2)
            │    └─ _low_t_losses           （t<200：x0 patch L1×2 像素回流）
            └─ synth 批 ─► _forward_synth   （ε 结构段 only；W+ 仅在此更新）
```

## 3. 数据流要点（v16）

- **条件**（BrushNet 5ch）：可见区=真照片前向 warp；洞=GAN inversion 渲染；
  接缝 erode3+blur21 软化（原版 process_mask 语义）。
- **参考**（attn1 bank，AnimateDiff 范式）：real pass1 主位=x_mirror 照片、
  副位=novel 渲染；pass2 主位=y_hat；synth=y_hat。参考图跑冻结主 UNet(t=0)
  逐层存 norm_hidden_states，denoising 同层检索 K/V 注入。
- **身份**（attn2 cross）：W+ codes→w_mapper→18×768 tokens 与空文本并行，
  固定 tanh(3) 门；只训 mapper（s1）或再解锁 QKV（s2）；仅 synth 更新。
- **监督**：pass1 dual-band 结构锚（照片证据全带/盲区低带 vs 渲染锚）；
  pass2 ε-MSE vs x + 低 t 像素回流（对齐照片——原版唯一照片质感通路同位）；
  synth 仅 ε 结构段（渲染质感=油漆红线，权重 0）。

## 4. 运行与守护

- 训练（从根目录）：`scripts/v16_resume.sh`（自动取最新 ckpt，目标 100K）。
- 100K 停止队列：等 `iteration_0099999.pt` → 停训 → `eval_step2_ab.py`。
- 看门狗：训练死且 100K 未到 → NFS 可用则自动 resume。
- 日志：`train_logs/step3_v16_200k_v5.log`；实验目录
  `experiments/train_inpainting_diffusion/v16_run/`。
