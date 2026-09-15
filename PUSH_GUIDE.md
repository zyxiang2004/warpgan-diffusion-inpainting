# 如何把本仓库推送到 GitHub

本训练服务器出网为白名单模式（GitHub / Gitee / 各加速镜像全部不通，仅 pypi 镜像等可用），
因此**推送必须在能访问 GitHub 的设备上完成**。仓库已在此打好完整 commit，并生成了
单文件快照 `warpgan-repo.bundle`（含全部历史，可用 git 直接克隆）。

## 步骤（在你本地电脑操作）

1. 在 GitHub 网页上新建空仓库（例如 `warpgan-diffusion-inpainting`），**不要**勾选
   "Add a README"（保持空仓库，避免首次推送冲突）。

2. 把服务器上的 bundle 文件下载到本地，二选一：
   - VS Code Remote-SSH：在资源管理器右键
     `/home/xzy/warpgan-diffusiong-inpainting/warpgan-repo.bundle` → Download…
   - 或命令行：`scp xzy@<服务器地址>:/home/xzy/warpgan-diffusiong-inpainting/warpgan-repo.bundle .`

3. 在本地执行：

   ```bash
   git clone warpgan-repo.bundle warpgan-diffusion-inpainting
   cd warpgan-diffusion-inpainting
   git remote set-url origin https://github.com/<你的用户名>/warpgan-diffusion-inpainting.git
   git push -u origin main
   ```

## 仓库内容与范围说明

- 已纳入：全部源码（`warpgan-diffusion-inpainting-main/` 下 models / training / scripts /
  configs / datasets / criteria 等）、三份研究报告（gpt.md、DMDX、ReSem）、预注册文档
  （PREREG_*.md）、参考实现（`reference/orig_WarpGAN-main/` 等）、参考项目
  （AnyDoor / PVA / dreambooth）、LPIPS 权重（57MB，
  `warpgan-diffusion-inpainting-main/weights/vgg16_sdturbo_lpips.pth`）、训练日志快照
  （train_logs/）。
- 未纳入（.gitignore 排除）：`experiments/`（257GB checkpoint 与验证图，仅存于训练服务器）、
  `*.zip` 压缩包、旧备份目录、Python 缓存。
- 服务器上的训练 checkpoint（续训必需）在
  `warpgan-diffusion-inpainting-main/experiments/train_inpainting_diffusion/*/checkpoints/`，
  如需在其他设备续训，需另行传输（scp / 网盘）。

## 当前训练状态（打 bundle 时刻）

- GPU 2：W3-P 基线从 90K 续训至 300K（v18w3 配方，无改动）
- GPU 0：W6-A = W3-P 配方 + anchor_median=5 + 真潜空间 LPIPS + 对抗权重 ×1.0（从零，300K）
- GPU 1：W6-B = 同 W6-A 但对抗 ×0.1（对照臂，从零，300K）
- 目的：300K 长训对齐原版 WarpGAN 训练时长；W6-A vs W6-B 隔离对抗剂量效应
