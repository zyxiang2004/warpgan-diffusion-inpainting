# 如何把本仓库推送到 GitHub（2026-09-18 W7 全貌版）

本训练服务器出网为白名单模式（GitHub / Gitee / 各加速镜像全部不通，仅 pypi 镜像等可用），
因此**推送必须在能访问 GitHub 的设备上完成**。仓库已打好完整 commit，并生成单文件快照
bundle（含全部历史与标签，可用 git 直接克隆）。

**注意：git 仓库根 = 本目录**（包含 `warpgan-diffusion-inpainting-main/`、`AnyDoor/`、
`PVA-CelebAHQ-IDI-master/`、`dreambooth/`、三份研究报告 md 与本指南）。

## 导出文件（服务器上）

- **bundle**：`/home/xzy/warpgan-diffusiong-inpainting/warpgan-repo-20260918-full.bundle`
  （md5/大小/HEAD 见同目录 `BUNDLE_INFO_20260918.txt`；该 txt 为本地核对用，不入库）
- 旧快照（已删除，被本文件取代）：`warpgan-repo.bundle`（9/15）、
  `main/warpgan-repo-w7.bundle`（9/18 早）

## 步骤（在本地电脑操作）

1. 在 GitHub 网页新建空仓库（如 `warpgan-diffusion-inpainting`），**不要**勾选
   "Add a README"。
2. 下载 bundle 到本地（VS Code Remote-SSH 右键 Download…，或
   `scp xzy@<server>:/home/xzy/warpgan-diffusiong-inpainting/warpgan-repo-20260918-full.bundle .`）。
3. 执行：

   ```bash
   git clone warpgan-repo-20260918-full.bundle warpgan-diffusion-inpainting
   cd warpgan-diffusion-inpainting
   git remote set-url origin https://github.com/<你的用户名>/warpgan-diffusion-inpainting.git
   git push -u origin main --tags
   ```

## 已纳入（本次 9/18 全貌清单，逐项核实）

- **全部源码**：`warpgan-diffusion-inpainting-main/` 下 training（含 W7 洞区照片 D 修改）、
  models、scripts（v16-v19 全部训练/评估/烟雾脚本 + 新增 `calib_user_calibration.py`、
  `calib_arm_sync_test.py`）、configs、datasets、utils、criteria。
- **全部项目文档**：`PROJECT_STATUS.md`（§7.1-7.13 振荡诊断→W7 全链路）、`PREREG_*.md`、
  `CODE_READING_GUIDE.md`、`docs_archive/`、本指南。
- **研究报告（根级）**：`gpt.md`、`DMDX_ICCV2025.md`、`ReSem-Face_arXiv2026.md`
  （文档里 `../gpt.md` 类相对引用在克隆中可原样解析）。
- **参考项目（根级）**：`AnyDoor/`、`PVA-CelebAHQ-IDI-master/`（含 thirdparty）、
  `dreambooth/`；`main/reference/` 下 `orig_WarpGAN-main/`（原版 = W7 理论出处）、
  `models_lib/`（BrushNet 等）、**`InvSR-master/`（9/18 新入库，原在 /tmp 易失）**。
- **权重**（唯一入库权重）：`main/weights/vgg16_sdturbo_lpips.pth`（57MB，InvSR 潜空间 LPIPS）。
- **训练定量记录（9/18 修复两处历史遗漏后首度完整入库）**：
  - `main/train_logs/*.log` —— 此前被 Python 模板的 `*.log` 全局规则误伤，从未入库；
  - `main/experiments/` 全部文本记录（83 文件 config*.yaml/log/md）—— 此前被根级
    目录级忽略挡住，细粒度文本规则从未生效。
- **用户校准记录**：`main/user_calibration_20260918/MANIFEST.md`（172 面板全指标表，
  §7.9/7.10 证据；图片本体留本地）。

## 未纳入（均为有意排除，体积/来源原因）

| 内容 | 体积 | 说明 |
|---|---|---|
| `main/experiments/` 二进制（checkpoints/val 图/tb events） | 257GB+ | 仅存训练服务器；续训需另行 scp |
| `main/data/`（FFHQ-EG3D / SynthData / celeba-hq） | ~TB | 数据集 |
| `main/pretrained_models/`（SD1.5 / BrushNet / LaMa 等） | ~10GB | 官方可下载 |
| `main/user_calibration_20260918/` 图片 | 158M | 可由 MANIFEST + val PNG 复现 |
| `*.zip / *.bundle / BUNDLE_INFO_*.txt / _backup_pre_mirror_anchor/` | — | 传输产物与旧备份（gitignore） |

## 打 bundle 时刻的训练状态（2026-09-18 晚）

- **GPU0 W6-A**（adv×1.0）：~86K/300K 运行中
- **GPU1 W6-B**（adv×0.1）：~89K/300K 运行中（= W7 的同血统对照，80K 分叉）
- **GPU2 W7**：W6-B@80K 分叉 + 洞区照片 D（原版 real-batch novel 分支的潜空间复原），
  烟雾通过（holeD 配对触发、D 分离 d_real≈+0.9 / d_fake≈-1.2、0 新 Traceback、14.7GB）
- **W3-P**：已停于 174,950 步（用户指令释放 GPU2；170K checkpoint 在盘可续）
- 基线 tag：`v19-w6-baseline`（= W6 双臂运行代码的冻结点）
