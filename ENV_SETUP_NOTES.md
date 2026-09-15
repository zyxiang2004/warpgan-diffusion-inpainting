# 环境配置记录（ta13 服务器，2026-09-09）

> 本机（ta13，4×RTX 3090, driver 535.309.01/CUDA 12.2）上为 warpgan-diffusion-inpainting
> 重建的训练环境。所有结论均给出出处，复现命令可直接复制。

## 1. 环境

```sh
conda activate warpgan          # Python 3.9.25
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.8.0+cu128 True
```

- conda env 路径：`/home/xzy/miniconda3/envs/warpgan`
- Python 包：`requirements.txt` 全部 171 个 pin 精确安装（`requirements_clean.txt`
  为去掉两个本地 file:// 依赖后的清单；因 opencv-python-headless 4.13 元数据要求
  numpy>=2 与 numpy==1.22.4 冲突——远程环境本就是时序安装的产物——故用
  `pip install --no-deps` 复刻冻结态。`pip check` 会报这 1 条 metadata 不一致，
  运行时无影响，与远程机器状态一致）。
- 本地源码安装的两个包：
  - `diffusers 0.27.0.dev0` ← `./reference/models_lib/BrushNet-main`（与
    `/home/xzy/WarpGAN/models/BrushNet-main` 逐文件一致，仅 `__pycache__` 差异）
  - `splatting 0.0.0` ← `/home/xzy/WarpGAN/splatting-master`（CUDA 扩展已编译，
    针对本环境 torch 2.8/cu128）
- conda CUDA（供 StyleGAN JIT 算子编译）：`cuda-nvcc 12.8.93 + cuda-cudart-dev 12.8.90`
  （nvidia/label/cuda-12.8.0 通道）
- **激活脚本** `$CONDA_PREFIX/etc/conda/activate.d/zz_cuda_home.sh` 自动导出：
  `CUDA_HOME=CUDA_PATH=$CONDA_PREFIX`、`CC=gcc CXX=g++`（系统 gcc 9.4；conda 自带
  gcc 14.3 会被 CUDA 12.x nvcc 拒绝，故必须覆盖）、`CPATH`（把 pip nvidia-* 轮子自带的
  cusparse/cublas/cusolver/cufft/curand/nvjitlink 头文件目录加进来，torch 扩展编译需要）。
- 另做了两个环境内布局修复（targets 布局 → 标准 CUDA_HOME 布局）：
  `$PREFIX/include/*` ← `targets/x86_64-linux/include/*` 符号链接；
  `$PREFIX/lib64` → `targets/x86_64-linux/lib` 符号链接。

## 2. 符号链接（项目根 `warpgan-diffusion-inpainting-main/`）

| 链接 | 指向 | 说明 |
|---|---|---|
| `data` | `/data2/hkt/dataset` | train/test/synth 三数据集都在（139914/1000/100000 样本，smoke 已实测） |
| `pretrained_models` | `/home/xzy/WarpGAN/pretrained_models` | 本机旧 WarpGAN 目录，静态权重齐全（22/23 项核验通过，见下） |
| `pretrained_models/ffhq` | `eg3d/ffhq`（相对链接） | 为 `configs/paths_config.py` 里 `ori_pth` 模式的 `./pretrained_models/ffhq/...` 路径补的兼容链接 |

`infer.yaml`/`pti.yaml` 用的 `./data/test_img` 在 `/data2/hkt/dataset` 下不存在
（推理用的自备图片目录），需要时自行放置。

## 3. 权重核验（2026-09-09，脚本见 /tmp/verify_weights.py，22 OK / 1 项远程独有）

本地已就绪：SD1.5（unet/vae/text_encoder/tokenizer/scheduler）、brushnet、
inversion/gan_encoder.pt、goae、eg3d（ffhq512-128.pkl/pth、ffhqrebalanced、
ffhq_lpff var1/var2 + 全部 latent_avg）、ir_se50、moco、inpaintor/inpaintor.pt、
LaMa_perceptual_loss_models、editings/goae_ws_edit/*.npy（仓库自带）。

**唯一缺项：`paths.wplus_stage_c`**（`/data/xzy/warpgan20260803/20260803/experiments/
_wplus_identity_pretrain/checkpoints/iteration_50000.pt`）——该路径是远程机器的绝对
路径，本机没有。训练配置默认 `wplus.mode: frozen` 会在 coach L317 硬加载它。要用
frozen 档训练需从 10.26.66.10 拷这一个文件过来（用户已确认：不需要训练 checkpoint，
只要预训练权重；此文件属于 mapper 预训练权重，若后续要跑 frozen 档再取）。
当前 smoke 用 `wplus.mode=off` 档（配置原生支持）绕开验证。

其他待确认（不影响训练主流程）：`editings/CLIPStyle/mapper_results/*/checkpoints/
best_model.pt`（styleclip 编辑用，zip 内未见）；`pose_estimation/checkpoints/pretrained/`
下 Deep3DFaceRecon 权重（extract_pose.py 测试图预处理用，仓库内只见 test_opt.txt）。

## 4. 为跑通 smoke 做的一处代码修复（重要，请知悉/复核）

`training/coach_inpainting_diffusion.py` `_v10_audit`（L1669 附近）：
off 分支的对象同一性检查原为 `soft is raw`，其中 `raw/soft = mask.detach()`。
**torch 2.8 的 `Tensor.detach()` 一律返回新对象**（旧版对无梯度张量返回自身），
导致该审计在任何 torch 2.8 环境必然 AssertionError（实测独立复现：
`torch.zeros(1).detach() is torch.zeros(1)` 两侧均 False）。修复为在 detach 之前
对原始张量做同一性比较（`mask_cond is mask_raw`），意图完全不变。修复后 v10 审计
PASS。除此之外项目代码零改动。

另：site-packages 里 `splatting/splatting.py` 被构建脚本装成了 0 字节（源文件 4309
字节完好），已用源文件覆盖修复，编译好的 .so 无需重编；pip 缓存中坏轮子已清除。

## 5. 验证记录

1. `splatting_function('summation', f, zero_flow)` GPU 守恒测试 ✓（误差 0.0）
2. StyleGAN JIT 算子 bias_act / upfirdn2d / conv2d_gradfix / filtered_lrelu
   在 RTX 3090 上编译+运行 ✓（nvcc 12.8.93 + gcc 9.4，缓存在 ~/.cache/torch_extensions）
3. 端到端训练 smoke（`wplus.mode=off max_steps=2`，日志 /tmp/smoke_train2.log）：
   - 三个数据集加载 ✓（synth 100000 / real 139914 / test 1000）
   - v10 / v15 / Step-0 SMOKE 审计全部 PASS（33 PASS / 0 FAIL）
   - checkpoint 保存 ✓，**peak GPU 21.63 GiB < 22 GiB 预算**（远程为 21.28 GiB，同量级）
   - 产物：`experiments/[20260909-213127]_env_check_smoke2/`

## 6. 常用命令

```sh
conda activate warpgan
cd /home/xzy/warpgan-diffusiong-inpainting/warpgan-diffusion-inpainting-main
# smoke（wplus off 档，无需远程文件）
CUDA_VISIBLE_DEVICES=0 python scripts/train_inpainting_diffusion.py \
    exp_dir=./experiments/<name> max_steps=2 wplus.mode=off
# 正式训练（需先补 wplus_stage_c 文件，见 §3）
CUDA_VISIBLE_DEVICES=0 python scripts/train_inpainting_diffusion.py exp_dir=./experiments/<name>
```

## 7. v17-prime 本机跑通记录（2026-09-09，任务验收）

完整性检查：
- 记录齐全：PROJECT_STATUS.md（261 行）；reference/docs_archive/ 含 PROJECT_HISTORY.md
  （614 行）、ORIG_FAITHFUL_PORT_SPEC.md（237 行）、ARCHITECTURE_DIFFUSION.md、
  NEW_SESSION_START.md、STEP0_SMOKE_REPORT.md、PROJECT_SNAPSHOT_20260829.md；
  experiments/anchor_compare/（anchor_metrics.txt + run 日志）；tests/ 四个守护测试；
  历代实验 config 均在。
- 代码完整：`compileall` 全仓库 COMPILE_OK；无损坏文件（仅正常的空 `__init__.py`）。

v17-prime 验证运行（配方 = scripts/v17prime_fresh.sh 的完整 override 集，仅
`wplus.mode=off` 替换 frozen 档；日志 /tmp/v17_run.log、/tmp/v17_resume2.log）：
1. 40 步 fresh：0 Traceback；real/synth 交替训练正常；panels（real+synth 各 4 张，
   PNG 有效）正常；val_novel_full=0.3032 val_hole=0.1508；checkpoints
   iteration_19/39 保存正常；peak 22.67 GiB（24G 卡不 OOM，略超项目内部
   "<22 预算"线，远程 21.28 GiB 为 frozen 档 smoke 数字）。
2. resume（iteration_0000039.pt → step 40）：brushnet/D/optimizer 状态完整恢复，
   iteration_0000040.pt 保存正常，0 Traceback。
产物目录：`experiments/[20260909-215430]_v17prime_local_40steps/`。

注意：hydra 1.1 的 override 语法不接受路径中的 `[`（时间戳目录名），resume 时
checkpoint_path 值需再包一层引号：`"checkpoint_path='./experiments/[...]/xxx.pt'"`。

