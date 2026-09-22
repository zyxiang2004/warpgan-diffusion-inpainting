W6-A / W6-B 全套代码备份（2026-09-22）
====================================

结论：W6-A/B 的运行基线代码已冻结在 git 提交 40c69df（2026-09-18，
提交信息明确 "code frozen as W6-A/B run base before W7"）。
本目录是它的实体化备份 + 冻结配置 + 复现说明。

内容
----
git_commit_40c69df/      W6 运行基线的完整源码树（git archive 40c69df 提取，
                        含 training/ scripts/ configs/ models/ datasets/ utils/
                        reference/ 等 —— 与 2026-09-15 W6 启动时代一致）
w6_launch_scripts/       v19_w6a.sh / v19_w6b.sh（取自 40c69df，非当前版本）
w6a_frozen_config.yaml   W6-A 实验目录内启动时冻结的完整配置（落盘原样）
w6b_frozen_config.yaml   W6-B 同上
w6a_config_resume.yaml   W6-A 目录内的 resume 覆盖记录（若在）
coach_diff_40c69df_to_w10.patch
                        40c69df → 当前(W10时代) coach 全部差异。
                        关键保证：legacy anchor median 路径数学未变，
                        W10 只是把旧 `if anchor_median>0:` 换成
                        `if anchor_median>0 and anchor_clean=='selective_hole':`
                        分支 + 新增配置项；anchor_clean=whole（默认）时
                        当前代码与 40c69df 训练数学等价。

复现 W6-A 的方法
----------------
1. 代码：本目录 git_commit_40c69df/（或 git checkout 40c69df）
2. 配置：w6a_frozen_config.yaml（或用 w6_launch_scripts/v19_w6a.sh 重放）
3. 数据：./data/FFHQ-EG3D_all_static_rebalanced 与
   ./data/SynthData100000_rebalanced（未包含在备份，位置见 config paths）
4. 权重：./pretrained_models/brushnet、sd1.5、VAE、wplus_stage_c（同上）
5. W6-B 同理，用 w6b_* 与 v19_w6b.sh

已验证（2026-09-22）
--------------------
- 40c69df 包含：coach、train 脚本、v19_w6a/b.sh、主配置、数据集类、
  warp 算子 —— 逐文件 git show OK
- 两份冻结 config.yaml 已被 git 跟踪（2/2）
- tar.gz 完整性：见同目录 .tar.gz（sha256 于生成时输出）
- 注意：备份与本仓库同盘；远程 origin(github) 存在但本次未推送，
  如需异地容灾请 git push（PUSH_GUIDE.md 有流程）。
