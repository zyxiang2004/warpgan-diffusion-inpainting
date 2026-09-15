# v18w1 运行事件记录 — 2026-09-10 GPU 故障与恢复

## 事件

- 物理卡 PCI **0000:3b:00.0**（CUDA 设备 1，用户编号"GPU1"）故障：
  `dmesg: NVRM: Xid (PCI:0000:3b:00): 79, GPU has fallen off the bus`。
- 影响：
  1. **CTRL 臂**（CUDA_VISIBLE_DEVICES=1）：挂在 step 1500（05:53 停更），
     无 checkpoint（<10000），进程已 kill -9 清理，需从零重启；
  2. **FULL 臂**（设备 2）/ **STRUCT 臂**（设备 3）：既有 CUDA 上下文不受
     影响，持续健康推进（09:29 时 6800 / 9400 步，零 Traceback）；
  3. **驱动被楔死**：三张卡的新 CUDA 上下文创建全部不可中断挂死
     （GPU0/2/3 探测进程连 timeout 的 SIGTERM 都杀不掉，D 状态）。
     这是 Xid 79 的典型行为：已打开的设备句柄继续工作，新 open() 卡死。
- 结论：**在服务器重启（或卸载重载 nvidia 模块，需先停全部任务）之前，
  无法在任何卡上启动任何新 CUDA 进程**——包括把 CTRL 挪到 GPU0。

## 恢复方案（经用户选择后执行）

- 推荐：等 STRUCT 到 10000 步保存首个 checkpoint（预计 ~09:55）后重启
  服务器，之后运行 `scripts/v18w1_recover_after_reboot.sh`：
  - 排除坏卡（按 PCI 总线号 0000:3b:00.0 识别，而非可能重编号的序号）；
  - 对候选卡逐个做 CUDA 探测，只把通过探测的卡分配给训练；
  - STRUCT 从 checkpoint resume；FULL 若已有 checkpoint 则 resume、否则
    从零；CTRL 从零（原 1500 步无保存点，损失 ~50 分钟）；
  - 重启健康监视器。
- 备选 A：等 FULL 也在 ~12:45 存到 10000 再重启（多等 3.2h，多保留
  FULL 的 10K 步；风险=楔死驱动上多运行 3 小时）。
- 备选 B：立即重启（STRUCT 损失 ~9.5K 步 ≈3.5h、FULL ~7K ≈3h；三臂
  同步从零，最干净但浪费最多）。

## 可安全重启信号

`train_logs/reboot_ready.status`（由 `scripts/v18w1_reboot_ready.sh` 每
5 分钟更新）：显示 STRUCT/FULL 是否已存 checkpoint。STRUCT=ready 即可
按推荐方案重启；FULL=ready 则备选 A 也就绪。

## 对实验有效性的影响评估

- CTRL 从零重启：无偏（三臂本就从零、同数据同代码同种子；CTRL 只是
  晚 ~45 小时完成，终点判据在 50K，不受影响）。
- STRUCT 若 resume：优化器/调度器状态随 ckpt 保存（v17-prime 已验证
  resume 完整性：brushnet/D/optimizer 状态恢复，0 Traceback）。
- 唯一时间线变化：三臂完成时间不再同步（预计 STRUCT 最早、CTRL 最晚，
  相差 ~1 天）。终评在全部到达 50K 后统一进行。
