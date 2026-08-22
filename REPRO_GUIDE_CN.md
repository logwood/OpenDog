# Pet-ReID-IMAG 现代 Windows 复现指南

## 1. 已验证环境

本机实测环境：Windows 11、RTX 4060 Laptop 8GB、NVIDIA 驱动支持 CUDA 12.7、Python 3.12.13、PyTorch 2.11.0+cu126。PyTorch wheel 自带 CUDA 12.6 runtime，不要求安装 CUDA Toolkit 或 `nvcc`。

在工作区根目录创建等价环境：

```powershell
conda env create -f .\environment.repro.yml
conda activate pet-reid-imag-modern
```

当前机器也可直接使用已验证解释器：

```text
D:\CondaData\envs\torch312\python.exe
```

默认 cosine 验证/推理不依赖 FAISS。只有主动开启 Jaccard re-ranking 时才需要另配 FAISS。

## 2. 数据和路径处理

资产准备脚本会把 ZIP 内含 `*` 的测试文件名安全映射为 Windows 可存储的 `_` 名称，并生成 `data/test/filename_map.csv`。模型内部始终把它映射回比赛原名，因此 pair CSV、四分支特征和最终提交不会错位。

已有工作区只需检查：

```powershell
D:\CondaData\envs\torch312\python.exe .\scripts\prepare_upstream_assets.py --verify-only
```

干净工作区首次准备时去掉 `--verify-only`。预期检查结果：38,636 张训练图、4,000 张测试图、4 个 checkpoint、8 个特征矩阵、5,400/600 个互斥身份、2,000 个平衡验证对，`ready: true`。

## 3. 最小端到端检查

```powershell
.\scripts\run_modern_pipeline.ps1 -Mode Smoke
```

它使用 8 个训练身份和 64 个验证对，真实执行 ResNeSt-101 混合精度训练、CrossEntropy/Triplet/Circle 三项损失、ROC-AUC、best/final checkpoint。Ada/Ampere 等支持 BF16 的 GPU 自动使用 BF16，其他 CUDA GPU 回退 FP16+GradScaler。验证恢复路径可直接执行：

```powershell
cd .\upstream\Pet-ReID-IMAG
D:\CondaData\envs\torch312\python.exe pet_id/train_net.py `
  --config-file configs/modern_smoke.yaml --resume SOLVER.MAX_EPOCH 2
```

本次实测 checkpoint 包含模型、optimizer、cosine scheduler、warmup scheduler 和 AMP GradScaler。当前 BF16 路径无需启用缩放；恢复检查仍还原了 519 份 Adam state（step 均为 4）并从 epoch 1 继续。FP16 回退路径会通过同一 checkpointable 保存动态 scaler 状态。

## 4. 留出验证训练

四个现代配置：

| 分支 | 配置 | 8GB batch | 输入 |
|---|---|---:|---:|
| ResNeSt-101 | `modern_s101_224.yaml` | 28 | 224 |
| ResNeSt-101 | `modern_s101_256.yaml` | 28 | 256 |
| ResNeSt-101 | `modern_s101_288.yaml` | 28 | 288 |
| ResNeSt-200 | `modern_s200_224.yaml` | 28 | 224 |

采样器每个身份抽取 4 张图，因此 batch 28 包含 7 个身份；每张图可使用来自另外 6 个身份的 24 张负样本。它比仅含两个身份的 batch 8 更适合 Triplet/Circle Loss。batch 32 虽可短时运行，但最重分支会保留约 7.3–7.4GB 显存；batch 28 实测降至约 6.42–6.54GB，更适合 8GB 笔记本 GPU 长时间训练。

例如：

```powershell
.\scripts\run_modern_pipeline.ps1 -Mode Train -Branch s101_224
.\scripts\run_modern_pipeline.ps1 -Mode Resume -Branch s101_224
```

这些配置只用 5,400 个训练身份，并每个 epoch 在 600 个未见身份构造的 2,000 个验证对上输出 `ROC_AUC`。输出分别位于 `logs/modern_*`。

### 4.1 持久 latent workspace 实验

第一版结构分支与 baseline 隔离，使用独立配置和输出目录：

```powershell
.\scripts\run_modern_pipeline.ps1 -Mode LatentSmoke
.\scripts\run_modern_pipeline.ps1 -Mode LatentTrain
.\scripts\run_modern_pipeline.ps1 -Mode LatentResume
```

如需把完整终端输出归档，可使用统一的外层日志参数：

```powershell
# 写文件并继续在终端实时显示
.\scripts\run_modern_pipeline.ps1 -Mode LatentTrain `
  -LogFile .\logs\latent_train.log

# 安静运行：stdout/stderr 只写文件
.\scripts\run_modern_pipeline.ps1 -Mode LatentTrain `
  -LogFile .\logs\latent_train.log -LogOnly
```

`LatentResume`、`Resume` 和 `ResumeFinal` 会追加同一外层日志而不是覆盖；每次 Python 调用前写入时间和命令。模型目录内原有的 `log.txt`、`metrics.json` 和 TensorBoard 文件保持不变。

2026-08-10 的首轮 latent 运行暴露并修复了混合 train/eval 状态恢复错误：验证结束不再递归重开仍处于冻结期的 backbone，冻结钩子也会在每个 step 重申该约束。冻结优化器的进度现在也进入 checkpoint；现有旧 checkpoint 会从 Adam step 自动推断。`model_0000.pth` 的 Adam 总步数为 1,242，在当前 1,000-step freeze 配置下会恢复为冻结已完成，续训立即解冻。该次运行的 `model_0001.pth` 受 BN 模式切换影响，仅保留作诊断；`last_checkpoint` 已指回验证前状态干净的 `model_0000.pth`，可直接使用 `LatentResume` 继续。

正式配置是 `configs/modern_latent_workspace_s101_224.yaml`。除 meta architecture 和验证 batch 外，它继承 `modern_s101_224.yaml` 的数据划分、增强、三项损失、训练 batch、学习率、warmup、backbone freeze 与 35 epoch 调度。验证 batch 从 64 降到 16 只为避免 8GB GPU 上 FP32 推理峰值，不改变样本特征或评价公式。

现代配置的 warmup 仍按作者 batch 换算样本曝光量（S101 1,143 steps、S200 914 steps），但 backbone freeze 统一保留作者的 1,000 次优化器更新，不再按 batch 放大。小 batch 导致的批内负身份减少不是冻结时长可以补偿的问题，后续由真实大 batch 或跨批次记忆实验处理。

workspace 使用 8 个跨 C2→C5 持久槽、192 维、8 头和 4 倍 FFN。C2 下采样后只读，C3/C4/C5 执行共享单元的 Read–Mix–Write；写回门为 `0.05 + 0.95*sigmoid(a)`，初值 0.10。workspace 是顶层模块，不属于 `backbone`，因此当前前 1,000 iter 冻结 backbone 时它仍从第 0 步以正常基础学习率更新。

每 200 iter 写入 `latent/*` 健康指标：三阶段 gate/write ratio、workspace 梯度范数、非零/有限梯度比例、参数梯度覆盖率和 slot variance/cosine。smoke 改为每 iter 记录。最终 dim-192 smoke 的 workspace 参数梯度覆盖率与有限梯度比例均为 100%，非零梯度元素约 99.97%。

同机 synthetic batch-28 稳态 A/B（BF16、三项损失、反向与 Adam 全部计时）：

| 模型 | 参数量 | 中位秒/iter | PyTorch 峰值分配显存 |
|---|---:|---:|---:|
| S101-224 baseline | 46.24M | 0.331 | 3,836 MiB |
| + latent workspace | 48.42M | 0.369 | 4,215 MiB |

即参数增加约 2.17M、峰值增加约 379 MiB、总 step 时间增加约 11.5%。复测命令：

```powershell
D:\CondaData\envs\torch312\python.exe .\scripts\benchmark_latent_workspace.py
```

这些结果是工程验收，不是效果结论；是否接受该结构必须等待相同 seed/划分下的完整 baseline 与 latent 35-epoch AUC 对照。

## 5. 模型选择后的全量训练

确定架构和训练轮次后，使用 6,000 个身份做最终训练：

```powershell
.\scripts\run_modern_pipeline.ps1 -Mode Final -Branch s101_224
.\scripts\run_modern_pipeline.ps1 -Mode ResumeFinal -Branch s101_224
```

对应 `modern_final_*.yaml` 配置。全量训练不再运行留出验证，checkpoint 写到 `logs/retrained_*`。四个分支应逐个执行；8GB 单卡不要并行启动。

从项目根目录一次顺序训练四个分支：

```powershell
$branches = 's101_224', 's101_256', 's101_288', 's200_224'
foreach ($branch in $branches) {
    .\scripts\run_modern_pipeline.ps1 -Mode Final -Branch $branch
}
```

如果训练被关机或中断，把 `Final` 改为 `ResumeFinal` 后重新执行同一循环；已完成的分支会跳过训练迭代，未完成的分支从最近一个 5-epoch checkpoint 继续。

本机 RTX 4060 Laptop 8GB、BF16、batch 28 的短时全量数据前后向实测如下。每个 epoch 为 1,379 iterations；时间估计在纯迭代测量之上预留了数据读取、checkpoint 和笔记本持续负载波动：

| 分支 | PyTorch 峰值保留显存 | 实测平均秒/iter | 35 epoch 预计 |
|---|---:|---:|---:|
| S101 224 | 4.02GB | 0.478 | 7–9 小时 |
| S101 256 | 5.02GB | 0.699 | 10–12 小时 |
| S101 288 | 6.42GB | 0.947 | 14–17 小时 |
| S200 224 | 6.54GB | 0.874 | 13–16 小时 |

四个分支顺序执行预计约 44–54 小时，即连续运行约 2–2.5 天；笔记本降频或同时占用 GPU 时可能接近 3 天。系统内存建议至少 16GB，并在启动前留出 8GB 左右可用内存；本机当前总内存 15.6GB，可以运行，但应关闭大型浏览器、游戏和其他模型进程并保持 Windows 页面文件开启。

为控制磁盘占用，全量配置每 5 个 epoch 在 `model_recent_0.pth` 和 `model_recent_1.pth` 两个恢复槽间交替覆盖；训练结束再写 `model_final.pth`。因此每个分支最多保留三份 checkpoint，不会累积 35 份历史模型，而且任一恢复槽写入中断时仍有另一份可用。

两份官方 ImageNet 初始化权重已位于 `upstream/Pet-ReID-IMAG/pretrain/`，SHA-256 分别以 `22405ba7` 和 `75117900` 开头，并已实际构建 ResNeSt-101/200 模型验证加载。

## 6. Phase B 推理和融合

重建作者 checkpoint 的提交：

```powershell
.\scripts\run_modern_pipeline.ps1 -Mode AuthorPhaseB
```

流程会为四个模型分别保存 `query_f.npy`、`gallery_f.npy` 和两份 filename 映射，再拼接四个 2048 维描述符、整体 L2 归一化并给 2,000 个 pair 计算 cosine×100。输出：

```text
upstream/Pet-ReID-IMAG/logs/fusion_submit/submit_modern.csv
```

四个全量重训 checkpoint 都完成后可执行：

```powershell
.\scripts\run_modern_pipeline.ps1 -Mode RetrainedPhaseB
```

输出位于 `logs/retrained_fusion/submit.csv`。

生成最高/最低相似度各三对的定性图：

```powershell
D:\CondaData\envs\torch312\python.exe .\scripts\make_pair_contact_sheet.py
```

输出为 `results/phase_b_pair_examples.png`，图内也明确标注了“无隐藏标签，不能据此宣称预测正确”。

## 7. 本次真实验收记录

- 完整验证：4,000 张图像描述符推理约 26 秒，作者权重 AUC 1.00（有训练身份泄漏，只作链路检查）；
- smoke 训练：batch 8，三项损失均为非零，训练+验证峰值显存约 5.1GB；
- 恢复训练：optimizer 与两个 scheduler 均加载，从下一 epoch 继续；
- 四模型 Phase B 推理：约 65s / 59s / 80s / 88s；
- 每个分支输出 query/gallery `[4000, 2048]`；
- 融合输出 2,000 行，prediction 全部有限，无缺图和重名冲突。

## 8. 仍无法本地完成的唯一指标

Phase B 的 `test_data.csv` 只有 `imageA,imageB`，没有比赛隐藏标签。本工作区能稳定重建提交分数，但官方 Phase B ROC-AUC 仍必须由官方标签或评测服务计算。几张图的人工对比可以做定性 sanity check，不能替代这项隐藏指标。
