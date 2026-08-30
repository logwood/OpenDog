# Pet-ReID-IMAG 完整复现工作区

## 结论

当前工作区已经达到“可训练、可验证、可恢复、可推理”的目标，并在原生 Windows 11 + RTX 4060 Laptop 8GB 上完成真实 GPU 验证，不需要 WSL、Linux 编译器或本机 `nvcc`。

已经实际跑通：

- Python 3.12.13、PyTorch 2.11.0+cu126、torchvision 0.26.0+cu126；
- 38,636 张训练图、6,000 个身份及 4,000 张 Phase B 测试图；
- BF16/FP16 自动混合精度前向、三项训练损失、反向和 Adam 更新；
- 训练、验证、best/final checkpoint、从 checkpoint 恢复下一 epoch；
- 四个作者 checkpoint 的完整特征重提取；
- 四分支融合和 2,000 对 Phase B 提交文件生成。

最终已验证的 Phase B 输出是 `artifacts/runs/legacy/fusion_submit/submit_modern.csv`。

## 本地验证协议

发布代码没有给出论文 validation AUC 对应的划分和标签。本工作区补了一套可重复、身份不相交的协议：

- 固定 seed 2022，以身份为单位划分 5,400 个训练 ID 和 600 个验证 ID；
- 验证集包含 1,000 个正对和 1,000 个负对；
- 以归一化描述符的 cosine similarity 计算 ROC-AUC、最佳阈值和该阈值下 accuracy；
- 划分清单位于 `data/processed/pet-reid-imag/splits/`。

作者 checkpoint 在这套验证对上得到 AUC 1.00，只能证明验证代码链路正确：作者 checkpoint 训练时见过全部 6,000 个身份，存在数据泄漏。使用 `PetID` 从头训练时，600 个验证身份没有进入训练，才是有效的本地泛化评估。该协议也不是论文未发布验证协议的逐项复原。

Phase B 的 `test_data.csv` 仍然没有隐藏标签，所以本地只能生成 2,000 个 prediction，不能声称复现论文/比赛的官方 Phase B AUC。

## 快速入口

在本目录打开 PowerShell：

```powershell
# 启动完整应用：CUDA ONNX / CPU ONNX（二选一），前端端口 3000
.\start-pet-reid.cmd
.\start-pet-reid-cpu.cmd

# 启动已迁移独立图库的 Semantic V3 + BIFOR：CUDA / CPU（二选一）
.\start-pet-reid-bifor.cmd
.\start-pet-reid-bifor-cpu.cmd

# 启动多专家 Agent V1：BIFOR + 冻结 MegaDescriptor，CUDA / CPU
.\start-pet-reid-agent.cmd
.\start-pet-reid-agent-cpu.cmd

# 查看状态或停止
.\scripts\pet-reid-stack.ps1 status
.\stop-pet-reid.cmd

# 真实 HTTP 全链路验收：独立临时图库，结束后自动停服
.\scripts\test-live-stack.ps1 -Provider cpu
# CUDA 版：.\scripts\test-live-stack.ps1 -Provider cuda
# BIFOR CUDA 版：.\scripts\test-live-stack.ps1 -Provider cuda -Model semantic-v3-bifor
# Agent CUDA 版：.\scripts\test-live-stack.ps1 -Provider cuda -Model agent-v1

# 非破坏性刷新模型、checkpoint、legacy 运行和 Git bundle 审计元数据
D:\CondaData\envs\torch312\python.exe .\scripts\generate_workspace_metadata.py

# 快速验证 AMP 训练、AUC、checkpoint（约 1 分钟）
.\scripts\run_modern_pipeline.ps1 -Mode Smoke

# 训练一个留出验证分支；可选 s101_224/s101_256/s101_288/s200_224
.\scripts\run_modern_pipeline.ps1 -Mode Train -Branch s101_224

# 从该分支最后一个 checkpoint 恢复
.\scripts\run_modern_pipeline.ps1 -Mode Resume -Branch s101_224

# 持久 latent workspace：真实 smoke、留出验证训练、恢复
.\scripts\run_modern_pipeline.ps1 -Mode LatentSmoke
.\scripts\run_modern_pipeline.ps1 -Mode LatentTrain
.\scripts\run_modern_pipeline.ps1 -Mode LatentResume

# 合并 stdout/stderr 到外层日志，同时保留实时终端输出
.\scripts\run_modern_pipeline.ps1 -Mode LatentTrain `
  -LogFile .\artifacts\workspace_logs\latent_train.log

# 只写文件，不在终端打印大量模型结构
.\scripts\run_modern_pipeline.ps1 -Mode LatentTrain `
  -LogFile .\artifacts\workspace_logs\latent_train.log -LogOnly

# 模型选择后，用全部 6,000 个 ID 做最终训练
.\scripts\run_modern_pipeline.ps1 -Mode Final -Branch s101_224

# 恢复中断的全量最终训练
.\scripts\run_modern_pipeline.ps1 -Mode ResumeFinal -Branch s101_224

# 用作者四个 checkpoint 重建 Phase B CSV
.\scripts\run_modern_pipeline.ps1 -Mode AuthorPhaseB
```

全量训练每 5 个 epoch 在 `model_recent_0.pth`、`model_recent_1.pth` 两个恢复槽间交替覆盖，完成时另存 `model_final.pth`；每个分支最多保留三份 checkpoint。

FastReID 仍会在各自 `OUTPUT_DIR/log.txt` 保存框架日志；`-LogFile` 是更完整的外层日志，还会捕获资产检查、环境信息、标准输出和标准错误。相对日志路径统一从本工作区根目录解析，恢复模式会追加到已有文件。

完整环境创建、配置矩阵和输出说明见 `docs/REPRO_GUIDE_CN.md`。作者代码疑点、当前复现偏离和复现完成后的消融顺序统一记录在 `docs/IMPROVEMENT_LEDGER_CN.md`，避免方法改进污染复现基线。

默认应用仍使用 Semantic V3。另有经过锁定协议验证的
`dogfacenet_semantic_v3_bifor_lowrank_v1` 研究候选，把 8% BIFOR 身体信息与
Semantic V3 联合到一个 512 维 ONNX 描述符中；它不是默认部署，而且坐标空间
不同，不能直接复用旧特征。现有持久图库的 2 个身份、4 张原图已经原子重编码到
`data/gallery_store/pet_api_gallery_semantic_v3_bifor_lowrank_v1`，并用 4 张未入库
同身份图片和完整 Java → Python → Web HTTP 流程验收。迁移、原图推理、API 参数
和验证证据见
`models/selected/dogfacenet_semantic_v3_bifor_lowrank_v1/README.md`。

## 多专家识别 Agent V1

Agent V1 保留 BIFOR 的 512 维主空间，同时加入冻结的
`MegaDescriptor-B-224` 1024 维身形专家。两个空间分别存储、分别计算 cosine，
只在分数层进行无训练、单调证据融合；不会拼接、压缩或污染原有 BIFOR Gallery。

- 独立 Gallery：`data/gallery_store/pet_api_gallery_agent_v1`；
- 2 个身份、4 张参考图均同时具有 512-D 主特征和 1024-D 专家特征；
- 4 张未入库查询融合 Top-1 4/4、接受 4/4、专家一致 4/4；
- 输出新增专家权重、逐专家结果、冲突判断、三态决策和补拍建议；
- 没有拟合 Platt/Isotonic 参数，`zero_shot_monotonic_v1` 分数明确不是概率。

迁移和查询报告位于
`artifacts/runs/agent_v1/gallery_migration/report.json` 与
`artifacts/runs/agent_v1/gallery_api/report.json`；Java → Python → Web 全栈
验收报告位于
`artifacts/runs/live-stack-e2e/20260829-agent-v1-full-01/live-stack-smoke.json`。

注意：MegaDescriptor 权重采用 **CC BY-NC 4.0**，只能用于非商业用途，除非另行
获得权利人的授权。详细模型记录见
`docs/AGENT_V1_MODEL_RECORD.md`。

## 关键内容

- `environment.repro.yml` / `requirements-modern.txt`：已验证的现代 CUDA 环境锁；
- `scripts/prepare_upstream_assets.py`：Windows 安全解压、文件名映射和固定验证划分；
- `scripts/run_modern_pipeline.ps1`：训练/恢复/最终训练/Phase B 一键入口；
- `scripts/benchmark_latent_workspace.py`：baseline/latent 同机 batch-28 资源 A/B；
- `scripts/test-live-stack.ps1`：隔离图库的 Java → Python ONNX → Web 全链路验收；
- `scripts/generate_workspace_metadata.py`：非破坏性生成模型 registry、checkpoint 清单、legacy 运行清单并验证 Git bundle；
- `scripts/fuse_and_score.py`：带自测的四分支融合与 pair 打分；
- `scripts/make_pair_contact_sheet.py`：从无标签 Phase B 分数生成高/低相似 pair 定性对比图；
- `docs/IMPROVEMENT_LEDGER_CN.md`：作者代码疑点、必要复现偏离和后续实验账本；
- `src/Pet-ReID-IMAG/configs/modern_*.yaml`：8GB GPU 配置；
- `src/Pet-ReID-IMAG/pet_id/`：完整数据集、验证器、特征导出和训练入口；
- `docs/LATENT_WORKSPACE_DESIGN_CN.md`：第一版结构创新的冻结设计、实现边界和验收结果；
- `docs/CHECKPOINT_RETENTION.md`：65 个模型/checkpoint 的保留状态、集中式 legacy 清单和人工隔离边界；
- `models/pretrained/resnest101-22405ba7.pth`、`models/pretrained/resnest200-75117900.pth`：已下载并通过官方 SHA-256 前缀校验。

已生成的定性样例位于 `artifacts/reports/phase_b_pair_examples.png`：最高分三对在视觉上高度一致，最低分三对明显不同；它是很好的 sanity check，但没有隐藏真值，不能换算为 accuracy/AUC。

作者四分支的原始 batch 80/64 无法原样放入 8GB 显存；现代配置使用 BF16 AMP 和 batch 28，每批由 7 个身份、每身份 4 张图组成，使 Triplet/Circle Loss 拥有 24 个跨身份负样本。最重两个分支实测峰值保留显存约 6.42GB/6.54GB。因此目标是功能与方法复现，不是旧软件栈、旧 batch 和逐 bit 数值复现。完整 35-epoch 四模型重训耗时较长，本次没有替用户启动这项长期计算，但它所依赖的预训练、数据、训练、验证、保存和恢复路径均已实测闭环。

第一版结构创新 `LatentWorkspaceBaseline` 也已实现并通过真实 GPU 闭环。它在 ResNeSt C2 只读、在 C3/C4/C5 双向读写一个跨阶段持久的 8-slot workspace；最终仍输出原有 2048 维 GeM/BN 描述符，损失与 baseline 完全相同。RTX 4060 Laptop、batch 28 的同机短测为 4.22GB 峰值、约 11.5% 中位 step 时间增量。当前只证明实现与资源可行，尚未用 35 epoch 未见身份 AUC 证明方法增益。
