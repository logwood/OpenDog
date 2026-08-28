# Pet-ReID-IMAG 工作区整理计划

制定日期：2026-08-28  
工作区：`D:\Pet-ReID-IMAG_repro_attempt_2026-08-09`

## 0. 使用原则

本计划按“先保护、再分类、后搬迁、最后删除”的顺序执行。

- [ ] 整理期间暂停训练、评估、模型导出和 API 服务，避免文件仍被写入。
- [ ] 每个阶段单独完成、验证和记录，不要一次性搬完整个工作区。
- [ ] 在 checkpoint 清单完成并人工确认前，不删除任何 `.pth`、`.pt`、`.onnx` 或数据文件。
- [ ] 在外层 Git 能完整保存源码前，不删除内层 `.git`。
- [ ] 优先移动到隔离区；永久删除作为最后一步。
- [ ] 遇到与本计划盘点结果不一致的文件时，先停止并重新分类。

## 1. 当前主要问题

上次盘点得到的主要情况如下；执行前需要重新核对一次，因为实验产物可能继续增长。

- 外层仓库：`logwood/OpenDog`，分支 `main`。
- 内层目录 `upstream/Pet-ReID-IMAG/` 还有一套 `.git`，来源为 `muzishen/Pet-ReID-IMAG`。
- 外层 Git 又逐文件跟踪内层源码，导致同一份源码同时受两套 Git 管理。
- `upstream/Pet-ReID-IMAG/` 约 38.8 GB，其中 `logs/` 约 35.2 GB。
- 根目录同时存在约 2.06 GB 的 `DogFaceNet_alignment/` 和约 1.96 GB 的压缩包。
- `1/`、`2/` 和 `new-images/` 是有实际用途的本地图库输入，但名称不清晰。
- 根目录混合了源码入口、数据、模型、日志、报告和下载归档。

## 2. 目标目录结构

最终建议整理为：

```text
Pet-ReID-IMAG_repro_attempt_2026-08-09/
├─ README_CN.md
├─ environment.repro.yml
├─ requirements-modern.txt
├─ SOURCES.md
├─ src/
│  └─ Pet-ReID-IMAG/             # 唯一源码目录，不含内层 .git
├─ scripts/                      # 工作区级辅助脚本
├─ docs/                         # 复现、设计和改进文档
├─ data/                         # 本地数据，Git 忽略
│  ├─ raw/
│  │  └─ DogFaceNet_alignment/
│  ├─ local_gallery/
│  │  ├─ local-1/
│  │  └─ local-2/
│  └─ queries/
│     └─ inbox/
├─ models/                       # 二进制模型忽略，元数据可跟踪
│  ├─ pretrained/
│  ├─ selected/
│  └─ registry.json
├─ artifacts/                    # 所有运行产物，Git 忽略
│  ├─ runs/
│  ├─ evaluations/
│  ├─ reports/
│  └─ workspace_logs/
└─ archive/                      # 临时隔离和可恢复备份，Git 忽略
   ├─ git/
   ├─ downloads/
   └─ quarantine/
```

`README_CN.md`、环境文件、许可证和主要入口留在根目录。次级设计文档可以移入 `docs/`。

## 3. 第一阶段：冻结现场并建立安全快照

### 3.1 确认没有活跃任务

- [ ] 确认没有 Python 训练或评估进程。
- [ ] 确认没有 Java/API/ONNX 服务正在使用模型文件。
- [ ] 记录暂停时仍未完成的实验及其恢复 checkpoint。

### 3.2 创建整理分支

- [ ] 在外层仓库创建分支 `codex/workspace-cleanup`。
- [ ] 不要为了获得“干净状态”而丢弃现有修改。

### 3.3 保存两套 Git 状态

需要分别记录外层仓库和 `upstream/Pet-ReID-IMAG/` 内层仓库的：

- [ ] 当前分支、commit 和 remote。
- [ ] `git status --short --branch` 输出。
- [ ] 已跟踪文件的普通 diff 和 staged diff。
- [ ] 未跟踪文件清单。
- [ ] 内层仓库的完整 Git bundle。
- [ ] 对 bundle 执行验证，并把验证结果写入文本文件。

建议产物放在：

```text
archive/git/2026-08-28/
├─ outer-status.txt
├─ outer-working-tree.patch
├─ outer-untracked.txt
├─ inner-status.txt
├─ inner-working-tree.patch
├─ inner-untracked.txt
├─ Pet-ReID-IMAG.bundle
└─ bundle-verify.txt
```

注意：Git bundle 只保存已提交历史，不能代替 working-tree patch 和未跟踪文件清单。

### 3.4 建立磁盘清单

- [ ] 列出所有文件的路径、大小和修改时间。
- [ ] 对所有模型、压缩包和准备删除的文件计算 SHA-256。
- [ ] 标记 Git 状态：tracked、untracked 或 ignored。
- [ ] 给每项标记分类：源码、原始输入、预训练模型、实验 checkpoint、日志、报告、缓存或未知。
- [ ] 对“未知”项目人工确认，不自动处理。

### 第一阶段验收

- [ ] 外层和内层源码状态都有独立记录。
- [ ] 内层 Git bundle 已验证可读。
- [ ] 没有删除或覆盖任何文件。
- [ ] 能明确指出所有未跟踪源码文件的位置。

## 4. 第二阶段：确定唯一 Git 所有权

推荐方案：外层 `OpenDog` 作为唯一主仓库；内层原始仓库只作为来源记录和可恢复 bundle 保存。

### 4.1 先让外层仓库完整保存源码

- [ ] 检查外层 `.gitignore`，确保不会忽略 `.py`、`.java`、`.yaml`、`.md`、测试和脚本。
- [ ] 检查所有未跟踪源码，排除数据、模型、缓存和生成报告。
- [ ] 按功能分批加入外层 Git，而不是一次加入全部未跟踪文件。
- [ ] 每批提交前检查 staged diff。

建议提交拆分：

1. 现代 PyTorch/Windows 兼容修改；
2. latent workspace 和实验配置；
3. multimodal/DogFaceNet；
4. gallery、API 和 ONNX；
5. 测试、工具和文档。

### 4.2 保存原始上游身份

- [ ] 在 `SOURCES.md` 中记录原始仓库 URL。
- [ ] 记录内层当前基线 commit `7a13155`，执行时再次核对。
- [ ] 记录工作区代码与该基线的偏离方式。
- [ ] 保留已经验证的内层 Git bundle。

### 4.3 隔离内层 `.git`

只有完成以下检查后才进行：

- [ ] 外层 Git 已经覆盖全部需要保留的源码和文档。
- [ ] 外层分支已经形成可恢复提交。
- [ ] 内层 bundle、patch 和未跟踪文件清单均已验证。
- [ ] `upstream/Pet-ReID-IMAG/.git` 的解析后绝对路径已经核对。

然后把内层 `.git` 移入 `archive/quarantine/2026-08-28/inner-git/`。先移动，不永久删除。

补充记录：第一阶段快照之后加入的 `src/BIFOR/` 也曾携带独立 `.git`。已先保存
`archive/git/2026-08-28/nested/BIFOR.bundle` 及状态清单，再将其隔离到
`archive/quarantine/2026-08-28/inner-git/BIFOR.git/`；BIFOR 源码由外层仓库统一接管。

### 第二阶段验收

- [ ] 工作区只剩外层一套有效 Git。
- [ ] 外层 `git status` 能显示所有预期源码变化。
- [ ] 原始上游 commit、remote 和历史仍可从文档及 bundle 恢复。
- [ ] 源码中没有模型、图片、数据集或日志被误加入 Git。

## 5. 第三阶段：按用途迁移目录

每次只处理表格中的一行；完成路径更新和验证后再处理下一行。

| 当前路径 | 目标路径 | 处理说明 |
|---|---|---|
| `upstream/Pet-ReID-IMAG/` | `src/Pet-ReID-IMAG/` | 完成 Git 整理后再改名 |
| `1/` | `data/local_gallery/local-1/` | 原始图片保持不变 |
| `2/` | `data/local_gallery/local-2/` | 原始图片保持不变 |
| `new-images/` | `data/queries/inbox/` | 作为未来待识别图片入口 |
| `DogFaceNet_alignment/` | `data/raw/DogFaceNet_alignment/` | 先更新所有数据路径 |
| `dog.pt` | `models/pretrained/dog.pt` | 同时记录来源和 SHA-256 |
| 根目录 `logs/` | `artifacts/workspace_logs/` | 与单次实验日志区分 |
| 根目录 `results/` | `artifacts/reports/` | 保留报告及定性图片 |
| 次级 Markdown 文档 | `docs/` | README 和 SOURCES 留根目录 |
| `DogFaceNet_alignment.zip` | `archive/downloads/` | 校验后再决定是否删除 |

### 5.1 每次迁移后的路径检查

搜索并更新：

- [ ] PowerShell 脚本中的旧路径。
- [ ] Python 和 Shell 脚本中的旧路径。
- [ ] YAML 配置中的 `OUTPUT_DIR`、数据目录和模型目录。
- [ ] README、复现指南和 `LOCAL_GALLERY.md`。
- [ ] 测试和 Java 代码中的固定路径。
- [ ] 所有硬编码的 `D:\Pet-ReID-IMAG_repro_attempt_2026-08-09`。

新代码优先使用：

- 相对于仓库或项目根目录的路径；
- 命令行参数；
- 单一工作区配置文件。

不要在不同脚本中重复写死本机绝对路径。

### 第三阶段验收

- [ ] 根目录不再有 `1/`、`2/` 和 `new-images/`。
- [ ] 源码、数据、模型和产物分区明确。
- [ ] Git 仍只跟踪源码、配置、元数据和文档。
- [ ] 旧路径已经通过全局文本搜索确认不存在，或仅存在于历史说明中。

## 6. 第四阶段：统一实验产物格式

以后每次运行使用：

```text
artifacts/runs/<workstream>/<run-id>/
├─ run_manifest.json
├─ resolved_config.yaml
├─ metrics.json
├─ stdout.log
├─ tensorboard/
├─ checkpoints/
└─ reports/
```

`run-id` 建议使用：

```text
YYYYMMDD-HHMM_<model>_<purpose>_<seed>
```

`run_manifest.json` 至少记录：

- [ ] 外层 Git commit。
- [ ] 启动命令。
- [ ] 配置文件和 resolved config。
- [ ] 随机种子。
- [ ] 数据清单或 split 版本。
- [ ] 开始、结束时间和运行状态。
- [ ] 最佳指标和对应 checkpoint。
- [ ] 是否允许删除中间 checkpoint。

不要再仅依靠 `model_0007.pth` 或目录名判断模型用途。

## 7. 第五阶段：checkpoint 保留与清理

这是主要磁盘回收阶段，但必须先生成清理预览。

### 7.1 生成 checkpoint 清单

为每个 `.pth`、`.pt`、`.ckpt`、`.onnx` 和 `.safetensors` 记录：

- [ ] 路径和大小。
- [ ] SHA-256。
- [ ] 所属实验和配置。
- [ ] epoch/step。
- [ ] 指标。
- [ ] 角色：best、final、recent、milestone、smoke、failed、release 或 unknown。
- [ ] 建议动作：KEEP、QUARANTINE 或 REVIEW。
- [ ] 保留或隔离理由。

输出建议为：

```text
artifacts/reports/checkpoint_inventory.json
artifacts/reports/checkpoint_cleanup_preview.md
```

### 7.2 建议保留规则

- 活跃训练：保留 `best + 最近两个恢复点`。
- 正式完成实验：保留 `best + final`；内容相同则只留一份。
- 消融实验：保留 `best + 最多一个有文档依据的诊断节点`。
- smoke 或失败实验：通常不留 checkpoint，只留日志、配置和指标。
- 发布模型：保留最终 `.pth`、导出的 `.onnx`、SHA-256 和来源说明。
- SHA-256 完全相同的重复模型只保留一份实体文件。
- 标记为 unknown 的模型必须人工确认。

已有 `CHECKPOINT_RETENTION.md` 中明确说明的诊断节点，在重新核对实验用途前不要删除。

### 7.3 两步清理

第一步：隔离。

- [ ] 审阅 `checkpoint_cleanup_preview.md`。
- [ ] 把候选文件移入 `archive/quarantine/<日期>/checkpoints/`。
- [ ] 保留原路径到隔离路径的映射清单。

第二步：永久处理。

- [ ] 运行第九阶段验证。
- [ ] 确认不需要恢复训练。
- [ ] 选择永久删除或转移到其他磁盘。
- [ ] 更新 `CHECKPOINT_RETENTION.md`，记录文件数量和释放空间。

注意：同一磁盘内移动到隔离区不会释放磁盘空间。checkpoint 通常也不适合依赖压缩来大幅节省空间。

## 8. 第六阶段：压缩包与缓存

### 8.1 `DogFaceNet_alignment.zip`

- [ ] 校验压缩包 SHA-256。
- [ ] 确认解压目录完整，并与预期文件数量和清单一致。
- [ ] 确认下载来源或其他恢复方式仍有效。
- [ ] 先将压缩包放入 `archive/downloads/`。
- [ ] 完成一次数据加载验证后，再决定是否永久删除压缩包。

永久删除该压缩包预计可释放约 1.96 GB，以执行时实际大小为准。

### 8.2 可再生成缓存

在确认没有进程使用后，可清理：

- [ ] `__pycache__/`
- [ ] `.pytest_cache/`
- [ ] `.ruff_cache/`
- [ ] 临时 contact sheet。
- [ ] 已被正式报告替代的中间导出。

不要把数据集、预训练权重或选定模型当作普通缓存处理。

## 9. 第七阶段：功能验证

完成目录迁移和隔离后执行。

### 9.1 Git 验证

- [ ] 全工作区只存在一个有效 `.git`。
- [ ] `git status` 只显示计划中的迁移和修改。
- [ ] `git ls-files` 中没有图片、数据集、模型、日志或密钥。
- [ ] `.gitignore` 覆盖 `data/`、`artifacts/`、`archive/` 和模型格式。

### 9.2 最小功能验证

- [ ] Python 关键模块能够 import。
- [ ] 配置文件能够加载。
- [ ] 数据集能够读取至少一个 batch。
- [ ] gallery 构建或验证流程能够读取新目录。
- [ ] 一次最小推理成功。
- [ ] 一次训练 smoke 成功保存并恢复 checkpoint。
- [ ] ONNX/API/Java 中实际仍需保留的工作流通过各自 smoke test。
- [ ] README 中至少一条快速入口命令能够原样运行。

### 9.3 模型验证

- [ ] 所有 KEEP 模型文件存在且 SHA-256 匹配。
- [ ] selected 模型能够加载。
- [ ] 隔离 checkpoint 不再被配置或文档引用。
- [ ] `models/registry.json` 能定位选定模型及其来源。

## 10. 提交建议

不要把所有整理塞进一个提交。建议依次提交：

1. `chore: capture workspace cleanup metadata`
2. `chore: consolidate repository ownership`
3. `refactor: organize workspace paths`
4. `chore: standardize experiment artifacts`
5. `docs: document retention and recovery workflow`

checkpoint 的隔离或删除不应作为源码提交内容；Git 中只保存模型清单、哈希和保留说明。

## 11. 完成标准

- [ ] 只有一套 Git 管理源码。
- [ ] 根目录只保留明确的项目入口和分类目录。
- [ ] `1/`、`2/`、`new-images/` 等无语义名称已消失。
- [ ] 数据、模型、实验产物和源码完全分区。
- [ ] 每个保留模型都有用途、配置、指标、来源和 SHA-256。
- [ ] 每个实验都能由 manifest 找到配置、指标和选中 checkpoint。
- [ ] 关键 smoke test 通过。
- [ ] 所有永久删除均有清单、有验证结果并经过人工确认。
- [ ] `CHECKPOINT_RETENTION.md` 和 README 已反映新结构。

## 12. 推荐实际执行顺序

如果时间有限，按以下顺序分几次完成：

1. 只完成第一阶段，取得可靠快照。
2. 完成第二阶段，解决双 Git 问题。
3. 先迁移 `1/`、`2/`、`new-images/`，验证 gallery。
4. 再迁移源码目录和工作区文档。
5. 最后迁移大数据、模型和实验产物。
6. 生成 checkpoint 清理预览并人工审核。
7. 隔离候选文件、运行完整验证。
8. 最终删除或外移大文件，记录实际释放空间。

最重要的停顿点是：完成安全快照后、隔离内层 `.git` 前、迁移数据集前、隔离 checkpoint 前，以及永久删除前。

## 13. 本次执行记录（2026-08-28）

本轮已完成源码归属、目录迁移和运行路径硬化；未执行任何永久删除或 checkpoint 隔离，因此原始数据、模型、图库和压缩包仍可恢复。

| 项目 | 结果 | 证据 |
|---|---|---|
| 外层 Git 唯一所有权 | 已完成 | 工作区递归只剩根目录 `.git`；内层仓库均有 bundle 和 quarantine 备份 |
| 源码/数据/模型/产物分区 | 已完成 | `src/`、`data/`、`models/`、`artifacts/`、`archive/` |
| 旧路径兼容 | 已完成 | `src/Pet-ReID-IMAG/pet_id/workspace_paths.py`，含旧绝对路径和 SAM2 Hydra 路径转换 |
| checkpoint 清单 | 已完成 | `artifacts/reports/checkpoint_inventory.json`（63 个文件、7 组重复） |
| 快速启动 | 已完成 | CPU ONNX + Java + 前端 3000 端口启动/停止闭环通过 |
| Python 回归 | 已完成 | `97 tests, OK` |
| Java 回归 | 已完成 | `13 tests, 0 failures, 0 errors` |
| 前端检查 | 已完成 | `npm run build`、`npx tsc --noEmit`、`npm run lint` 均通过 |
| 真实 CPU ONNX 比对 | 已完成 | 临时图库双分支查询正确识别 `pet-a`，`CPUExecutionProvider` |
| 真实 HTTP 全链路 | 已完成 | 独立图库经 Java 8080 → Python ONNX 8000 完成录入、识别、历史复核、4 图批量/CSV、难例、备份/幂等恢复；37 项语义断言通过，结束后 8000/8080/3000 均释放 |
| 破坏性清理 | 保留待人工确认 | 本轮没有移动或删除 checkpoint、数据、图库、压缩包 |

注意：第 5、6、8 阶段中涉及永久删除的勾选项保持未选中是有意的；需要人工确认后再执行。后续重复运行 `scripts/generate_workspace_metadata.py` 不会仅因时间戳改写 `models/registry.json`。
