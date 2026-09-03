# 当前工作区状态与恢复记录

更新时间：2026-09-02  
工作区：`D:\Pet-ReID-IMAG_repro_attempt_2026-08-09`

这份文件是项目状态的持久记录，不依赖 Codex 聊天窗口。新会话、任务中断或界面显示不完整时，先读本文件，再读 `models/registry.json` 和对应发布包的 README。除非明确要求，不要用 Git reset/clean 覆盖当前修改。

## 当前结论

| 范围 | 实际状态 |
|---|---|
| Git | 当前工作区分支为 `codex/workspace-cleanup`；递归检查只剩根目录 `.git`。工作区仍有大量有意保留的未提交源码、实验配置和文档修改。 |
| 源码 | 唯一源码根为 `src/Pet-ReID-IMAG/`。原始上游 Git、bundle 和补丁保存在 `archive/`，用于恢复，不作为运行入口。 |
| 原始数据 | 正式路径为 `data/raw/DogFaceNet_alignment/`。根目录 `DogFaceNet_alignment/` 是指向该目录的 Windows junction，只为兼容旧命令，不是第二份数据。 |
| 本地输入 | `data/local_gallery/` 与 `data/queries/inbox/` 已建立；根目录旧的 `1/`、`2/`、`new-images/` 不再作为入口。 |
| 模型 | 当前研发主线是 `unified_pet_reid_v4_v1`（UnifiedPetReID V4，已验证候选）；生产基线、默认部署和回滚点仍是 `unified_pet_reid_v3_v1`（UnifiedPetReID V3）。 |
| 统一图库 | `pet_api_gallery_unified_v3_v1` 和 `pet_api_gallery_unified_v4_v1` 目前都尚未创建；首次启动对应后端时自动创建。已有的 semantic/BIFOR/Agent/temporary 图库不能直接混用。 |
| 服务 | 本次状态核对时 3000、8000、8080 均无监听，未留下常驻 Python、Java 或前端服务。 |

## 模型与证据

- V3 生产基线后端：`unified-onnx`；raw RGB 动态 `[N,3,H,W]` 单输入，单 512D 输出。
- V3 E2E ONNX：`models/selected/unified_pet_reid_v3_v1/onnx/e2e/unified_pet_reid.onnx`
  - SHA-256：`2db41b25d770eb285cd313f4e81a1f77c2017e70d827c0b9a1e48cf74edaf8a5`
  - 图内步骤：centered black letterbox、ImageNet normalization、learned geometry/crops、fusion、L2。
  - graph contract：输入 `float32 RGB 0..255 [N,3,H,W]`，输出 `float32 [N,512]`，无 external tensor/model。
  - 验证：`models/selected/unified_pet_reid_v3_v1/onnx/e2e/validation.json`；CPU/CUDA ORT 最大绝对误差约 `4e-6`，余弦最低 `0.9999999`。
- 旧固定方图 `models/selected/unified_pet_reid_v3_v1/onnx/unified_pet_reid.onnx` 保留为 rollback artifact，默认 runtime 不再使用。
- V4 当前研发后端：`onnx-highres`；单动态 RGB 输入，单 512D 输出，独立图库。它已通过候选验收，但尚未执行生产激活。
- V4 ONNX：`models/selected/unified_pet_reid_v4_v1/onnx/unified_pet_reid_v4.onnx`
  - SHA-256：`dbd4448133efec28efb770a6ce77c749b4f8b0913c8f40273420be571fe7b000`
- V4 checkpoint SHA-256：`edb8287ccc93b043e6a8e99584df92f30e402ca61ddaeb209b198f8cc1aab72d`
- V4 候选锁 SHA-256：`36b3dd6a7e0140b13c41bc1f8275d4419ac81bec08afcc7342d7156a44e4e01f`
- V4 blind 报告 SHA-256：`9e2c00655a4f835a70c1a3c276d29304344e859c78742a8d26f65eb01bb0025d`
- blind 已消费且禁止 post-blind tuning；不要重新运行该 split 来调参。

## 已验证的运行入口

```powershell
# 生产基线（当前解析为 UnifiedPetReID V3）
.\start-pet-reid.cmd
.\start-pet-reid-cpu.cmd

# 当前研发候选（UnifiedPetReID V4）
.\start-pet-reid-highres.cmd
.\start-pet-reid-highres-cpu.cmd

# 状态与停止
.\scripts\pet-reid-stack.ps1 status
.\stop-pet-reid.cmd
```

前端为 `http://localhost:3000`，Java 网关为 `http://127.0.0.1:8080`，Python API 为 `http://127.0.0.1:8000`。版本角色规则见 [`VERSIONING_CN.md`](VERSIONING_CN.md)，V4 的完整说明见 [`UNIFIED_HIGHRES_V4.md`](UNIFIED_HIGHRES_V4.md)，API 说明见 [`PET_API.md`](PET_API.md)。

最近一次已记录的验证包括：V3/V4 的 CPU 与 CUDA 四种 ONNX 栈组合均完成 Python → Java → Web 隔离全链路，完整 Python 回归为 `201 passed, 15 warnings`。原整理阶段的更大回归数字仍保存在 `docs/WORKSPACE_CLEANUP_PLAN_CN.md` 和 `artifacts/` 报告中；不同协议不要混写成同一个指标。

## ONNX 端到端验收（2026-09-02）

四次验收均使用独立端口和临时图库，结束后服务与端口已释放：

| 模型 | Provider | 报告 |
|---|---|---|
| UnifiedPetReID V3 | CPU | `artifacts/runs/live-stack-e2e/onnx-e2e-v3-custom-20260902/live-stack-smoke.json` |
| UnifiedPetReID V3 | CUDA | `artifacts/runs/live-stack-e2e/onnx-e2e-v3-cuda-20260902/live-stack-smoke.json` |
| UnifiedPetReID V4 High Resolution | CPU | `artifacts/runs/live-stack-e2e/onnx-e2e-v4-custom-20260902/live-stack-smoke.json` |
| UnifiedPetReID V4 High Resolution | CUDA | `artifacts/runs/live-stack-e2e/onnx-e2e-v4-cuda-20260902/live-stack-smoke.json` |

每份报告都验证了模型 SHA-256、单图单 512D 输出、无外部模型、API 鉴权、注册/识别、批处理、历史、图库备份/恢复和前端入口。V3 的独立导出验证还覆盖动态 H/W、图内 letterbox 和 CPU/CUDA parity，且明确 `blind_data_used=false`。隔离栈脚本现在会把 `SERVER_PORT` 与前端 origin 传给子进程，避免 Java/前端回落到默认端口。`generate_workspace_metadata.py` 会跳过 V3/V4 已锁定发布证据，只更新 registry 的实际文件清单。

收尾复跑（当前工作区）：V3 CPU 使用隔离端口 8122/8123/3122 和临时图库再次通过完整 Python → Java → Web smoke；报告为 `artifacts/runs/live-stack-e2e/onnx-e2e-final-v3-cpu-20260902/live-stack-smoke.json`，服务已自动停止且端口已释放。

最终严格契约复核又增加了两项运行时保障：raw RGB tensor 必须是 finite
`float32 0..255 [N,3,H,W]`，越界输入直接拒绝而不在图外 clip/scale；
unified 图输出只检查其 L2 norm，不再由 Python/API/图库层除以范数改写。
图库 prototype 的“多参考均值后归一化”仍属于检索索引数学，不是模型输出
后处理。对应回归覆盖图内动态 letterbox、raw 像素边界、branch 为 `null`
以及禁止二次输出归一化。

最终代码上的隔离报告如下，均未读取或重跑 blind split：

| 模型 | Provider | 最终报告 |
|---|---|---|
| UnifiedPetReID V3 | CPU | `artifacts/runs/live-stack-e2e/final-20260902-unified-v3-cpu/live-stack-smoke.json` |
| UnifiedPetReID V3 | CUDA | `artifacts/runs/live-stack-e2e/final-20260902-unified-v3-cuda/live-stack-smoke.json` |
| UnifiedPetReID V4 High Resolution | CPU | `artifacts/runs/live-stack-e2e/final-20260902-unified-v4-cpu/live-stack-smoke.json` |
| UnifiedPetReID V4 High Resolution | CUDA | `artifacts/runs/live-stack-e2e/final-20260902-unified-v4-cuda/live-stack-smoke.json` |

四份报告均为 `passed=true`，并验证 Java health 透传单图、raw-spatial
输入、空 external-model 列表、模型哈希、注册/识别、批处理、历史、
图库备份/恢复及 Web 入口。前端 `eslint` 与生产 `vinext build` 也通过。

## 会话中断说明（工具层，不是项目故障）

2026-09-01 核对到 Codex 任务的最新正式回合从 2026-08-28 09:06:12 开始，仍显示 `inProgress`，没有 `completedAt`。该任务历史共有 51 个回合，其中 9 次上下文压缩、12 次失败（11 次额度耗尽、1 次本地响应服务 401）和 3 次中断。这个状态会让聊天窗口看起来停在旧时间，后续输入也可能不显示成独立的已完成回合。

恢复规则：

1. 先以本文件、`models/registry.json` 和运行报告为准，不以聊天摘要推断模型或目录状态。
2. 若任务仍卡在活动回合，先在 Codex 界面停止/中断旧任务，再开新的短任务；不要反复发送“继续”消耗额度。
3. 恢复训练只使用 checkpoint 和对应 manifest；不要因聊天缺段而重新导出或重新盲测。
4. 每完成一个阶段，先把命令、结果、哈希和未完成事项写回本文件或同目录的专项记录。

## 权威文件顺序

1. `models/registry.json`：模型角色、默认部署和资产哈希。
2. `docs/VERSIONING_CN.md`：模型代际、部署角色、实验编号和接口/格式版本的边界。
3. `models/selected/<package>/README.md`：单个发布包的契约和限制。
4. `docs/UNIFIED_HIGHRES_V4.md`、`docs/PET_API.md`：当前运行与接口说明。
5. `docs/WORKSPACE_CLEANUP_PLAN_CN.md`：整理计划及历史执行记录；未勾选的破坏性清理项目仍需人工确认。
6. `artifacts/` 与 `archive/`：不可直接当作当前默认配置，须以 manifest/registry 中的引用为准。

## Reference-aware 研究实验（2026-09-03）

- `ReferenceAwarePetReID` 的模型级 query/reference 路径、训练入口、ONNX 边界和
  held-out/open-set 评估入口已加入源码；checkpoint 分发器可恢复 V3 external-joint
  与 V4 high-resolution 封装，V4 也已完成只读恢复验证。
- V3 head-only 开发实验：
  `artifacts/runs/reference_aware_model/v3_head_only_dev_20260903/`；使用
  `dogfacenet_shared_v3_protocol_v1/dev_train_manifest.json` 与
  `dev_validation_manifest.json`，未读取 blind split。
- 评估报告：`evaluation.json`。2-reference Top-1 `193/200` 对 centroid/top-k
  `190/200`；3-reference Top-5 略低，open-set matcher AUC/误接受率也未优于基线。
  因此该 checkpoint 仍为研究候选，未写入 registry、未替换生产模型、未迁移图库。
- 续训 ablation `v3_head_only_dev_refs3_20260903/` 将训练采样扩到 1–3 张参考图；
  2-reference 为 `194/200`，但 1-reference 降为 `277/300`，3-reference 仍为
  `94/100`，open-set AUC 仍未超过基线。不要把“多训一个 epoch”当作稳定改进。
- 训练/评估入口不能直接使用 `unified_v4_full_standard35` raw manifest；该 manifest
  缺少固定尺寸 episode 所需的几何字段。需要 raw 高分辨率训练时应另建动态尺寸数据
  路径，不能把 1280 方图当作高分辨率细节训练输入；固定尺寸入口现在会显式拒绝这类
  checkpoint。
