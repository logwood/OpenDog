# 参考集感知模型

`ReferenceAwarePetReID` 是一个可选的模型级扩展。它保留现有单图 encoder 的
`RGB 图像 → 512D descriptor` 契约，同时增加一条训练/研究契约：

```text
query RGB [B,3,H,W] + reference RGB [B,K,3,H,W] + mask [B,K]
        ↓ 共享 image encoder（query 与每张参考图都经过同一套权重）
query descriptor + reference descriptors
        ↓ QueryConditionedReferenceMatcher
每个身份的匹配分数 [B]
```

这不是把多张图片先做一个固定平均。matcher 以 query 为条件给不同参考视角分配
权重，因此正面、侧面和背面可以保留为不同证据；`mask` 允许一个 batch 中的身份
拥有不同数量的参考图。训练时参考图和 query 由 episode 采样器分离，query 不会
出现在自己的参考集合中。

## 代码入口

- `pet_id/reference_aware_model.py`：共享 encoder + set matcher、checkpoint 和
  固定宽度 ONNX export wrapper；训练/导出入口会按 checkpoint 封装类型恢复
  基础、V3 external-joint 或 V4 high-resolution encoder，并验证各自的来源链；
- `pet_id/reference_token_model.py`：可选的 token-level cross-view matcher 和
  coverage gate；它在 descriptor-level 头之前保留空间互补信息，详见
  `docs/REFERENCE_TOKEN_MODEL_CN.md`；
- `pet_id/reference_aware_training.py`：图像 episode 采样、一次编码后展开候选
  身份、检索损失；
- `tools/train_reference_aware_model.py`：从现有单图 checkpoint 开始的可恢复训练
  入口。
- `tools/evaluate_reference_aware_model.py`：在身份不相交的 held-out manifest 上
  比较 centroid/top-k 基线与 query-conditioned matcher，并报告参考图数量和
  open-set 拒识统计。

最小调用示例：

```python
from pet_id.reference_aware_model import ReferenceAwarePetReID
from pet_id.reference_set_model import QueryConditionedReferenceMatcher

matcher = QueryConditionedReferenceMatcher(
    descriptor_dim=512,
    max_references=4,
)
model = ReferenceAwarePetReID(single_image_model, matcher)
score = model(query_rgb, reference_rgb, reference_mask)
```

默认训练先冻结 image encoder，只优化 set head；确认 held-out、多参考数量和 open-set
阈值均不劣后，再用 `--unfreeze-identity` 只解冻 encoder 的末端 identity blocks 做
小步联合微调。这样联合训练确实会把梯度传回图像模型，但不会自动改变现有生产
descriptor 或图库。

训练入口示例：

```powershell
python src/Pet-ReID-IMAG/tools/train_reference_aware_model.py `
  --train-manifest <train-manifest.json> `
  --validation-manifest <validation-manifest.json> `
  --base-checkpoint <single-image-model.pth> `
  --output-dir artifacts/runs/reference_aware_model/experiment
```

已打包的 UnifiedPetReID V3/V4 checkpoint 会从自身记录的来源恢复；只有未打包的
基础 checkpoint 才需要显式传 `--arcface-checkpoint`。episode 训练使用含
`resized_size`、`face_roi_xyxy`、`nose_roi_xyxy` 和 `roll_angle_radians` 的几何
manifest（例如 `dogfacenet_shared_v3_protocol_v1/dev_train_manifest.json` 与
`dev_validation_manifest.json`），不能直接把高分辨率 raw manifest 当作输入；固定尺寸
训练、评估和导出入口会主动拒绝高分辨率 raw checkpoint，避免把动态细节静默压成
1280 方图。

评估示例：

```powershell
python src/Pet-ReID-IMAG/tools/evaluate_reference_aware_model.py `
  --manifest <held-out-manifest.json> `
  --base-checkpoint <single-image-model.pth> `
  --matcher-checkpoint <reference-aware-model.pth> `
  --reference-counts 1,2,3,4 `
  --output artifacts/runs/reference_aware_model/experiment/evaluation.json
```

每个身份至少需要 `参考图数量 + 1` 张图；不足时评估会明确标记为 skipped，避免
把 query 同时放进自己的参考集。评估工具不会读取 blind split。

`ReferenceAwarePetReIDExport` 可以导出三输入的固定 `K` 图。它适合研究和批量离线
评估；现有 HTTP 服务仍默认使用单图 descriptor 和可选的 descriptor matcher，因为
在线逐身份重新编码原始参考图会显著增加延迟。后端若使用 descriptor gallery，可
调用 `model.score_descriptors(...)`，其数学与 set-aware 图像路径相同。

## 上线门槛

这个模型已经具备可训练、可保存、可恢复和可导出的实现，但不能仅凭一次开发集
提升就替换默认模型。正式启用前应锁定身份不相交的 held-out split，并分别报告：

1. 1/2/3/4 张参考图以及随机视角子集；
2. 新身份（open-set）阈值、拒识率和误接受率；
3. 与当前单图 descriptor、centroid 和 descriptor matcher 的非劣比较；
4. CPU/CUDA、PyTorch/ONNX 的分数一致性和实际延迟。

通过这些门槛后，可以把联合 checkpoint 注册为一个独立的研究候选；生产单图模型
和现有图库保持可回滚，不需要改名或复用历史代际编号。

## 当前开发验证（2026-09-03）

使用 V3 生产基线的 encoder、`dogfacenet_shared_v3_protocol_v1` 的
`dev_train_manifest`/`dev_validation_manifest`（700/100 个身份，身份互斥）完成了
3 个 epoch 的冻结 encoder 训练。产物位于
`artifacts/runs/reference_aware_model/v3_head_only_dev_20260903/`，完整对比报告是
`evaluation.json`。validation 中 2 张参考图的 Top-1 为 `193/200`，透明
centroid/top-k 基线为 `190/200`；1 张时两者均为 `279/300`，3 张时两者均为
`94/100`（matcher 的 Top-5 比基线少 1 个）。open-set 的 known/unknown 各 50 个
身份上，matcher AUC `0.9448`、阈值误接受率 `26%`，基线分别为 `0.9478` 和 `20%`。

这说明 query-conditioned head 已经能工作并在 2 张视角上有小幅收益，但目前还不
满足“多参考数量和 open-set 均非劣”的上线门槛；该 checkpoint 仅是研究实验，未注册
到 `models/registry.json`，也没有替换默认模型或图库。每身份只有 4 张图，因此本次
严格无重叠评估的 4-reference 桶被标记为 skipped。

随后从该 checkpoint 续训 1 个 epoch，并把训练采样范围扩到 1–3 张参考图；结果为
2-reference `194/200`、1-reference `277/300`、3-reference `94/100`，且 open-set
AUC 仍略低于 centroid 基线。由此暂不继续堆叠 epoch；下一轮若继续，应加入显式的
hard-negative/open-set margin 或阈值校准，并在同一评估报告中验证单视图不退化。
