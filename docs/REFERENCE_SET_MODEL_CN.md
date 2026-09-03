# 可学习的多参考图匹配器

当前图像模型仍然保持单图输入：每张图片输出一个 512 维 descriptor。新增的
`QueryConditionedReferenceMatcher` 是一个独立的可训练模型头，输入查询 descriptor
和某个身份的参考 descriptor 集合，输出一个身份分数。

它不会把所有参考图简单平均。模型先用查询与每张参考图的关系计算 attention，再
聚合与当前查询最相关的参考图；因此正面、侧面、背面等视角可以保留为不同证据。
输出还带一个以 centroid + masked top-k 为基线的有界 residual。新建模型在训练前
与现有 baseline 完全一致，便于回滚和做非劣性比较。

## 训练

训练脚本只读取已经生成的 descriptor 缓存，不会读取或修改锁定的 RGB 模型：

```powershell
python src/Pet-ReID-IMAG/tools/train_reference_set_matcher.py `
  --train-features <train-features.npz> `
  --validation-features <validation-features.npz> `
  --output-dir artifacts/runs/reference_set_matcher/experiment `
  --reference-count 2 `
  --max-references 4
```

每个 episode 采用 P-way 查询集。训练时会随机改变参考图数量，验证时固定前 N 张
参考图，其余图片严格作为 held-out query。checkpoint 中会记录输入缓存 hash、模型
配置和每轮 learned/baseline 指标。

## 部署

PyTorch 运行时：

```python
from pet_id.reference_set_model import ReferenceSetMatcherRuntime

matcher = ReferenceSetMatcherRuntime.from_checkpoint("model_best.pth")
scores, details = matcher.score_gallery(query_descriptor, gallery_prototypes)
```

`gallery_prototypes` 中需要保留 `reference_features`（形状为 `[K, 512]`）。
如果要部署到不带 PyTorch 的服务，可用：

```powershell
python src/Pet-ReID-IMAG/tools/export_reference_set_matcher.py `
  <model-best.pth> `
  --output-dir artifacts/runs/reference_set_matcher/experiment/onnx
```

导出的图固定最大参考数，输入为 `query [N,512]`、`references [N,K,512]` 和
`reference_mask [N,K]`，输出为 `score [N]`。`ReferenceSetONNXRuntime` 提供了对应
的 NumPy 适配器。

单个身份如果录入的图片超过 checkpoint 的 `max_references`，服务 runtime 会将
该身份切成多个小集合分别评分，再对最强的两个集合做 top-2 mean 聚合；完整参考图
数量和分块分数会写入诊断。因此不会因为图库继续补图而静默丢掉旧视角，也不会把
所有视角压成一个 centroid。

服务层可以直接加载上述两种格式：

```powershell
python src/Pet-ReID-IMAG/tools/serve_pet_api.py `
  --reference-matcher-checkpoint <model_best.pth-or-reference_set_matcher.onnx>
```

启动参数只负责加载候选 matcher；请求仍需显式指定
`scoring_mode=learned_reference_set`（或把该模式设为服务默认）。未加载
checkpoint 时选择该模式会得到明确的请求错误，不会静默退回 centroid。
matcher 必须使用与当前图像 encoder 相同的 descriptor 空间；如果训练缓存来自
另一套 encoder，应重新提取缓存并训练新的 matcher，不能只因为维度同为 512 就混用。

## 启用策略

现有服务默认仍使用 `centroid`，已有的 `reference_set` 启发式也继续可用。学习头
应先作为显式的 `learned_reference_set` 候选进行开发集评估；只有在不同参考数量、
不同视角子集和 open-set 阈值上都通过非劣性门槛后，才把它接到默认服务配置。这样
不会因为一次训练偶然提升就改变现有图库的行为。

## 当前开发探针

在现有 700 身份训练缓存、100 身份 held-out 开发缓存上，2 张参考图设置的最佳探针
结果为 Top-1 `193/200 (96.5%)`，baseline 为 `191/200 (95.5%)`；Top-5 为
`197/200 (98.5%)`，baseline 为 `197/200 (98.5%)`。1 张参考图时为 `279/300`
对 `277/300`，3 张参考图时与 baseline 持平。这个结果足以继续做正式实验，但还
不足以替换生产模型；训练后期出现过拟合，选择策略必须以 held-out 指标为准。
