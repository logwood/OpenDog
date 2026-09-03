# Token-level 参考集模型

`TokenReferenceAwarePetReID` 是多视角参考集的显式结构实验。现有的
`ReferenceAwarePetReID` 仍然保留为 descriptor-level 路径；新模型不会替换默认
服务、生产模型或已有图库。

## 结构

```text
query/reference RGB
        │ 共享单图 encoder
        ├─ 512D descriptor（保留单图契约）
        └─ layer4 feature map → 固定 token 网格
                         │
             query ↔ reference cross-token matching
                         │
             novelty / coverage gate（抑制重复视角）
                         │
             centroid/top-k 安全锚点 + 有界 residual
```

每个 query token 都可以在每张参考图中选择最匹配的局部 token。这样正面、侧面、
背面等互补视角不会在进入 512D descriptor 之前被强行平均。参考图之间还会计算
token-summary 的相似度；与其他参考图过于相似的行会得到较低 novelty，coverage gate
会限制它对最终分数的重复贡献。

如果 encoder 没有可发现的四维 feature map，`ImageTokenAdapter` 会退回一个明确的
descriptor-to-token 投影。这种 fallback 只用于保持接口完整；要获得真正的空间互补
能力，应使用带 `layer4`（或等价 feature-map）模块的 encoder。

## 运行一次结构实验

训练入口增加了显式开关，默认仍是 descriptor 路径：

```powershell
python src/Pet-ReID-IMAG/tools/train_reference_aware_model.py `
  --train-manifest <dev-train-manifest.json> `
  --validation-manifest <dev-validation-manifest.json> `
  --base-checkpoint <single-image-checkpoint.pth> `
  --interaction-level token `
  --token-dim 128 `
  --token-grid 4
```

token checkpoint 使用独立的 `reference-token-aware-pet-reid` 格式，可恢复训练；
descriptor checkpoint 与 token checkpoint 不会被互相静默加载。

## 当前 smoke 结果

在现有开发 manifest 上进行了 1 epoch / 1 step 的受限 smoke：

- token 结构成功从真实 encoder 捕获
  `base_model.geometry_frontend.identity_encoder.backbone.layer4`；
- 训练 loss `0.02067197`，验证 50 episodes / 100 queries 全部通过；
- focused regression 为 `20 passed`，Ruff 和语法检查通过；
- 合成 ONNX 导出成功（固定 token 网格，动态 batch）；
- 产物位于 `artifacts/runs/reference_aware_model/token_structural_smoke_20260903/`。

同一产物中的固定开发集对比也已完成：1/2/3 张参考图的 token matcher 与
centroid/top-k 基线分别为 `93.0% / 95.0% / 94.0%` Top-1，三档都完全持平。
这是预期的结构 smoke 结果：pair/score head 采用零初始化安全锚点，短跑不会把未训练
的 residual 当成“提升”。它证明 token 张量已进入模型和 checkpoint，但不替代正式训练
后的非劣性评估。

这个 smoke 只证明结构、梯度、checkpoint 和部署边界接线正确，不代表已经优于生产
模型。下一次正式比较应固定同一开发划分，同时报告单图、不同参考数量、leave-one-view-out、
hard-negative 和 open-set 指标；在这些指标通过前不应注册为默认模型。

## 固定预算试验（2026-09-03）

这次没有做温度、epoch 或参考数量 sweep。token 与 descriptor 两条路径使用同一
随机种子、同一冻结的单图 encoder、同一 8 个 epoch × 10 个 step 预算、最多 3 张
参考图和每身份 1 个 query。token 路径使用 4×4 网格、64 维 token；训练和评估都
可以直接接当前空间细节候选的高分辨率输入（每张原图先放到 2048 画布，模型内部仍
保持动态原图契约）。生产默认模型、registry 和图库没有改动。

高分辨率开发协议只有 8 个身份，结果很快饱和：

| 评估 | token | centroid/top-k 基线 |
| --- | ---: | ---: |
| 1 张参考 Top-1 | 95.83% | 95.83% |
| 2 张参考 Top-1 | 100% | 100% |
| 3 张参考 Top-1 | 100% | 100% |
| 留一视角（3 张参考）Top-1 | 100% | 100% |
| 重复视角子集 Top-1 | 100% | 100% |
| 互补视角子集 Top-1 | 100% | 100% |
| open-set false accept | 0% | 0% |

因此又做了一个 100 身份的固定画布补充检查（仍使用当前候选权重，但明确不把它
当作高分辨率细节结论）：

| 评估 | token | centroid/top-k 基线 |
| --- | ---: | ---: |
| 1 张参考 Top-1（300 queries） | 93.00% | 93.00% |
| 2 张参考 Top-1（200 queries） | 95.00% | 95.00% |
| 3 张参考 Top-1（100 queries） | 94.00% | 94.00% |
| 留一视角 Top-1（400 queries） | 96.25% | 96.00% |
| 重复视角子集 Top-1（100 queries） | 91.00% | 91.00% |
| 互补视角子集 Top-1（100 queries） | 99.00% | 99.00% |
| open-set false accept | 20% | 20% |

token 分数对最强误匹配的平均 margin 在 1/2/3 张参考时分别为 `0.362 / 0.384 /
0.396`，基线为 `0.290 / 0.306 / 0.320`；这说明新 head 学到了更保守的分数分离，
但在当前协议上还没有稳定的身份排序提升。结论是“结构已经真正接入模型，但收益尚
未证实”，所以候选 checkpoint 只保留在实验目录，不注册为默认路径。下一步若继续，
方向应是加入显式视角/局部覆盖监督和针对难负样本的训练目标，而不是继续调一组温度
或 epoch 参数。

作为额外 sanity check，单独训练的 descriptor-level head 在大样本固定画布上得到
`92.0% / 96.5% / 95.0%`（1/2/3 张参考）Top-1；token head 没有稳定超过这个
可学习 descriptor 对照。因此当前证据支持“接口和结构正确”，还不支持“结构已经带来
可部署收益”。

本次产物：

- `artifacts/runs/reference_aware_model/token_structural_smoke_20260903/`
- `.tmp/reference_token_v4_smoke_20260903/`
- `.tmp/reference_v4_fixed_token_20260903/`
- `.tmp/reference_v4_fixed_descriptor_20260903/`
- `.tmp/reference_fixed_canvas_large_token_20260903/`
- `.tmp/reference_fixed_canvas_large_descriptor_20260903/`
