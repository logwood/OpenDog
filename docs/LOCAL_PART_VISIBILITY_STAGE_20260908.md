# 模型内部位可用性：工程阶段与数据审计

日期：2026-09-09。状态：`engineering_verified_training_data_blocked`。

本轮已完成实现、数据审计、旧模型全量复现和新图导出验证；没有启动可见性训练，
没有形成新的准确率收益证据，也没有更换生产模型。仍遵守用户要求：不增加整图身份
保底分支，保持一个 raw RGB 输入、一个 512D 输出的完整端到端模型。

2026-09-09 已将标注数据问题推进到可人工操作：从合规身份池按固定 seed 抽取 120 个
身份（384 条 train、96 条 validation），实际运行当前完整模型并物化 1440 个局部 crop
（每条源图两个 face 尺度和一个 nose crop）。所有 960 个部位标签仍为 `unknown`；
这是标注队列，不是训练集，也没有把检测器或注意力统计写入标签。

## 已验证的结果

| 检查 | 实测结果 | 能说明什么 |
|---|---|---|
| 旧候选 checkpoint 恢复 | 自动映射已知旧配置，未知/部分缺失字段仍拒绝 | 兼容修复没有偷偷选择新门控 |
| 旧候选 512 图重放 | Top-1 **177/256**；所有特征与历史记录最大绝对差 **0** | 旧结果精确复现，而非只复现总准确率 |
| 新模型初始化回退 | 输出与模型内部局部脸部描述符逐值相等 | 关闭鼻部时没有 sigmoid 尾部或 refiner 泄漏 |
| 新 checkpoint | 完整真实模型安全保存/恢复一致；冻结 V3 状态逐张量校验 | 不是只测试小型替身或遗漏鼻部的模型 |
| 动态 ONNX | 关闭、人工开启两种工程探针 × CPU/CUDA × 3 种 N/H/W，共 **12/12** 通过 | 数值与单图契约有效，不代表分类效果 |
| ONNX 图 | 各 4926 个标准节点；一个 `rgb` 输入、一个 `embedding` 输出；0 external initializer | 可见性头、鼻编码器和 refiner 均保留在图内 |
| 联合回归 | **64 passed**，16 条已知导出弃用/追踪警告 | 覆盖数据、可见性、局部路径、原融合/refiner、预处理和输入审计 |
| 静态检查 | 本轮 7 个源码/测试文件 Ruff 通过 | 未发现该检查范围内的静态问题 |

动态输入形状为 `[1,3,73,121]`、`[2,3,121,73]`、`[1,3,2,3]`。
CPU/CUDA ONNX 最大绝对误差不超过 `3.17e-7`，最小余弦大于 `0.999999999996`。
真实模型只开放 **4514** 个可见性头参数；原 V3 可训练参数数为 **0**。
测试中对合成张量做的一次优化器更新只用于验证梯度/冻结，不是数据实验训练。

## 实现边界

`pet_id/unified_local_visibility.py` 中新增 `LocalPartVisibilityHead` 和
`LocalVisibilityUnifiedPetReID`。新模型类型为 `unified_local_part_visibility_v1`，
仅显式加载，不加入默认 checkpoint 分发器或 registry。

- 复用既有 geometry spatial feature map；分别做脸/鼻空间特征汇聚，预测两个
  `usable` logit。全图上下文只用于这两个标量，不产生独立全局身份描述符。
- 空间 attention 仍用于汇聚特征，但不把其集中度/熵解释为部位正确概率。
  `nose_valid_fraction` 仍只是采样合法性诊断，不作为可见性标签或新门控依据。
- 固定工程映射为 `a_part = clamp(2*sigmoid(logit_part)-1, 0, 1)`，
  `a_joint = a_face * a_nose`。这是未校准的设计约定，不是开发集扫描所得最优阈值。
- 鼻部系数为原融合模块保存的最大鼻部权重乘 `a_joint`；完整 refiner 修正也乘
  `a_joint`。refiner 的脸部信号改为显式脸部预测值，所有旧权重保持冻结。
- 任一部位 logit 不大于零，鼻部系数和整个 refiner 修正均为严格零，使用 `where`
  直接返回原局部脸部向量，不再对回退向量做一次可能改变末位的归一化。
- 初始化 classifier 权重为零、bias 为 -2，因而初始输出保守回退；BCE 监督仍有梯度。
  `train()` 会让冻结模型保持 `eval()`，避免 BN running buffer 漂移。

这只完成了“错误鼻部不能继续污染结果”的结构条件，没有证明头能判断正确，
也没有解决整图到局部脸部阶段的主要信息损失。若脸、鼻都不可用，保持 512D 接口
仍会返回局部脸部向量；不能因此宣称该输入可识别或已具备可靠拒识能力。

旧 `LocalPathUnifiedPetReID` 默认继续是 `legacy`。其历史 `reliability_mode="off"`
仍不承诺数学上的精确零，避免改变旧实验含义；严格零保证只属于新 opt-in 模型。

## 可用数据到底有什么

所有 manifest 均显式按 UTF-8 读取。以下统计是记录字段盘点，没有自动生成监督标签。

| 清单 | 图像 / 身份 | 已有部位相关字段 | 模型实际裁剪的可用性标签 |
|---|---:|---|---:|
| V4 raw train | 3200 / 800 | 无检测、关键点、分割、quality、viewpoint | 0 |
| YT-BB-Dog training extension | 4096 / 1024 | 无上述部位字段 | 0 |
| DogFaceNet dev train | 2800 / 700 | 2800 条均有检测、人工位置点、检测器关键点、分割、quality、viewpoint | 0 |
| DogFaceNet dev validation | 400 / 100 | 400 条均有相同几何/模型输出字段 | 0 |
| YT-BB-Dog development | 512 / 128 | 无上述部位字段 | 0 |
| highres development | 32 / 8 | 32 条有眼/鼻位置点 | 0 |

有鼻子坐标不代表鼻纹清晰，有检测框不代表当前模型实际裁剪正确，SAM/检测器输出也
不是人工可用性真值。特别是没有检测结果必须是 `unknown`，不能当作“鼻子不可用”。

交集检查发现：

- V4 train 与 DogFaceNet dev train 完全共享 700 身份、2800 张图。
- V4 train 还包含 DogFaceNet dev validation 的全部 100 身份、400 张图。
- V4 train 与 highres development 重叠 7 身份、25 个源文件 SHA-256；其中 1 身份
  已属于上述 dev validation，其余 6 身份在旧 dev train 内。
- 旧 dev train 与 highres development 重叠 6 身份、22 个源文件 SHA-256；按整个
  身份排除后必须删出训练资格的是 24 条，而不只是这 22 个同文件样本。
- YT-BB-Dog training extension 与列出的 development 清单无身份/源文件哈希交集。

因此保留三份 development 清单为保护范围后，DogFaceNet 合规候选池为
**2776 张 / 694 身份**，外部候选池为 **4096 张 / 1024 身份**。
合并去重共 **6872 张 / 1718 身份**；这是可安排标注的候选池，不是已经具备监督的训练集。
没有删除或改写任何原始 manifest/image；这些都是资格排除统计。

### 已消费 blind 的隔离

V4 `validation.manifest.json` 的指纹为
`33dbb399506b2122f1fca77b579336962b9071a1fe9be8d3e53c8a71ff5da236`，其协议锁引用的
来源指纹 `6f7ec3164e7f1535cc9eaf26c0cfa57abb0f62e40a6307eecb3ab034a9137d5a` 已只读核对，
确为历史 `dogfacenet_joint800_protocol_v1/blind_test_manifest.json`。
两者均进入新 loader 的禁止来源指纹表，不能换名后重新用于训练、标注或选模。
本轮没有读取该 blind 的图像或重新评测；审计器只记录协议锁中的来源关系。

身份交集是规范化身份名交集，图像交集是 exact source SHA-256；不是感知去重，也不
证明跨数据集别名无重叠。上游预训练数据重叠仍是 unknown，不宣称全新盲测。

## 标注 schema 与训练前置条件

新增 `pet_id/local_visibility_data.py`。schema 名为 `model_local_part_visibility`，
顶层绑定 `target_graph_sha256`、split 和带 SHA-256 的 source manifests。
每张图的 face/nose 必须分别给出 `usable`、`provenance`、`evidence`。

| 来源 | 含义和证据要求 | 默认损失权重 |
|---|---|---:|
| `human` | 人工看过模型实际局部裁剪；记录 annotator、annotation_id、scope | 1 |
| `teacher_weak` | 带模型指纹、目标匹配的阳性检测弱证据；不等同人工真值 | 0，必须显式 opt-in 才使用 |
| `synthetic` | 有完整部位范围的人工证据，且裁剪完全排除该范围 | 1 |
| `unknown` | `usable=null, evidence=null`，未观测/没有有效监督 | 0 |

合成负样本不接受“随意裁一块就算鼻部不存在”：必须提供完整部位范围、父图身份与
SHA-256、整像素保留范围，几何上无交集。像素 loader 还会核对父图/派生图哈希、
EXIF 转正后的尺寸，以及派生像素是否严格等于所声明父图裁剪。仅有一个鼻子点不能
证明完整鼻部被排除；教师框也不能冒充人工完整范围证明。

Loader 会拒绝来源哈希失配、越权 split、未声明/不属于来源的图片、未知字段、旧图
标签套用到新裁剪图，以及把 unknown 或检测失败写成负例。训练前必须调用
`require_independent_visibility_splits()`，并传入三份受保护 development 的全部 records；
它检查身份、图像及合成父图谱系交集，并要求两个部位都有训练正负监督及人工真实
正负验证样本。仅通过这些必要条件仍不等于样本量充分，需要事先确定标注覆盖与验收协议。

当前没有通过这些条件的数据，所以没有创建优化器来训练真实可见性模型。
下一步需要人工标注模型实际脸/鼻裁剪的可用与不可用案例，并预先锁定身份独立验证集。
不能使用开发集识别排名、margin、attention concentration 反推“真值”。

## 产物与复现

审计：`artifacts/runs/local_visibility/data_audit_20260908/data_audit.json`。

可人工标注的真实模型 crop 队列：
`artifacts/runs/local_visibility/crop_queue_120id_20260909_v3/crop_queue.json`；局部图片在
同目录 `crops/`，按 `face_crop_paths`、`nose_crop_path` 引用。队列状态为
`awaiting_human_model_crop_labels`，train/validation 身份不相交。

工程验证目录：`artifacts/runs/local_visibility/engineering_verification_20260908/`。

- `verification.json`：旧 177/256 全量重放、真实 checkpoint 恢复、12 条 ONNX 一致性记录。
- `unit_tests.xml`：64 项联合回归结果。
- `untrained_visibility_engineering_only.pth`：未训练工程快照，不能作为部署候选。
- `visibility_initialized_off_engineering_probe.onnx`：初始化关闭状态。
- `visibility_manually_enabled_not_trained_engineering_probe.onnx`：仅人工开启门控来覆盖
  鼻部/修正数值路径，不是另一次训练，也不代表更好的模型；不得拿它报识别提升。

新 checkpoint 指纹：`84dcfe920bbce1ad4cbc071b6f4cc66911dd762a5f91d892b20faa4b601668b4`。
本轮实际局部裁剪目标图指纹：
`1d8f961d836027ca4e52d4dac63b9f1e44824d779199f373de630b7a27b276ba`。

```powershell
& D:\CondaData\envs\torch312\python.exe src/Pet-ReID-IMAG/tools/audit_local_visibility_data.py --output-dir artifacts/runs/local_visibility/<new-audit>
& D:\CondaData\envs\torch312\python.exe src/Pet-ReID-IMAG/tools/verify_local_visibility.py --output-dir artifacts/runs/local_visibility/<new-verification>
```

两个命令都拒绝覆盖已有目录。验证只重放允许的 YT-BB-Dog development 和合成工程输入；
没有扫描新阈值，没有启动大规模训练。生产 registry 前后 SHA-256 均为
`267edfc1355646c4497f20459c1437ea02af25e65bf91c20d8802f2520b902ff`。
