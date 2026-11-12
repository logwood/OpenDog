# 狗脸跟踪网络与“跟踪 / 精定位”两级机制（2026-11-12）

结论：慢的根源确实在网络和机制上。原来每一帧都用 YOLO-pose 检测器在 320 px 的裁剪上找关键点（单线程 WASM 每次约 104 ms），而且它看不了转过 45° 以上的脸，所以还要四个方向轮流试。现在改成两级：

- **每帧**用一个专门为裁剪训练的小网络（128 px）跟踪脸，单线程 WASM 每次约 9.5 ms，任何旋转角度一次就能读出朝向；
- **只在要选图的帧**上，用原来的检测器在跟踪网络已经摆正的裁剪上精定位一次，选图核实和服务端复检都用这份精定位结果。

浏览器引擎（onnxruntime-web WASM，本机，1280×720 帧）提速 1.5–2.7 倍；正面素材的结果不变，你的视频没有误收；**但雪纳瑞素材的侧脸会漏采**，所以目前是可选功能，模型文件放进模型目录才启用，默认行为与之前逐字节一致。

## 1. 为什么慢：每帧推理次数与单次耗时

| 网络（裁剪任务） | 128 px | 160 px | 320 px |
|---|---|---|---|
| 现用 YOLO11n-pose（完整检测头） | 25.5 ms | 29.3 ms | 104 ms |
| MobileNetV3-small | 6.3 ms | 8.9 ms | – |
| YOLO11n 骨干 + 小头（本次采用） | 7.6 ms | 17.6 ms | – |

（单线程 WASM，Node，随机权重测速。）原机制每帧还要额外的搜索：干净正面素材每帧 1.3 次检测器推理，难素材约 3 次。

## 2. 跟踪网络 `dog_face_landmarks.onnx`

- 输入 128 px 规范化裁剪（脸占 35%，与原来的裁剪约定相同）；骨干是 YOLO11n 第 0–9 层，用现有狗脸检测器的权重初始化（已经学过狗脸特征）；小 FPN 输出 8 张热图（soft-argmax 取坐标）。
- 输出：6 个关键点；脸框（中心、长边，给下一帧建裁剪）；是否有狗脸；每个点是否可见；**整圈的朝向**。
- 朝向的做法：关键点只在离正立 ±55° 以内训练（跟踪时裁剪每帧都已摆正）；另外两张热图——鼻子、两眼中点——在任意角度训练（这两个点没有左右之分，倒过来也认得出），由它们的连线算出朝向。直接让全局特征回归朝向不行（中位误差 35°，全局池化丢了空间布局）；全角度训练关键点也不行（10 轮只到 12%）。
- 训练：dogface6 训练集 6 773 张，随机平移/缩放模拟跟踪误差、颜色/模糊/JPEG、随机遮挡物（被盖住的点标成不可见）、非狗脸区域作负样本；CPU 上 80 轮约 40 分钟。
- 验证集（关键点误差 < 0.25 眼距算合格）：离正立 ±55° 以内的裁剪 **66%**；朝向中位误差 17°，71% 在 30° 以内。

## 3. 机制

1. 整图检测器只负责发现新脸（与原来相同的节奏）。
2. 跟踪：每张脸每帧跑一次跟踪网络。新脸或跟丢时：先跑一次读出朝向，再在摆正的裁剪上跑一次（共 2 次，原来 5–6 次）。
3. 选图候选（在位置带内、质量够、能改进当前槽位）：检测器在跟踪网络摆正的裁剪上跑一次（`FaceRefiner.precise`）。精定位的关键点必须仍然落在这个位置带里，判定规则与服务端复检相同（区间 ± `pose.exit_margin`、倾斜上限），否则记为 `view_unconfirmed`；质量、五官核实、时间稳定性和最终保存的照片都用精定位结果。五官核实（开环复查 + 镜像）仍用检测器，原来标定的阈值不变。
4. 候选失败（精定位、位置、五官、眼睛）后，这条轨迹 0.25 s 内不再尝试（`refine.candidate_retry_s`）；时间稳定性还在累积通过次数时不受此限制。两侧素材上这一条把单线程从 7.5 提到 9.3 fps。
5. 抗混叠：跟踪裁剪在视频近景里要缩小 6–18 倍（训练裁剪中位 2.6 倍），双线性点采样会混叠，模型看到的是随框的亚像素移动而跳动的噪声。现在缩小超过 2 倍时先对覆盖区域做面积平均，再以约 1:1 旋转裁剪（`refine.warp_crop`，几何误差 ≤ 0.01 px）。默认只用于跟踪网络的裁剪（`refine.antialias: tracker`）：检测器的裁剪若也改，原来的核实阈值要重新标定（试过 `all`：两侧素材右脸被服务端拒、你的视频端上收了不合格帧）。
6. 服务端复检：同样先跟踪网络、再精定位，模型随会话配置下发给浏览器（`models.landmarks`）。

## 4. 结果

浏览器引擎（TS，onnxruntime-web WASM，Node，本机；1280×720，15 fps 时间戳）：

| 素材 | 单线程：改前 → 改后 | 4 线程：改前 → 改后 |
|---|---|---|
| 正面小狗 | 6.0 → **13.1 fps** | 12.1 → **17.5 fps** |
| 雪纳瑞两侧 | 3.5 → **9.3 fps** | 8.0 → **18.7 fps** |

每帧检测器推理（原生 Python 引擎，同逻辑）：正面 1.29 → 0.26，两侧 2.49 → 0.70，你的视频 2.96 → 0.51；另加约 1 次跟踪网络。

验证集正面对比（400 张，整张图随机旋转，模拟跟踪框）：

| 流程 | 离正立 < 45° | 45–135° | > 135° | 全部 | 原生耗时 |
|---|---|---|---|---|---|
| 原来：四向搜索 + 重裁（检测器 5–6 次） | 62% | 57% | 73% | 62% | 83 ms |
| 只用跟踪网络（2 次） | 54% | 34% | 38% | 40% | 9 ms |
| **跟踪 2 次 + 精定位 1 次** | **72%** | 52% | 56% | 57% | 45 ms |

跟踪时脸一直接近正立，这正是精定位比原来更准的区间（72% 对 62%）；大角度靠朝向读数，还不如原来的四向搜索。

三段素材（按服务端复检）：

| 素材 | 原来 | 跟踪网络 |
|---|---|---|
| 正面小狗 | 正面通过 | 正面通过（10 fps 分析也通过） |
| 雪纳瑞两侧 | 右侧通过（左侧服务端不收） | **右侧漏采**；左侧端上收、服务端不收 |
| 你的视频 | 无 | 无，未误收不合格帧 |

浏览器端到端：比对场景通过（两张正面服务端都收），多下载 5 MB 模型，无控制台错误；建档场景仍不通过（左侧被服务端拒、右侧未采到）。

## 5. 还差什么

雪纳瑞侧脸漏采的原因：跟踪网络在这只狗的侧脸上 yaw 系统性偏大（中位 +0.13，抗混叠前 +0.18；正面素材只有 −0.05），位置状态在区间边缘来回跳，等不到停留确认。验证集上它的 yaw 反而略小（比值 0.85），所以不是一个比例系数能校正的。下一步按收益排序：

1. 用检测器在视频帧上的精定位结果蒸馏跟踪网络（大量无标注视频帧，正好覆盖近景和侧脸）；训练裁剪也按视频的缩小倍数做面积缩放。
2. 在 RTX 4060 上用更多轮次（本次 CPU 80 轮）、或输入 160 px（单线程 WASM 约 17.6 ms，仍是原来的 1/6）。
3. 位置判定用跟踪 yaw 时，按同一轨迹最近几次精定位的差值在线校正。
4. 跟踪模式下整图检测的节奏（现在 0.3 s）可以放宽到约 1 s，单线程还能再省约 15 ms/帧。

## 6. 使用

启用：把 `dog_face_landmarks.onnx` 和同名 `.json` 放进模型目录（`DOG_CAPTURE_MODELS`），或 `--landmarks 路径`，或配置 `refine.landmarks`；`off` 关闭。服务端会自动下发给浏览器。

训练与评估：

```bash
python src/Pet-ReID-IMAG/tools/train_dog_face_landmarks.py train --data <dogface6> --init <dog_face_pose 的 best.pt> --out runs/lm --epochs 80
```

```bash
python src/Pet-ReID-IMAG/tools/train_dog_face_landmarks.py export --init <同上 best.pt> --weights runs/lm/best.pt --output dog_face_landmarks.onnx
```

```bash
python src/Pet-ReID-IMAG/tools/benchmark_landmarks.py --data <dogface6> --detector dog_face_pose.onnx --landmarks dog_face_landmarks.onnx
```

浏览器与 Python 的一致性（真实模型）：`export_refine_parity.py --landmarks dog_face_landmarks.onnx …`，再用 `DOG_CAPTURE_REAL=<目录> npx vitest run test/real-models.test.ts`。

## 7. 视频蒸馏首轮实测与 checkpoint 归档（2026-11-13）

已基于 `3dd87c3` 接通视频下载、教师标注、权重恢复、GPU 微调、ONNX 导出、原生回放与 SDK 一致性验证。**首轮候选尚未通过替换基线的验收**：在 Wolfhound 上 yaw 偏差有所下降，但验证集整体关键点误差和实时回放丢帧都出现回归。完整机器可读结果见 `assets/dog_capture/video_distillation_pilot_result.json`。

### 模型长期存放位置

项目根目录下保留两套独立文件，不再依赖用户的 Downloads 目录：

| 用途 | 路径 |
|---|---|
| 用户提供的原始 ONNX 与 JSON | `models/pretrained/dog_capture_landmarks_v1/dog_face_landmarks.onnx`、同名 `.json` |
| 从原 ONNX 恢复的可训练权重 | `models/pretrained/dog_capture_landmarks_v1/recovered_trainable.pt` |
| 原始模型来源与文件哈希 | `assets/dog_capture/landmark_checkpoint.json`；归档目录内另有 `archive_manifest.json` |
| 首轮实验候选 | `models/pretrained/dog_capture_landmarks_video_pilot_20261113/dog_face_landmarks.onnx`、同名 `.json`、`best.pt` |
| 候选验收结果 | 候选归档目录内 `acceptance.json` |

原 ONNX SHA256 为 `1fb6ad2185d0e4d65addf62c1d5fd79f32f84e61e62ae0dd9694609ca7cc1171`。
它包含推理权重，没有原优化器状态。`import_dog_landmark_weights.py` 按原导出结构恢复可训练参数，保留已经融合的 Conv/BN，拒绝未映射的权重；12 组输入与原 ONNX 最大绝对输出差为 `0.0000534`。这是可验证的微调起点，不是恢复原训练进度。

这些二进制与视频按现有仓库约定保留在本地归档，Git 跟踪脚本、来源清单、哈希和实验结果。启用任一版本时使用明确的 `refine.landmarks` 路径或 `--landmarks`，实验工具不会自动替换运行中的模型。

### 数据与训练约束

来源、作者、许可、下载地址、内容哈希和固定拆分均在 `assets/dog_capture/video_distillation_sources.json`。视频来自 Wikimedia Commons，先看内容，再按整段视频及狗分组；镜像里的同一只狗和同一视频里的多只狗不能跨训练与验证。用户视频和已有 Wolfhound 视频只作回归检查。

筛查了 15 段新下载短片，选取其中 8 段进入标注；其中 2 段没有产生合格标签。实际训练来自 4 段视频，共 **159 帧**，验证来自另外 2 段，共 **45 帧**；受保护回归视频留下 116 帧。连续近似重复帧会去重，采样按视频与姿态平衡，不能把相邻帧拆到训练、验证两边。

教师固定使用 320 px 检测器，关闭跟踪网络，进行规范裁剪、四方向搜索、开环复查、镜像与轻微平移的一致性检查，并检查眼部图像证据。单点不稳定或被遮挡时屏蔽该点的坐标监督；远侧眼没有支持时也屏蔽 yaw 及双眼中点监督。被拒绝的帧不会自动变成“没有狗”的负样本。

**这批视频的有效侧脸覆盖仍然不足**：159 个训练样本中只有 5 个侧脸样本，其中仅 1 个的双眼与鼻子同时得到支持；验证集没有可靠的侧脸 yaw 标签。镜像增强可以变换方向，不能增加独立狗或独立侧脸的证据。因此该轮不能验证普遍侧脸精度。

另对 Africanis 转头片段的 8 帧检查了 0.35 / 0.45 / 0.55 / 0.65 四种教师裁剪占比；32 组中只有 1 组正面通过一致性筛选，没有补回可靠侧脸（`teacher_scale_probe.json`）。扩大教师裁剪中的脸部比例不足以修正这组样本的远侧眼标注。

微调保持 128 px 输入、35% 脸部占比、24 维输出和原网络结构；训练裁剪直接调用部署用的 `warp_crop(..., antialias=True)`。冻结骨干、存在性、可见性与脸框分支，只更新 FPN 和热图的 107,336 个参数，用原模型约束无监督点及任意旋转方向。RTX 4060 上训练 12 轮，按整段验证视频等权的 NME 与 yaw MAE 之和选择第 4 轮，同时单独报告关键点回归和缺少侧脸验证的问题；综合分下降不等于允许部署。

### 精度与采集结果

下表是**相对教师伪标签的误差，不是人工真值精度**。NME 以教师眼距归一化，yaw 是无量纲比例，不是角度。

| 范围 | 指标 | 原模型 | 候选 |
|---|---|---:|---:|
| 独立验证视频，45 帧 | 平均 NME | 0.1396 | 0.1530 |
| 独立验证视频，44 个支持 yaw 的帧 | yaw MAE | 0.0752 | 0.0641 |
| Wolfhound 回归视频，113 帧 | 平均 NME | 0.1868 | 0.1717 |
| Wolfhound，62 个支持 yaw 的帧 | yaw MAE | 0.1204 | 0.0847 |

原始用户视频 1,314 帧、Wolfhound 1,212 帧及空场景、模糊场景均完整回放。用户视频、空场景和模糊场景都未选图。Wolfhound 离线回放原版选中正面、左侧，候选额外找到了右侧候选；但该右侧图眼睛状态不可靠，独立复检在无损像素上拒绝、JPEG95 上通过，不能以其中一次通过作为验收结论。实时丢帧改变采样后原版也能产生右侧候选，离线覆盖差异不能直接当作采集成功率提升。

### 速度与稳定性

CPU 为 i9-13980HX，GPU 为 RTX 4060 Laptop。单次跟踪调用在相同的 32 张真实裁剪上交替测原版/候选，各 600 次，包含 RGB 转换、同步推理、GPU 传输及输出解析：

| 后端 | 原版平均 / P95 | 候选平均 / P95 |
|---|---:|---:|
| CPU，4 线程，关闭空闲忙等 | 1.38 / 2.38 ms | 1.21 / 2.02 ms |
| CUDA FP32 Graph | 1.22 / 2.67 ms | 1.22 / 2.67 ms |

**单模型耗时不代表完整实时采集耗时。** 按视频原始节奏回放，保留检测、身体门控、方向、跟踪、选图及容量为 1 的工作队列，单轮 CPU 结果如下；不同回放的设备状态和采样也会影响测量，不把这些数值当作稳态速度优劣定论。

| 视频 | 原版 P95 / 丢帧率 | 候选 P95 / 丢帧率 |
|---|---:|---:|
| 用户视频，30 fps | 71.56 ms / 11.72% | 90.56 ms / 22.53% |
| Wolfhound，24 fps | 49.81 ms / 2.72% | 78.86 ms / 6.77% |

本次也修复了 `benchmark_dog_capture_v1.py` 的漏测：原来回放时重新构建 `CaptureEngine` 未传入已经加载的 landmarker，实际又走了检测器跟踪。现在回放显式传入它，校验其执行后端，记录模型哈希和各模型调用次数。模型级测速使用新的 `benchmark_dog_landmark_pair.py`；两种计时口径分开报告。

### 复现入口

在包含本提交的仓库目录下运行；先按训练设备安装 PyTorch，再安装 `requirements-dog-landmark-training.txt`。已有归档无需再次导入。需要从另一个原始 ONNX 建立训练起点时：

```bash
python src/Pet-ReID-IMAG/tools/import_dog_landmark_weights.py --onnx models/pretrained/dog_capture_landmarks_v1/dog_face_landmarks.onnx --output models/pretrained/dog_capture_landmarks_v1/recovered_trainable.pt
```

公共视频下载及标注：

```bash
python src/Pet-ReID-IMAG/tools/fetch_dog_distillation_assets.py --manifest assets/dog_capture/video_distillation_sources.json --root .
python src/Pet-ReID-IMAG/tools/prepare_dog_video_distillation.py --manifest assets/dog_capture/video_distillation_sources.json --root . --teacher models/pretrained/dog_capture_v1/dog_face_pose.onnx --out data/dog_capture/distillation/pilot_v2 --sample-fps 4 --max-per-video 200
```

标注目录保留原帧、逐点权重、拒绝计数、来源快照和 `audit_train.jpg` / `audit_val.jpg` / `audit_test.jpg`，先检查侧脸覆盖与叠加点位再训练。以下工具拒绝覆盖现有数据、训练目录或 checkpoint；重做实验需更换输出路径。

```bash
python src/Pet-ReID-IMAG/tools/train_dog_video_landmarks.py --data data/dog_capture/distillation/pilot_v2 --weights models/pretrained/dog_capture_landmarks_v1/recovered_trainable.pt --out artifacts/runs/dog_capture/video_distillation_20261112/train_pilot_v1 --device cuda --epochs 12 --repeats 4 --batch 16 --lr 0.00005
python src/Pet-ReID-IMAG/tools/benchmark_dog_landmark_pair.py --data data/dog_capture/distillation/pilot_v2 --baseline models/pretrained/dog_capture_landmarks_v1/dog_face_landmarks.onnx --candidate models/pretrained/dog_capture_landmarks_video_pilot_20261113/dog_face_landmarks.onnx --output artifacts/runs/dog_capture/paired_cpu.json --providers cpu
```

导出后进行了 ONNX checker 和 16 张真实裁剪的 PyTorch/ORT 对照，最大输出差 `0.0000305`。Python 测试 103 项通过；SDK 34 项通过、1 项因缺少独立检测器参考照片而跳过，其中候选模型的 4 帧真实 Python/WASM refinement 对照实际运行并通过。速度、连续回放、独立选图复检和测试日志保存在 `artifacts/runs/dog_capture/video_distillation_20261112/`。

下一轮应按内容补数据：以 6–10 只不同狗的短片为小规模采集目标，包含清楚的“正面—左转—正面—右转”、近景及侧躺；先确认左右两侧都有可靠眼鼻标注，再增加帧数。对教师持续放错的远侧眼或闭眼帧，补少量校正锚点及可见性标签。这个数量是采样目标，不是精度保证；回归视频继续隔离，新增侧脸验证视频后再选 checkpoint。
