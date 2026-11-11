# 犬脸优选抓拍引擎 v1（`dog_capture.engine`）

v1 把 v0.x 的“转头采集 / 指定位置选帧”重写成一条**和成熟人脸抓拍一致的流水线**：
单阶段人脸式检测（框 + 关键点）→ 物种门控 → ByteTrack 多目标跟踪 → 五点相似变换对齐 →
FIQA 式多因子质量 → 按轨迹的优选抓拍策略 → 操作引导。模型统一走 ONNX Runtime，一份模型
文件覆盖 Windows / Linux / macOS / Jetson / 手机的各类加速器。引擎本身只依赖 NumPy + OpenCV。

v0.x 的模块、入口和验证记录全部保留（`run-dog-*.cmd`、`TargetViewPipeline`、TensorRT 后端），
可以并行对照；v1 不修改它们。

## 1. 为什么重写：问题与对应的成熟做法

| 方面 | v0.x 的问题 | v1 做法 | 借鉴的成熟机制 |
|---|---|---|---|
| 易用性 | 6 个 `.cmd` 入口、3 套独立环境；输出目录必须为空；阈值散落在代码各处；失败时只给原因码 | 一个入口 `python -m dog_capture run 视频/照片/文件夹/摄像头/RTSP`；一个环境；一个 JSON/YAML 配置（未知键直接报错）；输出目录自动加时间戳、从不覆盖；`doctor` / `models` 自检；逐帧中文引导（“请靠近一些”“请保持稳定”“请把头转向画面左侧”）并做防抖 | 人脸录入 SDK 的引导提示与质量反馈 |
| 边端部署 | TFLite + WebGPU + 手工 TensorRT 引擎（绑定 RTX 4060 + Windows + TRT 10.7）；三模型级联，关键点 384×384；权重 CC BY-NC | 单阶段“检测 + 关键点”网络（YOLO-pose / SCRFD / YOLOv5-face 格式均可解码）；ONNX Runtime 执行器自动选择与回退（TensorRT / CUDA / DirectML / CoreML / OpenVINO / NNAPI / XNNPACK / CPU），TensorRT 引擎在目标机自动构建并缓存；FP16 / INT8 工具会和 FP32 对比精度后再给结论；提供 2.7M 参数学生模型的训练配方 | RetinaFace / SCRFD / YOLOv5-Face 单阶段检测；InsightFace 的 ONNX 分发；大模型蒸馏小模型 |
| 稳定与鲁棒 | 画面里有两只狗就整帧拒绝；光流 + 定期重检；四个 90° 方向反复推理；用“眼睛区域够不够暗”判断睁眼，黑狗和浅色眼睛容易判错；时间戳不递增直接抛错；文件按声明帧率算时间；摄像头读失败即结束 | ByteTrack 给每只狗稳定的轨迹 ID，不同轨迹的图片绝不混用；用关键点几何定义“上”（鼻子在眼线下方），直接计算转正，再配合方向锁；关键点可见度来自模型输出，不再用像素亮度阈值；用 YOLOX 判断是不是狗，排除人脸和猫脸；时间戳异常默认丢帧并计数；文件用容器 PTS，支持可变帧率；摄像头 / RTSP 断线后退避重连；内存有上界 | ByteTrack + Kalman；人脸 SDK 的五点几何姿态与一致性检查；One-Euro 滤波（MediaPipe） |
| 成熟度 | 手写 pending / anchor / verification 状态机；清晰度只看裁剪尺度不一的 Laplacian | 质量统一在 224×224 对齐图的固定区域上计算，分项有限值、可解释，按加权几何平均得到总分和 HIGH/MEDIUM/LOW 等级；抓拍策略分 fast / best / hybrid 三种；去抖用“进入带 + 退出带 + 最短驻留” | ISO/IEC 29794-5 质量分项、SeetaFace QualityAssessor、Crete 2007 无参考模糊度；人脸抓拍机的“快速 / 优选 / 快速+优选”模式 |

## 2. 流水线

```
帧（采集时间戳）
 └─ 方向锁：关键点判断画面是否侧放 → 0/90/180/270° 转正（auto，可固定）
 └─ 人脸式检测器：所有狗脸的框 + 关键点(+可见度)            [ONNX Runtime]
 └─ 物种门控：YOLOX 狗身体框包含脸心；特写时对头部裁剪复核     [可选，Apache-2.0]
 └─ ByteTrack：高/低分两阶段 IoU 关联，Kalman 按真实时间预测
 └─ 每条轨迹：
      几何：眼睛按“鼻在下”重排 → yaw / pitch / roll / 眼距，口或下巴线索做一致性检查
      视角：One-Euro 平滑 + 进入带/退出带 + 最短驻留（时间与观测数）
      对齐：Umeyama 相似变换到 PetFace 犬脸五点模板（224×224）
      质量：分辨率、清晰度、眼部细节、曝光、置信度/可见度、运动、姿态贴合、对齐残差
      抓拍：每个视角保留 top-k；驻留确认前先暂存，确认后补入；未确认就离开则丢弃
 └─ 主目标（最大、带粘性）→ 引导提示（0.6 s 多数表决防抖）
```

默认视角沿用 v0.x 的鼻眼比例带：正面 `[-0.13, 0.13]`，左 `[-0.75, -0.25]`，右 `[0.25, 0.75]`。
`yaw` 是无量纲线索，不是角度。结果里的 `approx_deg` 按固定口鼻长度换算，只用于显示，判定从不使用它。
不同检测器对“鼻子”的定义不同（DogFLW 取鼻垫中心，StanfordExtra 取鼻尖），换模型后可以在配置里调整视角带。

## 3. 使用

Windows：把视频、照片或照片文件夹拖到仓库根目录的 `dog-capture.cmd`；摄像头用 `dog-capture.cmd 0`。
Linux / macOS / Jetson：`scripts/dog-capture.sh 视频或摄像头 [选项]`。首次运行会创建
`.tmp/dog-capture-v1-env`，并下载 YOLOX 物种门控（约 20 MB，已固定 SHA-256）。

```bash
cd src/Pet-ReID-IMAG
python -m dog_capture doctor                      # 依赖、执行器、可用模型
python -m dog_capture models                      # 模型清单、哈希、获取方式、许可
python -m dog_capture models --download yolox_tiny
python -m dog_capture run dog.mp4                 # 逐帧分析
python -m dog_capture run dog.mp4 --realtime      # 按原速回放，跟摄像头一样会丢旧帧
python -m dog_capture run 0 --show                # 摄像头实时窗口（需要非 headless 版 OpenCV）
python -m dog_capture run photos/                 # 相册模式：每张独立，挑出每个视角最好的一张
python -m dog_capture run dog.mp4 --views front --mode fast --providers cuda
python -m dog_capture config --write my.yaml      # 导出全部默认参数，修改后用 --config my.yaml
```

已有上游检测结果时（例如服务端已经跑了 AnyFace）：`--detections dets.jsonl`，不会加载任何人脸模型。
每行格式为 `{"frame_index": i, "detections": [{"box": [...], "score": s, "keypoints": [[x,y],...], "schema": "anyface5"}]}`，
也兼容 v0.x 的 `{"frame_index", "observation": {bbox_xyxy, landmarks_xy, confidence}}`。

Python 接入：

```python
from dog_capture.engine import CaptureConfig
from dog_capture.engine.zoo import build_engine

engine = build_engine(CaptureConfig.load('my.yaml'))   # 或 CaptureEngine(config, detector, body_gate)
for frame, t, i in frames:                             # BGR uint8、采集时间（秒）
    result = engine.process(frame, t, i)               # 也可传 detections=[...] 复用上游检测
    show(result.guidance['message'], result.captured_views)
best = engine.finalize()                               # 一条轨迹 = 一只狗，不会混合
for view in best.captured():
    shot = best.best(view)                             # shot.aligned（224×224）、shot.frame、shot.quality
```

`process()` 不读写磁盘、不联网，内存有上界。它需要在单个工作线程里调用。实时场景使用
`LatestFrameRunner`：队列容量为 1，推理跟不上时丢掉旧帧并计数，不会越积越多。

## 4. 输出

| 文件 | 内容 |
|---|---|
| `views/<视角>_aligned.png` | 224×224 对齐图（PetFace 模板），可直接送比对模型 |
| `views/<视角>_face.png` / `_frame.png` | 原始像素的脸部裁剪和整帧；只有自动转正时做过整 90° 旋转 |
| `result.json` | 选中轨迹、每个视角的框/关键点/姿态/质量分项、`comparison_inputs`、其他轨迹、统计（处理 fps、丢帧、各阶段 P50/P95）、引导统计、模型和执行器 |
| `review.html` | 离线复核页（浅色 / 深色，手机可看） |
| `frames.jsonl` | 逐帧完整诊断（`--no-frame-log` 可关闭） |
| `preview.mp4` | 带标注的诊断视频（`--preview`） |

退出码：0 表示至少采集到一个视角，2 表示一个都没有（原因见引导统计）。

## 5. 模型

模型放在 `models/pretrained/dog_capture_v1/`，也可以用 `DOG_CAPTURE_MODELS` 指定。`detector.kind=auto`
会按下表优先级选择第一个可用的模型。清单在 `assets/dog_capture/model_zoo_v1.json`。

| 名称 | 类型 | 关键点 | 体量 | 许可 | 获取 |
|---|---|---|---|---|---|
| `dog_face_pose` | YOLO11n-pose | 双眼、鼻、下巴、双耳根（带可见度） | 2.7M 参数，320–640 输入 | Ultralytics AGPL-3.0；StanfordExtra 仅限研究；用 DogFLW 做教师时还继承其非商用条件 | `tools/train_dog_face_pose.py`（见第 6 节） |
| `anyface_yolov5l6` | YOLOv5-face | 双眼、鼻、双嘴角 | 约 76M 参数，适合桌面 GPU | AnyFace 代码 MIT、YOLOv5-face 代码 GPL-3.0；训练数据为研究数据集 | `tools/export_anyface_onnx.py --weights models/pretrained/anyface/yolov5l6_best.pt`（本地已有 .pt） |
| `scrfd_dog` | SCRFD | 5 点 | 0.5–2.5 GFLOPs | 取决于训练数据 | 用 InsightFace SCRFD 在犬脸 5 点数据上训练 |
| `dogflw_legacy` | v0.x 三模型 TFLite 级联 | 46 点简化为双眼 + 鼻 | 约 33 MB | CC BY-NC 4.0，禁止商用 | 已有 v0.x 资产；需要 `ai-edge-litert` |
| `yolox_tiny` / `yolox_nano` | YOLOX（COCO） | 物种门控 | 5.1M / 0.9M 参数 | Apache-2.0 | `python -m dog_capture models --download yolox_tiny` |

权重文件不进 Git（`.gitignore` 已排除 `models/pretrained/`），清单记录来源、哈希和许可。

## 6. 训练边端犬脸模型（推荐路线）

人脸领域的常规做法：用单阶段“检测 + 关键点”小网络，在真实标注上训练；缺少的标注由大模型蒸馏补齐。

1. **数据**：Ultralytics dog-pose（StanfordExtra 标注于 Stanford Dogs，训练 6773 张，验证 1703 张）。
   `prepare` 取其中的面部点（鼻、下巴、双耳根），由可见面部点外扩 18% 得到脸框；鼻子不可见或可见面部点少于 3 个的实例丢弃，没有脸的图作为负样本保留。
2. **补眼睛**：该数据集**没有眼睛标注**（StanfordExtra 的第 20–23 点全为空）。`pseudolabel` 用教师模型补眼：
   可以用本地 AnyFace 的 ONNX 导出，也可以把 DogFLW 46 点模型直接跑在真值脸框上。只有教师鼻子与真值鼻子的距离在 0.6 倍眼距以内、
   且眼睛置信度不低于 0.5 时才写入；其余眼睛保持未标注。0.6 留出了两种鼻子定义的差异
   （正脸上中位相差 0.35 倍眼距）。
3. **训练 / 导出 / 评估 / 优化**：

```bash
python tools/train_dog_face_pose.py prepare --source datasets/dog-pose --output datasets/dog-face6
python tools/train_dog_face_pose.py pseudolabel --data datasets/dog-face6 --teacher <AnyFace.onnx 或 DogFLW 目录>
python tools/train_dog_face_pose.py train --data datasets/dog-face6/dog-face6.yaml --epochs 100 --imgsz 640 --device 0
python tools/train_dog_face_pose.py export --weights runs/pose/dog_face6/weights/best.pt --imgsz 640
python tools/evaluate_dog_face_detector.py --data datasets/dog-face6 --model models/pretrained/dog_capture_v1/dog_face_pose.onnx
python tools/optimize_dog_onnx.py --model models/pretrained/dog_capture_v1/dog_face_pose.onnx --fp16 --int8 --calib <你的相机帧>
python tools/calibrate_dog_quality.py --data datasets/dog-face6        # 在你的数据上重新标定清晰度常数
```

Ultralytics 还能直接导出 TFLite / CoreML / NCNN / RKNN / OpenVINO / TensorRT，用于各类设备。
引擎侧只需要 ONNX；其他格式需要在对应设备上用同样的解码方式接入。

## 7. 部署矩阵

| 平台 | 安装 | `--providers` |
|---|---|---|
| Windows + NVIDIA | `onnxruntime-gpu`（CUDA 12 / cuDNN 9） | `tensorrt`（首次运行构建并缓存引擎）或 `cuda` |
| Windows 其他 GPU | `onnxruntime-directml` | `directml` |
| Linux / Jetson | `onnxruntime-gpu`（Jetson 使用 NVIDIA 提供的 wheel） | `tensorrt` / `cuda` |
| macOS / iOS | `onnxruntime`（自带 CoreML EP）/ onnxruntime-objc | `coreml` |
| Intel CPU / iGPU / NPU | `onnxruntime-openvino` | `openvino` |
| Android | onnxruntime-android（AAR） | NNAPI / XNNPACK / QNN |
| 树莓派等 ARM | `onnxruntime` | `cpu`（可配合 INT8，前提是经过对照验证） |

`auto` 按 TensorRT → CUDA → DirectML → CoreML → OpenVINO → CPU 的顺序选择，失败时自动回退，并在 `result.json` 里记录实际使用的执行器和回退原因。
引擎逻辑（跟踪、对齐、质量、抓拍、引导）约 1 500 行纯 NumPy/OpenCV 代码，不依赖框架，可以按原样移植到 Kotlin / C++。

## 8. 配置要点

`python -m dog_capture config --write my.yaml` 会导出全部参数。常改的几项：

```yaml
pose:
  views:                         # 位置带可以自定义；名称会成为输出文件名
    - {name: front, center: 0.0, lower: -0.13, upper: 0.13, max_roll_deg: 30}
  min_dwell_s: 0.12              # 视角至少保持这么久……
  min_dwell_obs: 3               # ……并且至少这么多次观测
selection:
  mode: hybrid                   # fast=首张达标即锁定；best=整条轨迹最好；hybrid=达标后再优化 2 s
  good_enough: 0.65
  top_k: 1
quality:
  min_iod_px: 18                 # 原图眼距下限
  min_sharpness: 0.15
body_gate:
  enabled: auto                  # auto=有 YOLOX 就启用；off=不做物种判断
runtime:
  providers: auto
  rotate: auto                   # 手机侧放的视频自动转正；也可以固定为 0/90/180/270
  timestamps: drop               # 时间戳不递增时丢帧（raise=抛错，适合 SDK 自测）
```

## 9. 本次验证（2026-11-11，云端 Linux 4 核 CPU，无 GPU）

见 [`DOG_CAPTURE_ENGINE_VALIDATION_CN.md`](DOG_CAPTURE_ENGINE_VALIDATION_CN.md)。要点：

- 新增 61 项测试，覆盖配置、几何、对齐、跟踪、质量、抓拍策略、方向锁、物种门控、全部 ONNX 解码器（用常量输出图验证坐标还原）、
  真实 ONNX Runtime 会话、视频 / 图片源、实时丢帧、CLI 全流程；v0.x 原有的 52 项测试保持通过。
- 质量常数在 300 张真实正脸上标定：清晰原图有 3% 被判模糊；轻微失焦（0.03 倍眼距）有 96% 被拒，
  0.15 倍眼距的运动模糊有 75% 被拒，且原图得分始终高于自身的模糊版本。
- 物种门控：人像、猫、杯子、马均未检出狗；完整的狗照片 92% 有狗框，头部特写裁剪 81% 能确认是狗。
- 学生模型（YOLO11n-pose，在本机 CPU 上以 320 输入训练 12 轮，DogFLW 作为补眼教师）的数字见验证文档。
  这只用来证明整条配方能跑通，不能当作发布模型；正式模型应按第 6 节在 GPU 上训练 100 轮。

## 10. 局限

- 眼睛的伪标签来自教师模型，教师在强侧脸上不可靠，所以学生对强侧脸的眼睛也弱。要更准，需要人工补标侧脸，
  或者换更强的教师（AnyFace 加人工抽检）。
- 视角带是鼻眼几何比例，会随品种（口鼻长短）和检测器的鼻子定义变化，需要用自己的视频确认。
- 质量分只覆盖成像质量（清晰、曝光、尺寸、可见度），还不包括“比对模型认不认得出来”。人脸领域会用 MagFace / SER-FIQ 这类
  可识别度指标，这里留作下一步，可以接入现有 Re-ID 模型来做。
- 物种门控依赖 COCO 的“dog”类别；狗身被大面积遮挡、只露半张脸时可能误拒。门控可以关闭或调低阈值。
- 不做活体检测或防翻拍。

## 11. 与 v0.x 的关系

- 旧入口和旧 API 原样保留。`--detections` 兼容 v0.x 的 `FaceObservation` JSONL，`dogflw` 检测器复用旧的三模型 TFLite 资产，
  两套结果可以在同一段视频上对比。
- 输出字段改名：`comparison_inputs` 现在指向对齐图；`coverage_complete` / `missing_views` 含义不变；
  v1 不再有 `sequence_complete`（动作序列不代表活体）。
