# 犬脸优选抓拍 v1 验证记录（2026-11-11）

## 环境与边界

- 云端 Linux，4 vCPU（Xeon 2.3 GHz），**无 GPU**；Python 3.13.18、NumPy 2.5.4、OpenCV 5.0.0、ONNX Runtime 1.31.1（CPU EP）。
  PyTorch 2.15 / Ultralytics 8.4.210 只用于训练和导出，运行时不需要。
- 学生模型训练期间同一台机器上还在做其他测量，所以本文的延迟数字偏保守（上限），不能代表空载 CPU，更不能代表桌面 GPU。
  TensorRT / CUDA / CoreML / 手机执行器**都没有在这里实测**。
- 本机拿不到用户视频 `data/dog_capture/user/2611101816341481.mp4` 和本地的 AnyFace 权重，下面没有与 v0.3 的同视频对照，需要在本地补跑（见最后一节）。

素材（均未提交到 Git）：

| 素材 | 来源 | 校验 |
|---|---|---|
| Ultralytics dog-pose（StanfordExtra 标注，训练 6773 张 / 验证 1703 张） | `github.com/ultralytics/assets` release `dog-pose.zip` | SHA-256 `0396a35f…d4aad667` |
| DogFLW 脸定位与 46 点模型 | pub.dev `dog_detection` 4.1.1 安装包中的 assets | 与仓库已固定的 `bf13f179…`、`4036cd3d…` **完全一致** |
| EfficientDet-Lite0 | 仓库清单中的 Google Storage 地址 | `2e04c53b…`，与清单一致 |
| YOLOX nano / tiny（0.1.1rc0） | GitHub release | `c789161e…` / `427cc366…`，已写入 `model_zoo_v1.json` |
| `01_dog.mp4`：1280×720，29.97 fps，289 帧 | SAM 2 仓库 `demo/data/gallery`（Apache-2.0） | `21425eb9…` |
| `astronaut` / `chelsea`（猫）/ `coffee` / `horse` | scikit-image 自带样例图 | — |

`01_dog.mp4` 拍的是一个人和一只小狗在玩：狗脸小、运动快，画面里一直有人脸。它不是抓拍场景，而是**压力测试**，
用来看会不会误拍、引导提示是否合理。

## 1. 自动化测试

`pytest src/Pet-ReID-IMAG/tests/test_dog_engine*.py`：64 项通过。覆盖配置校验、Umeyama 对齐（多种尺度与旋转）、
姿态的旋转不变性与眼序规范化、视角去抖、One-Euro 滤波、ByteTrack（跳帧、低分延续、两只狗、过期）、质量排序与拒绝、
关键点可见度门控、三种抓拍策略、驻留确认前的暂存、top-k 多样性、两只狗不混用、物种门控（含头部裁剪阈值）、
方向搜索与关键点转正、相册模式、时间戳策略、非法输入、引导防抖、主目标粘性；ONNX 方面用常量输出图验证 YOLO-pose
（含端到端导出格式）、YOLOv5-face、SCRFD、YOLOX 的坐标还原，并通过真实 ONNX Runtime 会话跑完整流水线；I/O 方面覆盖
视频 PTS、坏图片、输出不覆盖、实时丢帧与异常上抛，以及 CLI 全流程（结果 JSON、三视角图片、复核页、逐帧日志、预览视频）。
v0.x 原有测试：48 项通过，4 项因本地专有资产缺失跳过，与改动前一致。

## 2. 导出与解码

- AnyFace 导出（`tools/export_anyface_onnx.py`）：本机没有真实权重，因此用随机初始化的 YOLOv5n-face（3 个检测层）和
  YOLOv5l6-face（4 个检测层，即 AnyFace 的结构）验证。ONNX Runtime 与 PyTorch 输出的最大差分别为 1.8e-4 和 1.2e-4，
  输出形状为 `[1, 6300, 16]` 和 `[1, 9180, 16]`，引擎能正常解码。新工具按模型实际的检测层数重建 `anchor_grid`
  （原 `export.py` 写死为 3 层，不适用于 4 层的 l6）。
- Ultralytics 真实导出：`output0 [1, 23, 3549]`，元数据带 `kpt_shape=[6,3]` 和 `kpt_names`，与解码器一致；在验证图上目视检查了框和关键点的位置。

## 3. 清晰度标定（真实犬脸）

用 `tools/calibrate_dog_quality.py`，在 300 张验证集近正脸（眼距 ≥ 40 px）上测量：先按 PetFace 模板对齐，再与人为加的
失焦、运动模糊比较。最终常数为 `LAPLACIAN_REF=80`，Crete 区间 `0.30–0.70`，`min_sharpness=0.15`。

| 条件 | 清晰度 P10 / P50 / P90 | 被判模糊 | 原图得分更高 |
|---|---|---:|---:|
| 原图 | 0.32 / 0.67 / 0.97 | 3% | — |
| 失焦 σ = 0.03 倍眼距 | 0.015 / 0.022 / 0.081 | 96% | 100% |
| 失焦 σ = 0.06 倍眼距 | 0.012 / 0.013 / 0.016 | 100% | 100% |
| 运动模糊 0.15 倍眼距 | 0.024 / 0.067 / 0.236 | 75% | 100% |
| 运动模糊 0.30 倍眼距 | 0.018 / 0.034 / 0.161 | 88% | 100% |

初始常数（`120`、`0.22–0.62`、`0.20`）会把 10% 的清晰原图判为模糊，所以按实测重新标定。
Stanford Dogs 里本身有压缩和失焦的照片，3% 的原图被拒里有一部分本来就不清晰。

## 4. 物种门控（YOLOX-tiny）

| 输入 | 结果 |
|---|---|
| 人像、猫、咖啡杯、马（整图） | 均无狗框 |
| 验证集整张狗照片 200 张 | 92% 有狗框 |
| 验证集脸框外扩 1.5 倍的头部裁剪 183 个 | 分数 ≥ 0.3：81%；≥ 0.5：72% |
| 猫的头部裁剪 | 被判为狗，得分 0.44 → 因此头部复核单独使用 0.5 阈值 |

v1 引擎、相册模式、四张非狗图：关闭门控时，早期学生模型把宇航员的人脸当成“正面狗脸”收进结果；
开启门控后，人、猫、杯子都被判为“不是狗脸”，马判为“未检测到狗脸”，一张都没有采集。
YOLOX-nano 在 20 张狗照片上检出 18 张，平均 IoU 0.78；YOLOX-tiny 检出 20 张，平均 IoU 0.82，因此默认用 tiny。

## 5. 学生模型（YOLO11n-pose，dog-face6）

（学生模型训练进行中，本节结果尚未测量。）

## 6. 与 v0.x 级联的对比（同一验证子集，每 4 张取 1 张，共 426 张）

（学生模型训练进行中，本节结果尚未测量。）

## 7. 端到端运行

（学生模型训练进行中，本节结果尚未测量。）

## 8. FP16 / INT8

（学生模型训练进行中，本节结果尚未测量。）

## 9. 还需要在本地补跑的验证

1. 用户视频的新旧对比：
   `dog-capture.cmd data\dog_capture\user\2611101816341481.mp4`，与 v0.3 的结果（正面第 271 帧、左侧第 282 帧）比较。
2. 导出本地 AnyFace 权重后，在同一视频上比较三种检测器：
   `python tools/export_anyface_onnx.py --weights ../../models/pretrained/anyface/yolov5l6_best.pt --size 640`。
3. 在 RTX 4060 上用 `--providers tensorrt` 和 `cuda` 测实时延迟（TensorRT 引擎首次运行时构建，之后从缓存加载）。
4. 用 GPU 按第 6 节的配方训练 100 轮，以 640 输入导出；最好再用 AnyFace 做教师并人工抽检侧脸。
5. 在目标手机上用 onnxruntime-android（NNAPI / XNNPACK）测延迟、温度和内存。
