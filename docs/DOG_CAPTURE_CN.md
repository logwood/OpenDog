# 犬只转头采集与清晰帧选择 v0.1

这是可运行的电脑端验证版：输入视频或摄像头帧，输出正面 / 左 / 右 / 未知、动作进度、质量分，以及用于后续 Re-ID 的原始正面帧。仅使用视频像素，不调用在线识别接口。手机端移植尚未执行。

## 直接使用

把视频拖到仓库根目录的 `run-dog-capture.cmd`，运行完整逐帧分析；结果默认写入 `artifacts/runs/dog_capture/日期时间-文件名/`。打开其中的 `review.html` 看标注视频和选图。

把视频拖到 `run-dog-capture-live.cmd`，按视频原速度实时回放；按 Esc 结束。实时模式会明确统计未处理的旧帧，不保证分析每个输入帧。完整选图建议使用逐帧模式，避免快速经过镜头的清晰帧被跳过。

命令行（在仓库根目录）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-dog-capture.ps1 -Video "data/dog_capture/user/2611101816341481.mp4"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-dog-capture.ps1 -Video "C:\path\dog.mp4" -Realtime -Show
```

首次使用需要 Python 3.12 和网络来安装隔离环境及模型；下载完成后可离线运行。依赖只安装在 `.tmp/dog-capture-env`，不改变 Re-ID 训练环境。ffmpeg 可选：存在时预览自动转 H.264；缺少时保留 OpenCV 的 MP4 预览，部分浏览器可能不支持，原始图片和 JSON 不受影响。

高级入口：

```powershell
.tmp/dog-capture-env/Scripts/python.exe src/Pet-ReID-IMAG/tools/run_dog_capture.py "C:\path\dog.mp4" --device auto --output artifacts/runs/dog_capture/my-test
.tmp/dog-capture-env/Scripts/python.exe src/Pet-ReID-IMAG/tools/live_dog_capture.py --camera 0 --show --output artifacts/runs/dog_capture/camera-test
.tmp/dog-capture-env/Scripts/python.exe src/Pet-ReID-IMAG/tools/prepare_dog_capture_assets.py --models --public
.tmp/dog-capture-env/Scripts/python.exe src/Pet-ReID-IMAG/tools/prepare_dog_capture_assets.py --user-video "C:\path\new-dog.mp4"
```

`--device auto` 优先使用 LiteRT WebGPU，失败则记录原因并回退 CPU；`--device gpu` 要求 GPU，失败直接报错；`--device cpu` 可作低依赖对照。Windows WebGPU 需要 `dxcompiler.dll` 和 `dxil.dll`，本机使用系统已有的 Edge WebView 运行库，没有修改系统文件。其他电脑可以把 Microsoft 官方 DirectXShaderCompiler 的 `bin/x64` 路径设为进程环境变量 `DOG_CAPTURE_DXC_DIR`。GPU 的首次编译和预热在视频计时之前完成；CPU 上的大角度重新定位更慢，不能把 GPU 成绩当成 CPU 成绩。

## 资产与结果

- `assets/dog_capture/sources.json`：全部原视频的来源、许可、相对路径和 SHA-256。
- `data/dog_capture/user/2611101816341481.mp4`：用户提供的视频，无修改归档；SHA-256 为 `6db00b4f53f1ac995df48bdd96a3f32879717dfa52af9d1ef793550c827ebc06`。
- `data/dog_capture/public/`：5 段 Wikimedia 公开样本，涵盖正面、转头、舔食、远景、栅栏和俯头。
- `models/pretrained/dog_face_landmarks_v1/`：3 个固定哈希的模型，总计约 33.4 MB（十进制）。部署只需清单中的 3 个模型。
- 每次运行独立目录；已有非空目录拒绝覆盖。

结果文件：

| 文件 | 用途 |
| --- | --- |
| `review.html` | 离线复核页面 |
| `preview.mp4` | 带状态、质量和帧号的视频，不含原音频 |
| `frames.jsonl` | 每个已分析帧的原始关键点、姿态、拒绝原因、阶段计时、动作进度 |
| `best_front_frame.png` | 被选中的完整视频帧，保持原始像素 |
| `best_front_face.png` | 对应犬脸区域裁剪，仅作整 90 度转正；没有增强或锐化 |
| `best_left_*` / `best_right_*` | 存在合格侧向候选时输出 |
| `result.json` | 最终选图、识别输入路径、完整动作状态、性能、来源 |

这里的“最佳帧”是合格候选中质量分最高者，不表示人工评价的绝对最清晰帧。评分采用眼睛、鼻口与下颌区域的固定尺度细节、对比度、曝光和有效分辨率；不会用背景草地的清晰度给狗脸加分。候选从稳定停留的开始保留，等停留确认后再正式选入，避免漏掉停留初始较清晰的一帧。对于短暂正面，0.5 秒内两次独立神经网络结果均通过较严格的不确定性门槛，也允许确认主图；这不会推进动作状态，结果中的 `selection_basis` 区分两种选入依据。不同跟踪段不混合左右动作或图片；优先保存完成动作的一段，否则保存正面质量最高的一段。

## 处理过程与接口

1. EfficientDet Lite0 检查画面有且只有一只狗，默认每 0.30 秒刷新。
2. EfficientNetB2 找脸框；MobileNetV3-Large 预测 DogFLW 的 46 个关键点。
3. 从原模型的热图导出不确定性指标，并检查眼部图像证据、眼鼻几何、裁切和姿态。正面选图要求双眼证据；侧转允许远侧眼自然遮挡，同时保留鼻部、热图和时间一致性检查。回归器在背头也可能输出坐标，所以“有坐标”不等于“有可用狗脸”。
4. 网络重定位之间用前后向光流跟踪；漂移、快速变化和超时触发重新推理。大角度转动时尝试 90 度方向归一化，转正方向变化会重置动作序列。
5. 使用双眼中点到鼻端的归一化偏移判断朝向，时间平滑、双阈值和至少 0.25 秒 / 3 帧停留共同确认动作。顺序为正面 → 左 → 正面 → 右 → 正面。遮挡打断停留，持续丢失 0.75 秒会重置。
6. 从合格帧选择原始主图，并写出 `recognition_input`。只有 `sequence_complete=true` 才表示本段完整动作走完；有主图和动作完成是两个不同字段。

左右定义是**转正后的脸部坐标系中，鼻端朝画面左 / 右**，不是狗自身的解剖学左右。`rotation_quarters` 表示对原始脸部裁剪执行 `np.rot90` 的次数。`--mirror` 在输入时统一镜像；默认不镜像。`yaw_cue` 是二维比例，不是以度数计的三维 yaw。

Python 接入（把 `src/Pet-ReID-IMAG` 加入模块搜索路径）：

```python
from dog_capture.models import DogFaceModels
from dog_capture.pipeline import CapturePipeline

pipeline = CapturePipeline(DogFaceModels("models/pretrained/dog_face_landmarks_v1"))
record = pipeline.process(bgr_frame, capture_timestamp_seconds, frame_index)
snapshot = pipeline.snapshot()
# record 可 JSON 序列化；snapshot["frames"]["front"]["image"] 是原始分辨率像素。
```

时间戳必须有限且严格递增，使用采集时间；一个 pipeline 由一个工作线程使用。实时入口将输入队列限制为 1 帧，预览保持当前帧；旧坐标不会画到新帧上，姿态标签超过 250 ms 即过期。JSONL 持续落盘，内存只保存有限候选与最近 4096 条统计。摄像头入口已实现，当前验收使用视频原速回放，没有宣称已测物理摄像头或手机。

当前文件入口按声明帧率生成时间戳，要求固定帧率；可变帧率文件请先转为固定帧率。用户提供的这段视频已用逐帧 PTS 检查为 30 fps。直接接相机或手机时传入实际采集时间戳，不用推理完成时间代替。

## 验证与边界

```powershell
.tmp/dog-capture-env/Scripts/python.exe -m pip install pytest==9.2.0
.tmp/dog-capture-env/Scripts/python.exe -m pytest src/Pet-ReID-IMAG/tests/test_dog_capture.py -q
.tmp/dog-capture-env/Scripts/python.exe src/Pet-ReID-IMAG/tools/validate_dog_capture.py --output artifacts/runs/dog_capture/new-validation
```

`assets/dog_capture/regression_labels.json` 是人工粗标回归点。原视频的模糊、遮挡和方向过渡不能都当成可识别帧；报告分别统计正确方向、错误方向、主动弃判及质量拒绝。开发时使用过这些样本，结果不能充当未见犬种的独立准确率。`validation.json` 保留逐个点的比较。

`build_dog_capture_fixtures.py` 可生成另 4 个工程测试：从公开源剪接的明确动作顺序、静止正面、模糊正面、空画面。它们的修改方法和来源记在 `assets/dog_capture/fixtures.json`；剪接样例仅用于流程回归，不作为自然视频准确率或活体安全证据。

热图熵、眼部亮度和几何阈值目前是可解释的启发式门槛，尚未做跨犬种的大规模标定；毛发遮眼、浅色眼睛、极端俯拍和鼻部近贴镜头可能被拒绝。现阶段已完成采集流程与本地运行，尚未验证支付宝级的产品可靠性。此模块也不判断身份、屏幕重放或其他攻击，不把动作完成当成安全活体结论。

移动端可复用这 3 个 TFLite 文件及同样的裁剪、镜像、时间戳、阈值、队列和状态机逻辑；需要在目标手机上单独测 GPU delegate、热量、延迟及内存，再定输入尺寸和刷新频率。

## 来源与许可

犬脸定位和关键点权重来自 [hugocornellier/dog-face-landmarks](https://huggingface.co/hugocornellier/dog-face-landmarks)，固定 revision `ffe69649279e0c226e7061f917550ca8608abbfe`；架构和输入契约见模型卡，训练代码见 [dog-face-landmarks-training](https://github.com/hugocornellier/dog-face-landmarks-training)。仅给原图增加热图输出，不修改权重；回归测试检查输出坐标不变。

权重遵循 **CC BY-NC 4.0，限非商业用途**；上游训练代码为 Apache 2.0。商业使用需获得 DogFLW 数据权利方的授权。EfficientDet Lite0 来自 Google 官方 TensorFlow Lite 示例模型；运行框架为 [LiteRT](https://github.com/google-ai-edge/LiteRT)。GPU 运行方式参照 [官方 GPU 文档](https://developers.google.com/edge/litert/next/gpu)。

公开视频、预览和截帧的作者、原始链接和 CC BY-SA 版本见资产清单及对应复核页。用户视频作为私人测试素材保存，没有上传公开服务。许可与修改说明也收录在 `assets/dog_capture/NOTICE.txt`。
