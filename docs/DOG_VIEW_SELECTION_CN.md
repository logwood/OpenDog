# 指定位置选帧接口（v0.2 / v0.3 兼容）

这版负责给现有比对模型提供少量合格原图。默认目标是正面、朝左、朝右，每个位置保留一张；位置可改，收集顺序不限，无需每转一次都回正。旧版动作序列入口仍可用。

## 推荐接法：复用现有输出

```text
视频帧 + 现有模型的狗/跟踪确认、脸框、5 点
  → 几何位置范围 → 脸部与眼鼻清晰度 → 短时确认
  → 每个位置一张原图/裁剪 → 现有比对模型
```

`TargetViewPipeline()` 不加载模型，不导入 torch、SAM、Re-ID 或 LiteRT，不联网。仓库的 `pet_id.localization.FaceDetection` 可以直接适配；调用方要提供狗的确认、跟踪 ID，以及属于**同一帧**的检测结果。

```python
from dog_capture.observations import FaceObservation
from dog_capture.views import TargetViewPipeline

selector = TargetViewPipeline()  # 只创建一次，由一个工作线程调用

# 放在现有视频处理循环里；不要为了本模块重新运行一次 detector。
observation = FaceObservation.from_anyface(
    face_detection,       # 已有 FaceDetection：bbox_xyxy / landmarks_xy / confidence
    frame_index=index,
    time_s=capture_time,  # 采集时间，必须严格递增
    track_id=dog_track_id,
    dog_count=dog_count,
)
record = selector.process(frame_bgr, capture_time, index, observation)

# 没有检测结果时传 observation=None，会拒绝该帧，不启动后备模型。
# 采集结束时：
selector.finalize()
bundle = selector.snapshot()
for view, item in bundle['frames'].items():
    original_bgr = item['image']
    box_xyxy = item['record']['box']
    points_xy = item['record']['landmarks']  # 原图坐标
    # 把 original_bgr / box_xyxy / points_xy 交给已有比对流程。
```

视频可以按原帧率采集；如果现有模型只在部分帧推理，就仅提交那些帧及其真实时间。默认相邻有效观测间隔不得超过 200 ms。不能把旧帧关键点套在新帧上，也不能用重复时间戳凑确认次数。`LatestFrameWorker.submit(..., observation=...)` 可复用实时队列，SDK 默认容量仍为 1。v0.3.1 的独立指定位置实时入口默认容量为 3，以保留耗时峰值期间的短暂转头；可通过 `--queue-capacity 1` 还原最新帧模式，队列总是有界。

五点顺序为两眼、鼻顶、两嘴角，与仓库 AnyFace 一致。默认根据现有五点几何确定整 90°转正方向，不再调用模型；也可显式传 `rotation_quarters=0..3`。其鼻顶与 46 点模型的鼻中心定义不同；两者都是二维比例，均不输出真实转角。AnyFace 本身识别动物脸，调用方应使用已有狗检测/业务上下文确认物种，不能把其类别直接当作狗的证明。

## 选图规则

1. 默认正面范围 `[-0.13, 0.13]`，左侧 `[-0.75, -0.25]`，右侧 `[0.25, 0.75]`；侧面优选中心为 `±0.45`，允许侧倾不超过 20°。这些范围是**鼻子相对双眼的几何比例**，不是左右转动角度。
2. 只在脸部区域评分，结合眼睛、鼻子的局部细节，限制低分辨率、模糊、过曝和异常几何。姿态范围先筛选，范围内以清晰度为主，适度偏好目标中心。
3. 每个位置独立确认：至少 3 次可信姿态观测、持续至少 120 ms，至少两次新模型观测相隔 80 ms。进入位置用严格范围，离开时增加 0.04 的迟滞余量。短暂模糊可辅助确认位置，但不能成为输出图片；遮挡、过暗、异常几何或明显位置跳变仍会中断确认。清晰候选与姿态确认分开，避免漏掉夹在短暂模糊之间的清晰峰值。此过程不是活体检测。
4. 原始视频后备路径中，光流帧只有可能替换当前最优图时才补一次关键点核验；核验的是候选图本身。已有五点输入路径复用同帧输出，不追加神经网络。
5. 不同 track ID、超过 750 ms 的失踪、尺寸/方向突变或多狗干扰会分段。最终选覆盖位置最多的一段，同等覆盖时优先含正面、再比较分数，不拼接不同跟踪段的狗。

每个位置只有一个最终输出。内部至多保留当前 3 张、一个待确认候选、已归档最佳段 3 张，事件队列也有上限。`process()` 不写磁盘；诊断日志由调用方选择是否保存。

`assets/dog_capture/view_targets.json` 可改变需要的位置、范围和优选中心；仅需正面时只保留 front 项。API 也可传入 `targets=(ViewTarget(...), ...)`。目标范围不得重叠。

## 独立视频与实时入口

把视频拖到 `run-dog-views.cmd` 做逐帧选图，拖到 `run-dog-views-live.cmd` 做实时窗口回放。旧入口 `run-dog-capture*.cmd` 保持动作序列模式。

```powershell
.tmp/dog-capture-env/Scripts/python.exe src/Pet-ReID-IMAG/tools/select_dog_views.py `
  data/dog_capture/user/2611101816341481.mp4 --output artifacts/runs/dog_capture/my_views --no-preview

.tmp/dog-capture-env/Scripts/python.exe src/Pet-ReID-IMAG/tools/live_dog_views.py `
  --video data/dog_capture/user/2611101816341481.mp4 --output artifacts/runs/dog_capture/my_live --show
```

`--device cpu|gpu|auto|trt` 控制后备模型；默认 auto。TensorRT 的安装与专用入口见 [DOG_TENSORRT_CN.md](DOG_TENSORRT_CN.md)。`--target-config` 加载目标配置；`--no-preview` 省去诊断视频编码。文件回放按声明的恒定帧率取时间，非恒定帧率素材应先规范化；SDK 接收真实采集时间。`--mirror` 会先翻转画面，左右定义随之改变。

复现已有输出的集成试验时可用 `--upstream-jsonl file --fps 15 --no-preview`。JSONL 每行是 `{"frame_index": 0, "observation": {...}}`，缺检测用 null；内部对象与 `FaceObservation` 字段一致。该模式完全不加载后备模型，不允许再镜像输入。缓存模式的耗时**只代表新增选帧成本**。

原始输入入口复用三份已有 TFLite：狗检测、脸定位、关键点。它适合在没有上游输出时验证视频；接入已有模型时无需同时运行这三份模型。

## 输出与后备模型改进

`result.json` 的 `comparison_inputs` 按位置给出原始整帧、裁剪文件、帧号、框和姿态；`selected` 含评分及确认依据。`coverage_complete` 只表示指定位置图像收齐，缺什么由 `missing_views` 给出。`sequence_complete` 在选帧模式始终为 false，不能作为活体结果。

整帧像素保持输入内容，裁剪最多作整 90°旋转，不锐化、不修复、不合成。主图是通过筛选的已分析候选里按评分选出的一张，不声称是视频中人工评价绝对最清晰的一张。

后备关键点图把置信度熵归约放进 TFLite 图内，每次输出从 6,783,344 字节降到 552 字节，减少传输与应用端数组运算。网络主干、46 点预测和预训练权重均保持不变；这不等于把模型体积或全部计算量缩小相同比例。为避免 fp16 下 `log(0)`，熵计算使用 1e-6 下限，CPU/GPU 数值偏差已记录。

可移植图为 `models/pretrained/dog_face_landmarks_v2/dog_view_landmarks_compact.tflite`，输入 `[1,384,384,3] RGB float32/255`，输出坐标和 46 个熵值。导出脚本为 `tools/export_dog_view_model.py`。桌面 CPU/GPU 已测；手机 delegate 和真机耗时尚未测量。现有五点模型复用路径不需要这份后备模型。

后备脸部权重继承原来源 CC BY-NC 4.0，来源和模型哈希见 `assets/dog_capture/NOTICE.txt`、`models.json`；复用上游模型时应采用上游本身的授权。

## 验证入口

```powershell
.tmp/dog-capture-env/Scripts/python.exe -m pytest `
  src/Pet-ReID-IMAG/tests/test_dog_capture.py src/Pet-ReID-IMAG/tests/test_dog_views.py -q

.tmp/dog-capture-env/Scripts/python.exe src/Pet-ReID-IMAG/tools/validate_dog_capture.py `
  --mode views --include-fixtures --no-preview --output artifacts/runs/dog_capture/recheck_views

.tmp/dog-capture-env/Scripts/python.exe src/Pet-ReID-IMAG/tools/benchmark_dog_views.py `
  --device gpu --output artifacts/runs/dog_capture/recheck_benchmark
```

测量结果与适用范围见 `DOG_VIEW_VALIDATION_CN.md`。v1 交付目录和 zip 保留，未覆盖。
