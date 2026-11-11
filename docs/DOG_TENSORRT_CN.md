# TensorRT 指定位置选帧

本模块从视频中独立收集正面、左侧、右侧的清晰原图。沿用 v2 的位置范围、短时确认、质量门控和候选原帧复核；TensorRT 仅替换独立视频入口的推理后端。已有脸框与五点的系统仍可直接使用 `TargetViewPipeline()`，新增模型调用为零。

本文和本地 TensorRT 包对应 `run-dog-trt*.cmd` 与 `tools/*dog_views.py` 兼容入口。仓库新增的 `python -m dog_capture`、SDK 和 H5 使用独立 v1 引擎，沿用它们各自的运行说明与验证记录；两套入口共存。

## 本机运行

在仓库根目录，用 Python 3.12 首次安装并构建：

```powershell
powershell -NoProfile -File scripts/setup-dog-trt.ps1 -Python python
```

TensorRT 依赖下载约 1 GB，安装和引擎构建只发生一次。预构建引擎对应 Windows、RTX 4060 Laptop、TensorRT 10.7.0.post1。更换 GPU、操作系统或 TensorRT 版本后执行 `-Rebuild`；引擎检查会拒绝已知不匹配项。

把视频拖到 `run-dog-trt.cmd` 做离线选帧，拖到 `run-dog-trt-live.cmd` 做实时回放。也可直接运行：

```powershell
powershell -NoProfile -File scripts/run-dog-trt.ps1 data/dog_capture/user/2611101816341481.mp4 -NoPreview -Output artifacts/runs/dog_capture/my-trt-run
powershell -NoProfile -File scripts/run-dog-trt.ps1 data/dog_capture/user/2611101816341481.mp4 -Realtime -Show -Output artifacts/runs/dog_capture/my-trt-live
```

输出目录必须为空。`result.json` 给出每个位置的原帧、裁剪、时间、质量与复核结果；缺失的位置明确列在 `missing_views`。`review.html` 可查看离线结果。`--trt-engines` 可指定其他引擎目录；`--no-cuda-graph` 用于直接入队的对照实验。

用户视频已原样收录到 `data/dog_capture/user/2611101816341481.mp4`，哈希和来源位于 `assets/dog_capture/sources.json`。视频、模型和生成结果留在本地交付包；Git 保存源代码、清单和复现说明。公开视频可通过 `tools/prepare_dog_capture_assets.py --public` 重取。

## 加速结构

- 固定 batch=1 与输入大小；脸定位 224×224、关键点 384×384、狗检测 320×320。
- 三个神经网络均使用 TensorRT FP16。坐标期望与熵相关计算强制 FP32，避免精度损失集中在判断阈值附近。
- GPU 内完成 46 个关键点熵的归约，主机只读回 92 个坐标分量和 46 个熵值。
- 保持关键点热图的空间维度连续，避免 ONNX 直转后跨通道步进的长 Softmax；沿用原模型实际存储的坐标网格，并用 FP64 参考值验证坐标与熵。
- 检测器在 GPU 解码全部 19,206 个候选框并归约类别；CPU 仅执行小规模 NMS。先保留每个锚点的最高分类，再筛选狗类别，不提前丢弃 Top-K 以外候选。
- 预分配页锁定主机内存与显存，CUDA Graph 复用上传、执行、下载调度；同步完成后才计时结束。
- 原有光流跟踪、定期重新检测、身份/方向重置和候选原图复核保留。
- v0.3.1 按需生成 RGB 与跟踪灰度图，无狗/等待重试时省掉无用转换；眼部检查仅转换两块原始眼部区域。图像、门槛和模型输出保持一致。
- 指定位置实时入口默认保留最近 3 张待处理帧，吸收多方向重捕获的短暂耗时峰值，减少短暂转头被跳过。缓冲满时替换最老待处理帧；原始时间戳、跳帧、替换数和排队延迟均显式记录。采集与显示独立运行，缓冲不会无限增长。`--queue-capacity 1` 可复现旧的最新帧模式；动作挑战与 SDK 工作者默认仍为 1。

EfficientDet 的原始权重为 INT8 量化模型。ONNX 导出将权重反量化并移除中间激活量化，因此检测输出应做语义回归，不能声称逐位一致。输入保留原始 0–255 像素域，由导出图执行归一化，避免重复归一化。

TensorRT 运行时只需要 NumPy、OpenCV、CUDA Python 和 TensorRT。TensorFlow/tf2onnx 仅安装到独立转换环境；PyTorch 不参与该入口。

## Windows 间歇推理与显卡电源策略

本机默认策略会在间歇推理时将显卡降到低频，因此连续离线吞吐不能代替真实采集延迟。验证报告保留默认电源策略下的完整回放结果。

安装脚本创建独立的 `dog_capture_trt.exe`，复制当前 Python 的解释器和两份 DLL 到本工具虚拟环境，避免给通用 `python.exe` 设置驱动规则。`run-dog-trt*.cmd` 自动使用此独立程序；不会复制到系统目录。

应用级 NVIDIA“最高性能”设置只匹配这个专用可执行文件，不修改全局电源计划、显卡功率限制或时钟。本机已经在用户授权后启用。安装到新目录时，需为新目录中的专用程序重新配置；交付包不会携带本机驱动配置或自动修改新电脑的策略：

```powershell
# 查询（默认只读）
.tmp/dog-trt-env/Scripts/python.exe src/Pet-ReID-IMAG/tools/configure_dog_trt_windows.py
# 开启 / 撤销本工具专属配置（需要时弹出 Windows 管理员确认）
powershell -NoProfile -File scripts/enable-dog-trt-performance.ps1
powershell -NoProfile -File scripts/enable-dog-trt-performance.ps1 -Disable
```

脚本在保存后重新打开驱动会话校验，保存前后状态与退出码到 `artifacts/runs/dog_capture/profile-*`；已处于目标状态时不重复写入。配置生效需重新启动采集进程。最高性能是驱动偏好，不是锁定频率，也不承诺延迟上限；实测范围见验证报告。配置工具拒绝修改通用 `python.exe` 和其他应用的配置，也拒绝删除已加入额外自定义设置的配置。

某些启动器会删去 `ALLUSERSPROFILE` 环境变量。本机驱动 566.07 在这种情况下静默读取出厂配置库，导致已保存的专用配置查询不到。v0.3.1 在加载 CUDA/NVAPI 前通过 Windows API 补齐该进程的变量，保留已有值，不写系统环境或注册表。普通权限与管理员查询结果现已一致。

## 复现与测量

```powershell
.tmp/dog-trt-env/Scripts/python.exe src/Pet-ReID-IMAG/tools/prepare_dog_trt.py --check
.tmp/dog-trt-env/Scripts/python.exe src/Pet-ReID-IMAG/tools/validate_dog_trt.py --output artifacts/runs/dog_capture/new-trt-validation
.tmp/dog-trt-env/Scripts/python.exe src/Pet-ReID-IMAG/tools/verify_dog_capture_outputs.py artifacts/runs/dog_capture/new-trt-validation
# 配置开启后的完整实时重复测试；按用户视频、公开视频顺序执行
.tmp/dog-trt-env/Scripts/python.exe src/Pet-ReID-IMAG/tools/validate_dog_trt_realtime.py --repeats 3 --output artifacts/runs/dog_capture/new-trt-realtime
# 可加 --controls，对普通 Python 做串行交替对照；可加 --queue-capacity 1 对照旧队列
```

验证脚本顺序执行 CPU、原 WebGPU 路径、TensorRT 直接入队和 CUDA Graph，包含模型输出误差、完整视频回归、用户视频实时回放。运行期间应避免其他推理或引擎构建任务争用 GPU。`--reuse-existing` 复用完成的结果，`--summarize-only` 只重新汇总。

离线 FPS 包含视频解码、逐帧分析和 JSONL 写入，关闭预览编码；单帧计算耗时包含该帧的门控与候选复核。实时延迟从已解码帧提交开始，到推理结果可用结束，不包含摄像头传感器、解码和显示。真实神经网络帧单独统计，缺狗或拒绝帧不会被当作昂贵帧的速度。

位置使用无量纲的鼻子与双眼相对关系，不是标定后的角度；本模块不输出活体认证结论。模型与素材的原有许可见 `assets/dog_capture/NOTICE.txt`。TensorRT 面向 NVIDIA GPU；CPU 和手机端需要分别测量。

## 参考

- [NVIDIA TensorRT CUDA Graph 使用与优化](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/optimization.html)
- [NVIDIA TensorRT 引擎兼容性](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)
- [tf2onnx 官方转换工具](https://github.com/onnx/tensorflow-onnx)
- [NVIDIA 官方 NVAPI 驱动配置接口](https://github.com/NVIDIA/nvapi)

本次实测数值与限制见同目录的 `DOG_TENSORRT_VALIDATION_CN.md`。
