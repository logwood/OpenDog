# dog_capture v1：用户 ONNX 的本机集成

基于 `codex/local-visibility-20260909` 的 `e2588193abd25e157d87aec87aec550133690cc2`。
2026-11-11 在 Windows 本机完成原生引擎、TypeScript SDK、H5、服务端复检和现有 Re-ID 服务的联调。

**代码与模型接入已完成；这份 12 轮训练参考权重的实际选图验收尚未通过。** 用户视频仍存在关键点定位和方向判断问题，不能作为已达标模型替代旧版。报告把链路测试与模型效果分开记录；本轮没有进行量化或 TensorRT 改造。

后续 CPU / CUDA 实时测速、慢帧与丢帧统计，以及 CUDA 禁用 TF32 的数值一致性修复，见 [本机速度报告](DOG_CAPTURE_V1_SPEED_CN.md)。该报告单独记录当前 ONNX，未混用旧版 TensorRT 成绩；模型效果验收状态不变。

## 1. 接入的文件

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `dog_face_pose.onnx` | 10,845,283 | `48c2b9f580a6b29f6ec92f81d58bffcb06c609b4da4d77e4c85f259dc1b8d74c` |
| `yolox_nano.onnx` | 3,659,407 | `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d` |

- 用户的 ONNX 与 JSON 保存在 `models/pretrained/dog_capture_v1/`，权重未改动。
- 原始元数据、模型目录记录、质量配置和验收规则在 `assets/dog_capture/v1_reference/`、`assets/dog_capture/model_zoo_v1.json`。
- 输入 `images [1,3,320,320]`；输出 `output0 [1,23,2100]`；FP32，opset 17；6 点依次为两眼、鼻、下巴、两侧耳根。
- 用户视频已归档至 `data/dog_capture/user/2611101816341481.mp4`，SHA-256 为 `6db00b4f53f1ac995df48bdd96a3f32879717dfa52af9d1ef793550c827ebc06`，共 1314 帧、43.8 秒。
- 二进制模型、视频、运行日志和交付 ZIP 沿用仓库的本地资产约定，不提交到 Git。模型来源与校验值、导入代码、验证摘要提交到 Git；本地交付包包含实际文件。用户提供的许可原文随元数据保留。

## 2. 安装与运行

以下命令从仓库根目录运行。Python 3.11+（验证版本 3.12.13）；构建前端需要 Node 22.12+（验证版本 24.13.1）。

```powershell
# 已有本机模型时可省略 -Model / -Metadata。-Web 安装服务依赖并构建 H5。
.\scripts\setup-dog-capture-v1.ps1 -Python D:/CondaData/envs/torch312/python.exe `
  -Model C:/Users/ASUS/Downloads/dog_face_pose.onnx `
  -Metadata C:/Users/ASUS/Downloads/dog_face_pose.json -Web

# 视频逐帧选图：质量优先配置；best 在整段视频中选最好的合格候选。
.\dog-capture.cmd data/dog_capture/user/2611101816341481.mp4 `
  --config assets/dog_capture/v1_reference/capture_quality.json --mode best

# 本机 H5 采集与服务端复检
.\scripts\start-dog-capture-v1.ps1
```

打开 `http://127.0.0.1:8765/capture/`。原生输出包含 `result.json`、`frames.jsonl`、`review.html`，以及每个角度的原图、脸部裁剪和 224×224 对齐图。退出码 2 表示没有合格照片；不是程序异常。已有输出目录不会覆盖。

独立采集服务只负责采集、复检和向下游提供照片，不模拟身份识别。服务启动即校验模型，缺失模型或物种门控会明确失败。H5 成功后，业务方可以通过以下接口获取已通过复检的原始 JPEG：

```text
GET /v1/capture/sessions/{session_id}                         # 结果、照片 id
GET /v1/capture/sessions/{session_id}/shots/{shot_id}         # 已接受的 JPEG 原图
```

以上接口沿用业务 API Key 鉴权（`X-API-Key`）；会话默认有效 15 分钟，业务方应在有效期内保存图片。独立服务会话不提供“建档/比对”提交按钮。

Git 新检出需要先导入用户模型；本地交付 ZIP 已含权重、视频与 H5 构建产物。只运行包中的现成 H5 时，不必安装 Node：先安装 `requirements-dog-capture-service.txt`，再启动服务。

## 3. 挂到现有识别服务

已有 `serve_pet_api.py --enable-capture` 路由继续使用，并新增 `--capture-config`。联调使用独立测试图库：

```powershell
D:/CondaData/envs/torch312/python.exe src/Pet-ReID-IMAG/tools/serve_pet_api.py `
  --profile production --device cpu --onnx-provider cpu `
  --storage-dir artifacts/runs/dog_capture/new-integration-gallery `
  --enable-capture --capture-models models/pretrained/dog_capture_v1 `
  --capture-config assets/dog_capture/v1_reference/capture_quality.json
```

也可以直接调用 `mount_capture(app, service, models_dir=..., config=CaptureConfig.load(...))`。
模型下载、SDK 推理、服务端复检使用同一 ONNX 哈希及关键点配置。客户端上传原始画面，识别模型接收完整 JPEG；对齐图用于质量评价及缩略图。

本轮修复了真实识别服务与 H5 的响应契约：识别服务返回 `candidates / predicted_pet_id / top1_score`；建档返回 `added_image_ids / duplicate_image_ids`。页面现在正确展示结果和实际新增数量，并尊重识别服务的 `accepted=false`。

## 4. 本轮修复

- 默认模型登记及导入命令：导入前验证目录固定哈希、JSON 和 ONNX 元数据、实际运行；校验失败不替换现有模型。
- 自动发现和推理初始化检查权重哈希、输入大小、关键点形状及顺序，坏文件不会静默切到别的模型。
- H5 模型 URL 含哈希，浏览器缓存随版本区分；错误版本请求返回 409。
- Windows H5 构建使用 `fileURLToPath`，修复 `D:\D:\...` 路径错误；安装器可修复未完成安装，门控下载失败会终止启动。
- Worker 请求带编号、超时和销毁处理，线程异常会结束等待并显示错误；重拍重新开始倒计时，使用新的视频元素。
- 每张上传图片独立做狗物种判断，清除上一张图片的框缓存；缺失 nonce 的请求不会消耗会话重试次数。
- 增加服务端已接受原图下载接口；质量优先配置要求 `high`，会话策略不能降低服务端设置的质量下限。
- E2E 脚本使用独立端口，失败返回非零，关闭自建进程；Windows 可直接使用已安装 Edge。Y4M 夹具无需 FFmpeg。

## 5. 本机验证结果与边界

机器：i9-13980HX / RTX 4060 Laptop；本轮采集推理、浏览器 WASM 和识别联调均使用 CPU。测试版本与完整摘要见 `assets/dog_capture/v1_reference/integration_report.json`。

- Python：83 项通过；SDK：25 项通过，其中包含真实 ONNX 的 Python / WASM 解码对照；类型检查和 H5 构建通过。
- 浏览器：真实权重 → Worker → 采集 → 正面重拍 → JPEG 上传 → 真实模型复检 → token → 实际 Re-ID 比对/建档通过，控制台无报错。
- 识别使用现有生产基线 `unified_pet_reid_v3_v1` 的真实 ONNX，独立图库完成两张侧脸入库，并正确识别预先登记的测试犬。这是单犬流程测试，不能解释为身份识别准确率。
- 浏览器摄像头素材由公开犬照片的抖动和左右镜像构成，适用于重复验证传输及流程；自然视频另行逐帧测试。端到端耗时与截图均保存。
- 10 段自然/退化视频逐帧运行，无推理崩溃；严重模糊及空场景没有输出，冻结正面没有虚构左右视角。

**模型效果仍有明确缺口：**

| 用例 | 兼容默认配置输出 | 验收结果 |
|---|---|---|
| 用户视频 | 正面候选第 865 帧，未采到左、右 | 未通过；865 帧横躺、眼部遮挡，被模型误判为正面 |
| Wolfhound 自然视频 | 正面、左侧 | 未通过完整三视角覆盖；缺右侧 |
| 严重模糊 / 空场景 | 无照片 | 通过拒绝检查 |
| 冻结正面 | 正面 | 通过“不虚构转头”检查；不代表活体能力 |

`capture_quality.json` 拒绝用户视频中的低质量候选，返回“没有合格照片”，不会把那张照片当作清晰正脸交付。这个配置不会弥补模型关键点定位能力。既有 v0.3.1 / TensorRT 入口及其用户视频正面 271、左侧 282 帧的验证保留，可继续使用原交付包 `delivery/dog_view_selection_v3_1_trt/`。

下一项模型工作应针对横躺近景、闭眼及两侧姿态补充训练/标定，并以这些失败用例重新验收。手机真机、自然动作活体防伪和跨犬开放集准确率均未在本轮得到验证。延迟优化继续排在准确性与采集完整性之后。

## 6. 复现与发布检查

```powershell
$env:PYTHONPATH = "$PWD/src/Pet-ReID-IMAG"
$py = "$PWD/.tmp/dog-capture-v1-env/Scripts/python.exe"
& $py -m pip install -r requirements-dog-capture-validation.txt
& $py -m pytest src/Pet-ReID-IMAG/tests/test_dog_engine.py `
  src/Pet-ReID-IMAG/tests/test_dog_engine_io.py src/Pet-ReID-IMAG/tests/test_dog_engine_onnx.py `
  src/Pet-ReID-IMAG/tests/test_dog_capture_service.py --basetemp ".tmp/tests-$([guid]::NewGuid())" -q

# 使用新的输出目录。当前参考权重应返回 3：效果验收未通过；报告仍完整保存。
& $py src/Pet-ReID-IMAG/tools/validate_dog_capture_v1.py `
  --output artifacts/runs/dog_capture/new-validation --enforce-acceptance

cd src/Pet-ReID-IMAG/frontend/dog-capture
npm ci
npm test
npm run typecheck
npm run build
```

完整浏览器联调：

```powershell
# 从 frontend/dog-capture 运行。照片夹具在本地验证产物 e2e-media，交付包另存至 data/dog_capture/e2e。
$root = (Resolve-Path ../../../..).Path
$env:E2E_MEDIA = "$root/artifacts/runs/dog_capture/v1_integration/e2e-media"
$env:DOG_CAPTURE_MODELS = "$root/models/pretrained/dog_capture_v1"
$env:PYTHON = 'D:/CondaData/envs/torch312/python.exe' # 现有识别模型环境
$env:E2E_BROWSER = 'msedge' # 或 npx playwright install chromium 后删除此变量
$env:E2E_RETAKE = '1'
$env:E2E_CAPTURE_CONFIG = "$root/assets/dog_capture/v1_reference/capture_quality.json"
$env:E2E_REID_MODEL = "$root/models/selected/unified_pet_reid_v3_v1/onnx/e2e/unified_pet_reid.onnx"
$env:E2E_GALLERY = "$root/artifacts/runs/dog_capture/gallery-$([guid]::NewGuid())"
$env:E2E_SEED = "$env:E2E_MEDIA/front.png"
node e2e/run.mjs "$root/artifacts/runs/dog_capture/browser-$([guid]::NewGuid())"
```

移除 `E2E_REID_MODEL` 后使用明确标记的替身识别服务，仅验证协议；它不等同于真实识别验收。
`e2e/make_video.py --front ... --side ... --side-face-x ... --out ...` 可由夹具 PNG 重新生成 Y4M，参数及来源见夹具目录 `source.json`。
真实 WASM 对照测试需设置 `DOG_CAPTURE_REAL`，该目录包含 `frame.bgr`、`ref.json`、`dog_face_pose.onnx`、`yolox_nano.onnx`；未设置时该测试明确跳过。

打包使用 `tools/build_dog_capture_onnx_delivery.py`，输出目录必须不存在。包内 `manifest.json` 校验每个文件，ZIP 另附 SHA-256。原始提交、最终实际文件内容及所有验证记录可追溯。
