# 狗脸采集协议 `dog-capture/1` 与 H5 接入指南

这一层只负责“识别之前”的一环：在手机或浏览器里引导用户拍狗，选出最适合识别模型的几张原图，
交给服务端复检，然后凭一次性 token 进入现有的建档 / 比对接口。结构参照支付宝和常见人脸认证 H5：

```
业务后端 ──① 创建会话(API Key)──▶ 采集服务            用户手机 / WebView
   │        ◀─ session_id + nonce ─┘                     │
   └──② 打开 /capture/?session=…&nonce=… ────────────────▶│
                                                          │③ 拉配置 + 模型（sha256 校验）
                                                          │④ 端上实时检测 / 跟踪 / 质量评估 / 引导 / 选图
                                                          │⑤ 上传采集包（manifest + 原图 JPEG）
                采集服务 ◀────────────────────────────────┘
                 │⑥ 服务端复检：哈希、重新检测、物种、角度、质量、识别模型一致性
                 │⑦ 通过 → 签发一次性 capture_token（HMAC，10 分钟）
业务后端 ──⑧ 查询结果(API Key) / ⑨ commit(token) ──▶ 现有 enroll / identify
```

端上的判断只决定“拍哪几张”，**是否采信由服务端复检决定**；客户端无法伪造通过结果。

## 1. 快速开始

```bash
cd src/Pet-ReID-IMAG/frontend/dog-capture
```

```bash
npm install && npm run build      # 生成 packages/h5/dist（含 .br/.gz 预压缩文件）
```

```bash
python src/Pet-ReID-IMAG/tools/serve_pet_api.py <原有参数> --enable-capture --capture-models <模型目录>
```

- `<模型目录>` 放 `dog_face_pose.onnx`（+ 同名 `.json`）和 `yolox_nano.onnx`，见
  [`DOG_CAPTURE_ENGINE_CN.md`](DOG_CAPTURE_ENGINE_CN.md) 第 5 节；也可用环境变量 `DOG_CAPTURE_MODELS`。
- 打开 `http://127.0.0.1:8000/capture/` 是开发入口（本机直接建会话）；正式接入由业务后端建会话。
- 不加 `--enable-capture` 时，原有接口和行为完全不变。

## 2. 采集策略（policy）

“宁多勿少，但每张都要对模型有用”：每个角度可以要多张，张与张之间至少隔 `min_gap_s`，
都是服务端能重新检测到、通过同一套质量门槛的**原始整帧**（识别模型在图内自己找脸，所以不上传裁剪图）。

| 策略 | 默认用途 | 正面 | 朝左 | 朝右 | 时限 | 说明 |
|---|---|---|---|---|---|---|
| `enroll_full` | 建档 | 2 张，必选 | 1 张，必选 | 1 张，必选 | 90 s | 左右顺序每次随机（为后续活体预留） |
| `identify_quick` | 比对 | 2 张，必选 | 1 张，可选 | 1 张，可选 | 45 s | 必选角度拍好即结束，侧面顺便拍到就一起上传 |
| `front_only` | 简易场景 | 2 张，必选 | — | — | 45 s | |

共同参数：`hybrid` 优选（达到 0.65 立即入选，再优化 1–1.5 s 后定稿），最低质量等级 `medium`，
每个会话最多 5 次上传。拍满后一直达不到 0.65 的角度，在 2 倍优化窗口后自动定稿，流程不会卡到超时。
引导只在一个角度定稿后才让狗狗转向下一个角度，定稿前一直提示“很好，请保持这个角度”。

策略定义在 `dog_capture/service/protocol.py` 的 `POLICIES`，新增策略只需加一项。

## 3. 接口

所有路径在 `/v1/capture` 下；错误统一返回 `{"code": "...", "message": "..."}`。

| 方法与路径 | 调用方 | 鉴权 | 作用 |
|---|---|---|---|
| `POST /sessions` | 业务后端 | API Key（`X-API-Key`） | 创建会话 |
| `GET /sessions/{id}/config?nonce=` | H5 | nonce | 取策略、引擎配置、模型清单、上传规范 |
| `GET /models/{name}` | H5 | 无（模型公开、按 sha256 缓存） | 下载模型 |
| `POST /sessions/{id}/package` | H5 | manifest 内的 nonce | 上传采集包，同步返回复检结果 |
| `GET /sessions/{id}` | 业务后端 | API Key | 查询状态、复检结果、`capture_token` |
| `POST /sessions/{id}/commit` | 业务后端（或 `client_commit` 会话的 H5） | API Key 或 `client_commit` | 用 token 建档 / 比对 |

**创建会话**

```json
POST /v1/capture/sessions
{"purpose": "enroll", "pet_id": "dog-001", "display_name": "旺财", "policy": "enroll_full", "client_commit": false}
```

```json
{"protocol": "dog-capture/1", "session_id": "cap_…", "nonce": "…", "expires_at": 1762862400.0,
 "h5_path": "/capture/?session=cap_…&nonce=…", "config_path": "/v1/capture/sessions/cap_…/config?nonce=…"}
```

`purpose` 为 `enroll`（必须带 `pet_id`）或 `identify`；`policy` 省略时按用途取默认值。
会话 15 分钟过期。`client_commit` 默认：服务端没配 API Key 时为 true，配了时为 false（token 只给业务后端）。

**复检结果**（`package` 的响应，也存在 `GET /sessions/{id}` 的 `result` 里）

```json
{"status": "accepted", "missing_views": [], "retake_views": [], "elapsed_ms": 128.4,
 "shots": [{"id": "front-1", "view": "front", "accepted": true, "reasons": [], "size": [1280, 720],
            "server": {"view": "front", "box_iou": 0.998, "faces": 1, "quality": 0.91, "level": "high"}}],
 "recognition": {"available": true, "per_shot": {"front-1": {"stability": 0.92, "consistency": 0.98}},
                 "pairwise_min": 0.98},
 "token_expires_at": 1762861200.0, "capture_token": "…(仅 client_commit 会话)"}
```

`status` 为 `retake` 时，`retake_views` 列出要补拍的角度，H5 直接进入补拍；会话可继续上传，直到次数用完。

**提交**

```json
POST /v1/capture/sessions/{id}/commit
{"capture_token": "…", "top_k": 5}
```

建档：所有通过复检的照片进入 `enroll(pet_id, …)`；比对：取质量最高的正面照进入 `identify(…)`（一次会话只记一条比对历史）。
token 一次有效：重复提交返回 `409 token_used`；token 与会话不符返回 `403 bad_token`。

## 4. 采集包

`multipart/form-data`：一个 `manifest` 字段（JSON）+ 若干 `files`（JPEG 原图，质量 0.95，长边 ≤ 4096，单张 ≤ 8 MB，最多 8 张）。
完整结构见 `dog_capture/protocol/capture-package.schema.json`，要点：

```json
{"protocol": "dog-capture/1", "session_id": "cap_…", "nonce": "…",
 "client": {"sdk": "…", …}, "models_used": {…}, "timing": {…},
 "shots": [{"id": "front-1", "view": "front", "rank": 1, "file": "front-1.jpg", "sha256": "…",
            "width": 1280, "height": 720, "time_s": 0.93, "frame_index": 28, "track_id": 1,
            "detection": {"box": [x1, y1, x2, y2], "keypoints": […], …},
            "pose": {"yaw": 0.02, "pitch": 1.1, "roll_deg": 3.4, "iod": 96.0},
            "quality": {"score": 0.91, "level": "high", "factors": {…}}}],
 "signals": {…}}
```

结构性检查（协议版本、会话与 nonce、每个角度的张数上限、文件齐全）在复检之前完成，不合格直接 4xx。

## 5. 服务端复检

每张照片独立重新跑一遍静态图模式的同一引擎（同一模型、同一阈值），而不是相信客户端的分数：

1. 文件：存在、大小、sha256 与 manifest 一致、能解码、尺寸合规；
2. 检测：能重新检测到狗脸，且与客户端声明的框重合（IoU ≥ 0.3）；YOLOX 物种门控；
3. 角度：重新估计的朝向落在该角度的位置带内；
4. 质量：同一套质量分项和等级门槛；
5. 识别就绪（接了识别模型时）：用线上识别模型对原图和 4% 内缩裁剪各编码一次，余弦相似度为 `stability`；
   多张时每张与其余照片的中位相似度为 `consistency`。默认只报告；用 `--capture-consistency-threshold`
   （或 `build_capture_router(stability_threshold=…)`）开启拦截。
6. 所有必选角度至少有一张通过，才 `accepted`。

拒绝原因码（H5 已有中文文案，见 `packages/h5/src/lib/texts.ts`）：

| 原因码 | 含义 | 原因码 | 含义 |
|---|---|---|---|
| `hash_mismatch` | 图片在上传中被改动 | `view_mismatch` | 角度不符合要求 |
| `missing_file` / `bad_image` | 图片缺失 / 无法解码 | `quality_too_low` 及质量分项码 | 质量不足（模糊、过暗、过曝、太小、遮挡…） |
| `image_size_out_of_range` / `file_too_large` | 尺寸或体积不合规 | `recognition_unstable` | 识别模型对轻微裁剪不稳定 |
| `no_face` / `not_dog` | 没有狗脸 / 不像狗 | `identity_inconsistent` | 与其他照片不像同一只狗 |
| `claimed_face_not_found` | 与采集时的狗脸位置不一致 | `landmarks_unreliable` / `eyes_not_visible` | 五官位置无法确认 / 看不到睁开的眼睛（第二阶段核实） |

## 6. Token

`capture_token` 是 HMAC-SHA256 签名的 JSON（会话、用途、pet_id、每张通过照片的 id 与 sha256、过期时间），
10 分钟有效，只能用一次。密钥取 `DOG_CAPTURE_SECRET`，没有时在图库暂存目录生成 `capture_secret.txt`。
多实例部署请设置同一个 `DOG_CAPTURE_SECRET`，并注意会话目前存在单进程内存里（见第 11 节）。

## 7. 活体信号（本期只预留，不判定）

采集包的 `signals` 字段记录：`track_ids` / `single_track`（全程是否同一条轨迹）、`view_sequence`
（稳定停留过的角度及时间）、`yaw_trace`（10 Hz 朝向曲线）、`guidance_counts`（每种提示出现的帧数）、
`prompt_order_followed`（是否按随机下发的角度顺序完成）。服务端原样保存在会话中，`GET /sessions/{id}` 返回
`signals_recorded`。后续可在此基础上做“照片 / 屏幕翻拍”判断，协议无需改版。

## 8. 嵌入方式

**① 跳转 H5（最简单，推荐）**：业务后端建会话后，把用户带到 `h5_path`；完成后 H5 通知宿主，业务后端查询结果并 commit。

**② Vue 组件**：

```ts
import { DogCapture } from '@opendog/dog-capture-h5';
// <DogCapture :session-id="id" :nonce="nonce" api-base="https://pet.example.com"
//             @finished="onResult" @cancel="onCancel" @error="onError" />
```

**③ iframe**：`<iframe src="https://pet.example.com/capture/?session=…&nonce=…" allow="camera; cross-origin-isolated">`，
监听 `message`：

```js
window.addEventListener('message', (e) => {
  if (e.origin !== 'https://pet.example.com' || e.data?.source !== 'dog-capture') return;
  // e.data.type: 'result' | 'error' | 'cancel'；result 带 status 与 retake_views
});
```

宿主页本身没有跨源隔离时，iframe 内的 `crossOriginIsolated` 为 false，推理退回单线程 WASM：能用，但更慢。
对速度敏感时用方式 ①。

**④ Android WebView**：开启 JavaScript 与 `mediaPlaybackRequiresUserGesture=false`；在
`WebChromeClient.onPermissionRequest` 中授予 `RESOURCE_VIDEO_CAPTURE`（App 自身需已获得相机权限）；
`addJavascriptInterface(bridge, "DogCaptureBridge")`，`bridge` 提供 `@JavascriptInterface onResult(String json)`。

**⑤ iOS WKWebView**（iOS 14.3+）：`allowsInlineMediaPlayback = true`；
`userContentController.add(handler, name: "dogCapture")`，在 `didReceive message` 里收结果。

宿主收到的消息都是 `{"source": "dog-capture", "type": …, "session_id": …, …}`，**只是通知**，结果以业务后端查询为准。

## 9. 部署要求

- **HTTPS**：浏览器只在安全上下文里开放摄像头（`localhost` 例外）。手机调试需要 HTTPS 地址。
- **跨源隔离**：`/capture` 带 `COOP: same-origin` + `COEP: require-corp`，启用多线程 WASM。
  `mount_capture` 已加好；前面若有 nginx / 网关，需要透传这两个响应头，并把 `/capture/**`、`/v1/capture/**` 转发到 Python 服务
  （Java 网关目前还没有这条转发，需要时加一条路由即可）。
- **体积**：页面 JS ≈ 86 KB（gzip 34 KB）；ORT WASM 运行时 14.8 MB，构建时预压缩为 2.8 MB（br）/ 3.8 MB（gzip），
  `mount_capture` 按 `Accept-Encoding` 直接发送压缩版本（nginx 可用 `brotli_static on; gzip_static on;` 达到同样效果）；
  模型 `dog_face_pose.onnx` 10.8 MB + `yolox_nano.onnx` 3.7 MB，按 sha256 校验并以 `immutable` 缓存，第二次打开不再下载。
  首次打开实测下载约 17 MB（未压缩时约 29 MB）。弱网场景可换卷积 INT8 学生模型（3.2 MB，关键点偏差均值 0.067 眼距，
  数据见验证文档第 8 节）；**它尚未在浏览器里验证过**，上线前需在目标机型上测。
- **算力**：推理放在 Web Worker 里，界面不卡；H5 默认用 WASM。SDK 的 Worker `init` 消息支持 `webgpu: true`
  （加载 `onnxruntime-web/webgpu`），H5 目前没有打开，需在目标机型测过再启用。

## 10. 移植与一致性

- `packages/sdk`（TypeScript，无框架依赖）：引擎、解码器、协议客户端、Worker；`packages/h5`（Vue 3）：界面。
- 其他平台（Kotlin / Swift / C++ / 小程序原生相机）按同一协议实现，用 `dog_capture/protocol/conformance/*.json`
  一致性向量验证：几何、相似变换、视角状态机、One-Euro、跟踪、选图（含定稿规则）、质量分项、引导、默认配置。
  向量由 Python 参考实现生成（`python tools/export_capture_conformance.py`），TS SDK 全部通过。
- JSON Schema：`session-config` / `capture-package` / `verification-result`；引导码表 `guidance-codes.json`。

## 11. 验证（2026-11-11，云端 Linux 4 核 CPU，无 GPU）

| 项目 | 结果 |
|---|---|
| Python 引擎 + 服务测试 | 76 项通过（含预压缩静态文件、定稿规则）；v0.x 旧测试 48 项通过、4 项跳过 |
| TS SDK | 一致性向量 + 引擎测试 20 项通过；真实模型对照（`DOG_CAPTURE_REAL=1`）关键点与 Python 相差 ≤ 2 px |
| 浏览器端到端 | 见下表 |

端到端：Playwright 驱动无头 Chromium，用真实狗照片生成的 1280×720 / 30 fps 视频作虚拟摄像头，
H5 → Worker（onnxruntime-web WASM，真实模型）→ 上传 → 服务端复检（真实模型）→ token → commit。
识别模型用替身（e2e/server.py），所以 `consistency` 数字没有意义，只验证流程。

| 场景 | 模型就绪 | 开始到选好 | 端上帧率 | 复检 | 提交 |
|---|---|---|---|---|---|
| 比对 `identify_quick`（正面视频） | 1.0 s | 1.96 s，正面 2 张 | 10–14 fps | 通过；IoU ≥ 0.998，质量 0.90–0.91（high） | 比对 front-1 |
| 建档（左、右必选，测试策略） | 0.95 s | 5.7 s（视频每侧 4 s） | 11–15 fps | 通过；IoU 0.983 / 0.992，质量 0.84 / 0.86 | 建档 2 张 |

两个场景页面都处于跨源隔离状态，控制台零报错；WASM 以 br 编码传输（2.8 MB）。

复现：

```bash
python src/Pet-ReID-IMAG/frontend/dog-capture/e2e/make_video.py --front <正面照> --side <3/4 侧面照> --side-face-x <侧面照中狗脸中心 x> --out <媒体目录>
```

```bash
E2E_MEDIA=<媒体目录> DOG_CAPTURE_MODELS=<模型目录> PYTHON=python node src/Pet-ReID-IMAG/frontend/dog-capture/e2e/run.mjs <输出目录>
```

输出目录里有每一步的截图和 `summary.json`（提示时间线、复检结果、传输大小）。

## 12. 局限

2026-11-11 本机补充：用户 ONNX 导入、Windows 构建、独立采集服务、真实 Re-ID 联调与现有模型效果缺口，见
[`DOG_CAPTURE_V1_INTEGRATION_CN.md`](DOG_CAPTURE_V1_INTEGRATION_CN.md)。新增
`GET /v1/capture/sessions/{id}/shots/{shot_id}` 向业务后端提供已通过复检的 JPEG 原图，沿用 API Key 鉴权。
`serve_pet_api.py --capture-config` 可设置质量下限，采集会话策略不能降低此下限。

- 会话和 token 状态在单进程内存里：多实例部署要么会话粘滞，要么把 `SessionStore` 换成 Redis 实现（接口很小）。
- 活体只记录信号，不做判定；目前挡不住“拿一张狗的照片对着镜头转”。
- 浏览器端到端用的是由静态照片做成的视频，真实狗狗的连续转头、抖动和光照变化还需要真机试拍。
- iframe 嵌入且宿主未跨源隔离时，推理是单线程的。
- 微信小程序的 `web-view` 能否调用摄像头因版本而异，未验证；建议用小程序原生相机 + 本协议上传。
