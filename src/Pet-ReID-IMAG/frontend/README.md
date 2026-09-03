# Pawprint ID 前端与接入说明

实际前端工程位于 `frontend/pet-reid-web`，使用 React、TypeScript，以及由 Vinext
提供 Next.js App Router 兼容层的 Vite 构建链。
界面已经接入健康检查、图片比对、模型诊断、临时图库、图片录入以及带确认的删除操作。
模型卡直接读取健康接口中的 `model_generation` 和 `deployment_role`：V4 显示为
“当前研发候选”，V3 显示为“生产基线”，不会再把所有单图后端统称为 V3。

前端统一访问 Java Spring 网关，不直接访问 Python CUDA 服务。默认通过前端服务器的同源
`/v1` 代理转发到：

    http://127.0.0.1:8080

本项目开发服务器使用 3000 端口，启动 Java 前设置：

    $env:FRONTEND_ALLOWED_ORIGINS = "http://127.0.0.1:3000,http://localhost:3000"

同源代理让安卓访问时只需连接电脑的 3000 端口，Java 8080 和 Python 8000 可以继续只监听
`127.0.0.1`。若浏览器改为直连 Java，Java 默认不返回跨域许可，需要显式配置精确 Origin。
生产环境建议让前端和 Java 经同一个带 HTTPS、认证与权限控制的反向代理域名提供服务。

## 安卓与 PWA

界面支持手机底部导航、后置相机拍照、运行时 API 地址设置、PWA manifest、标准/Maskable
图标、Service Worker 和离线说明页。仓库根目录的 `start-pet-reid-mobile.cmd` 与
`start-pet-reid-mobile-cpu.cmd` 会自动选择局域网 IPv4、监听前端 3000 端口并打印手机访问
地址。完整步骤及“局域网 HTTP 可用、完整 PWA 安装需 HTTPS”的边界见工作区
`docs/ANDROID_PWA_CN.md`。

另外提供一个真正的原生 Android APK（不是 WebView 壳），工程在
`frontend/pet-reid-android`，已构建 APK 在
`artifacts/releases/pawprint-id-android/pawprint-id-debug.apk`。APK 使用 Java/XML 原生
View、Android 相机/相册和 `HttpURLConnection` multipart 客户端，直接调用电脑端 API；
它不加载网页、不把模型打进手机。首次启动时填写 `start-pet-reid-mobile.cmd` 打印的地址。

## 建议页面

1. 系统状态：读取 `GET /v1/upstream-health`，显示 provider、模型架构、模型哈希、
   图库身份数和参考图数。
2. 宠物图库：读取 `GET /v1/pets`，进入 `GET /v1/pets/{pet_id}` 查看详情。
3. 图片录入：向 `POST /v1/pets/{pet_id}/images` 提交 `multipart/form-data`，
   字段为一个可选的 `display_name` 和 1–8 个同名 `files`。
4. 图片比对：向 `POST /v1/identify?top_k=5` 提交单个 `file`。
5. 图库维护：下载或删除单张参考图，也可以删除整个宠物身份。

## 识别结果展示

JSON 字段统一使用 `snake_case`。界面至少区分：

- `accepted`：是否通过当前拒识条件。
- `decision`：`closed_set_top1` 表示只是封闭集最近邻，不代表能够拒绝陌生宠物。
- `top1_score` 和 `margin`：只作为分数展示，阈值校准前不要标成概率。
- `candidates`：候选身份、相似度和参考图数量。
- 统一模型通过
  `query.inference.descriptor.runtime_diagnostics.unified.single_graph=true`
  标识；界面显示“RGB → 512D”，不能把兼容容器中的 `branch_available=[false,true]`
  解释成单路回退，也不展示鼻脸权重。
- 只有显式启动旧 Semantic V3/BIFOR/Agent experiment 时，才根据
  `branch_available`、`fusion_weights` 与 `detection` 展示多分支诊断。

浏览器构造 `FormData` 后不要手动填写 `Content-Type`，否则 multipart boundary 会丢失。

## 错误处理

错误响应固定为：

    {
      "error": {
        "code": "gallery_empty",
        "message": "No enrolled pets",
        "details": {}
      }
    }

前端按 HTTP 状态与 `error.code` 处理，不解析英文 `message`：

- 400：参数或图片无效。
- 404：宠物或图片不存在。
- 409：图库为空、模型指纹冲突或图片归属冲突。
- 413：文件或请求过大。
- 502：Java 无法连接 Python 推理服务。

## 安全边界

当前 Java API 没有面向浏览器用户的登录系统，因此桌面默认只绑定本机；手机模式只应在
可信专用局域网短时使用。参考图片属于敏感数据，正式前端需要 HTTPS、权限控制、删除确认
和操作日志。Service Worker 明确不缓存 `/v1/**` API 或识别结果。

进入 `frontend/pet-reid-web` 后复制 `.env.example` 为 `.env.local`，执行 `npm install` 和
`npm run dev`，再访问 `http://localhost:3000`。完整的启动与人工验收步骤见该工程的
`README.md`。

当前准备状态：前端构建、TypeScript 与 lint 检查通过；Java 13/13 测试通过；允许 Origin
的真实预检返回 200，未授权 Origin 返回 403。验证记录位于工作区
`artifacts/runs/legacy/frontend_cors_smoke/result.json`。
