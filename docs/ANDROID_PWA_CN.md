# Pawprint ID 安卓与移动端说明

当前移动端采用同一套 React 界面的 PWA 方案。它不是把模型塞进手机：模型仍在电脑的
Python ONNX 服务中运行，安卓负责拍照、上传、展示候选结果和管理图库。这样 Web 与安卓
共享功能和交互，模型更新也不需要重新发布 APK。

```text
安卓 Chrome / PWA
        │  http(s)://电脑地址:3000/v1/*
        ▼
电脑上的 Vinext/Vite 前端（同源代理）
        │  http://127.0.0.1:8080
        ▼
Java Spring 网关 ──► http://127.0.0.1:8000 ──► Python ONNX
```

手机只需访问 3000 端口。默认手机模式不会把 Java 8080 或 Python 8000 暴露到局域网。

## 局域网快速使用

1. 让安卓手机和运行模型的电脑连接同一个可信 Wi-Fi。
2. 在仓库根目录双击 `start-pet-reid-mobile.cmd`；没有可用 CUDA 时使用
   `start-pet-reid-mobile-cpu.cmd`。
3. 启动窗口会输出形如 `http://192.168.1.20:3000` 的 `Ready` 地址，在手机 Chrome
   中打开它。
4. Windows 防火墙首次询问 Node.js/Vite 时，只允许“专用网络”。
5. 页面右上角“手机连接”中保持 API 根地址为空；这表示使用当前网页的同源 `/v1`
   代理。随后可以直接点“拍照”进行识别。

自动选错网卡时，显式指定手机能够访问的 IPv4：

```powershell
.\scripts\pet-reid-stack.ps1 start -Provider cuda -Model production `
  -Lan -LanAddress 192.168.1.20
```

用 `ipconfig` 查看地址；不要填 `127.0.0.1`。停止方式仍是 `stop-pet-reid.cmd`。

## 安装到安卓桌面

PWA 已包含 manifest、192/512 图标、maskable 图标、Service Worker、独立窗口显示和离线
说明页。浏览器会在满足安装条件时显示“安装应用”。也可以在 Chrome 菜单选择“安装应用”
或“添加到主屏幕”。

标准 PWA 的完整安装要求页面来自 HTTPS（`localhost` 是开发例外）。直接打开局域网 HTTP
地址可以使用全部在线识别功能，也通常可以创建桌面快捷方式，但浏览器可能不会授予完整
PWA 安装资格。正式使用时应把 3000 端口放到带可信证书、登录和访问控制的 HTTPS 反向
代理后；不要通过关闭浏览器安全策略来规避这个要求。

Service Worker 只缓存界面静态资源，明确排除 `/v1/**`。断网时能看到离线说明，但推理、
图库和历史不会伪造缓存结果，仍要求电脑服务在线。

## API 地址设置

推荐使用空地址，即同源代理。仅在前端和 Java 分开部署时，才在“手机连接”中填写完整
API 根地址，例如 `https://pet-api.example.com`。不要附加 `/v1`。设置保存在当前浏览器的
本地存储中；“恢复构建默认”可以清除覆盖值。

HTTPS 前端不能直连 HTTP API，这是浏览器的 mixed-content 安全限制。此时应继续使用同源
反向代理，或同时给 API 配置 HTTPS。

## 安全边界

局域网模式会让 3000 端口可被同网段设备访问，而普通识别和图库接口目前没有用户登录。
参考图片属于敏感数据，因此只应在可信专用网络短时使用。管理员批量、难例和备份接口仍需
`X-Admin-Key`，但这不能替代普通用户认证。公网或组织内正式部署前至少需要 HTTPS、登录、
权限控制、访问日志和请求限流。

## 原生 APK（当前可侧载）

仓库现在附带真正的原生 Android 应用（不是 WebView 壳），工程位于
`src/Pet-ReID-IMAG/frontend/pet-reid-android`。它使用 Java/XML 原生 View，直接通过
`HttpURLConnection` 调用电脑端 API，提供：

- 原生“状态 / 识别 / 图库”三页界面，不加载任何网页；
- Android 系统相册选择与相机拍照，支持一次选择 1–8 张参考图；
- 原生 multipart 上传、识别候选/分数展示和图库状态刷新；
- 首次启动填写电脑 API 地址，后续保存在本机，并提供局域网连接检查；
- TLS 证书错误拒绝、HTTP 局域网地址校验，以及相机临时文件清理。

已构建的调试 APK：

`artifacts/releases/pawprint-id-android/pawprint-id-debug.apk`

构建、校验和 ADB 安装命令见工程内的
`src/Pet-ReID-IMAG/frontend/pet-reid-android/README.md`。这是调试签名包，适合 USB
调试或手动侧载开发测试；暂不涉及应用商店发布签名。手机仍需要能访问电脑上的
API 端口（默认由手机模式打印的地址提供），APK 本身不包含模型，也不支持离线推理。正式公网部署仍需要 HTTPS、登录、
权限控制、访问日志和限流。

从仓库根目录可以直接运行 build-pet-reid-android.cmd 重新构建。
