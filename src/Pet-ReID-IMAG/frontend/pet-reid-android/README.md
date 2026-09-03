# Pawprint ID Android（原生 APK，可侧载）

这个工程产出的是一个真正的 Android APK，不是浏览器页面的 WebView 包装。主界面由
Java/XML 原生 View 构成，包含：

- 原生状态、识别、图库三个页面；
- Android 相册（ACTION_OPEN_DOCUMENT）和相机（ACTION_IMAGE_CAPTURE）；
- 原生 HttpURLConnection multipart 上传；
- 识别结果、候选身份、相似度分数和图库状态展示；
- Android 原生分享面板，可分享当前识别结果的文字摘要；
- 本机最近 10 条识别摘要（每条最多 3 个候选，不保存原图或文件名）；
- 宠物显示名称修改、单张参考图删除和整个身份删除，破坏性操作均二次确认；
- 页面、结果卡和列表项采用轻量原生动效，并遵循系统动画开关；
- 电脑端地址保存、连接检查和断线提示；
- 一次选择 1–8 张参考图录入临时图库。

识别模型仍运行在电脑端的 Python/ONNX 服务中，APK 不内置模型。手机和电脑需要在同一个
可信 Wi-Fi；这只影响模型部署位置，不影响 APK 是原生应用。

## 使用

1. 在电脑上运行仓库根目录的 start-pet-reid-mobile.cmd（没有 CUDA 时使用
   start-pet-reid-mobile-cpu.cmd）。
2. 让手机和电脑连接同一个可信 Wi-Fi。
3. 安装 artifacts/releases/pawprint-id-android/pawprint-id-debug.apk。
4. 首次打开，在“连接”中填写启动窗口显示的地址，例如
   http://192.168.1.20:3000。只填到端口，不要附加 /v1。
5. 在“识别”页拍照/选择图片并点击“开始识别”；在“图库”页填写身份 ID 后录入参考图。

地址保存在 APK 的本地 SharedPreferences 中，顶部“连接”按钮可以修改。电脑服务停止
时原生状态页会显示断线原因；恢复后点击“重新连接”。

## API 边界

APK 直接调用电脑地址的 JSON API，不加载任何网页资源：

- GET /v1/upstream-health（如果连接的是直接 Python 服务，也兼容 GET /health）
- GET /v1/pets
- GET /v1/pets/{pet_id}
- PATCH /v1/pets/{pet_id}
- DELETE /v1/pets/{pet_id}
- DELETE /v1/pets/{pet_id}/images/{image_id}
- POST /v1/identify?top_k=5，multipart 字段 file
- POST /v1/pets/{pet_id}/images，multipart 字段 display_name 和同名 files

图片只上传到用户配置的电脑端地址。APK 不保存图库数据库；最近识别仅在 Android
SharedPreferences 中保存少量文字和数值摘要，可从识别页一键清空，不包含原图、
图片 URI 或文件名，也不使用网页或浏览器存储。

## 构建

构建脚本将 Android SDK、Gradle 缓存和分发包隔离在工作区 .tmp/，不会改动系统
Android Studio。工具链准备好后运行：

    .\build-apk.ps1

也可以直接运行：

    .\gradlew.bat :app:assembleDebug
    .\gradlew.bat :app:lintDebug

输出 APK：

    app/build/outputs/apk/debug/app-debug.apk

构建脚本会复制到：

    artifacts/releases/pawprint-id-android/pawprint-id-debug.apk

同目录的 pawprint-id-debug.apk.sha256 是 SHA-256 校验文件。

这是调试签名 APK，适合 ADB 或手动侧载开发测试；暂不处理应用商店发布。公开部署前
仍应使用自己的 release keystore、HTTPS、认证和权限控制。

## 安装与校验

有 USB 调试设备时：

    adb devices
    adb install -r artifacts/releases/pawprint-id-android/pawprint-id-debug.apk

校验签名和对齐：

    & "$env:ANDROID_HOME\build-tools\36.0.0\apksigner.bat" verify --verbose --print-certs artifacts/releases/pawprint-id-android/pawprint-id-debug.apk
    & "$env:ANDROID_HOME\build-tools\36.0.0\zipalign.exe" -c -P 16 -v 4 artifacts/releases/pawprint-id-android/pawprint-id-debug.apk
    Get-FileHash artifacts/releases/pawprint-id-android/pawprint-id-debug.apk -Algorithm SHA256

## 安全边界

- HTTP 只允许本机/私有局域网主机（或 .local）；公网地址必须使用 HTTPS。
- HTTPS 证书校验使用 Android 默认安全策略，不提供绕过证书错误的按钮。
- 相机照片先写入 APK 私有 cache，上传后由系统按 cache 生命周期清理。
- 当前服务没有普通用户登录；手机模式只应在可信专用局域网短时使用。
