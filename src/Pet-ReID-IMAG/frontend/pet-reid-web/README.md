# Pawprint ID Web

这是 UnifiedPetReID 的本地浏览器工作台，同时支持 V4 当前研发候选、V3 生产基线
以及显式兼容/研究后端。浏览器只访问 Java Spring 网关，不直接访问 Python ONNX
服务。版本角色规则见工作区 `docs/VERSIONING_CN.md`。

## 启动

最简单的方式是在仓库根目录双击启动器：`start-pet-reid.cmd` 和
`start-pet-reid-cpu.cmd` 启动 V3 生产基线；`start-pet-reid-highres.cmd` 和
`start-pet-reid-highres-cpu.cmd` 启动 V4 当前研发候选。它们都会在健康检查通过后
打开 3000 端口的前端；`stop-pet-reid.cmd` 用于停止服务。

安卓手机与电脑在同一可信 Wi-Fi 时，使用：

```powershell
.\start-pet-reid-mobile.cmd
# 没有可用 CUDA 时：.\start-pet-reid-mobile-cpu.cmd
```

窗口会打印手机可访问的局域网 URL。手机只连接前端 3000 端口，`/v1` 由 Vite 同源代理到
仍绑定 `127.0.0.1:8080` 的 Java 网关。页面支持后置相机、移动底部导航和运行时 API 设置。
PWA 完整安装需要 HTTPS；局域网 HTTP 可正常识别并可创建桌面快捷方式。详见工作区
`docs/ANDROID_PWA_CN.md`。

先按 `java/pet-reid-spring-client/README.md` 启动 Python CUDA 服务。启动 Java 网关前，
把前端地址加入精确的 CORS 白名单：

```powershell
$env:FRONTEND_ALLOWED_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
```

然后在本目录运行：

```powershell
Copy-Item .env.example .env.local
npm install
npm run dev
```

打开 `http://localhost:3000`。右上角变为“已连接”后，模型卡应按实际后端显示
“UnifiedPetReID V4 · 当前研发候选”或“UnifiedPetReID V3 · 生产基线”，并显示
“单一联合模型 · RGB → 512D”、ONNX provider、模型指纹和图库数量。若只显示
“UnifiedPetReID · 单图模型”，说明后端未提供代际字段，但界面不会擅自猜成 V3。

## 人工验收

1. 在“图片比对”选择一张宠物图，点击“开始比对”。统一模型结果应显示 Top-1、margin、
   候选身份和“RGB → 512D / 单一联合模型”，不应出现“单路回退”或鼻脸权重。
2. 每个模型空间的图库首次创建时都是空的，先录入至少两个身份，再用未录入的同身份图
   查询。若显式启动 Semantic V3/BIFOR/Agent experiment，界面才会显示对应的分支或
   专家诊断。
3. 点击“录入”，填写只含字母、数字、点、下划线或短横线的身份 ID，选择 1–8
   张图片。成功后图库数量会自动刷新。
4. “比对历史”会保存查询图、模型指纹、图库快照、Top-K、耗时与人工复核结果；标记为
   “错误”或“不确定”的记录会自动进入管理员难例。
5. 点击宠物行右侧箭头可修改显示名称、补充参考图、查看质量提示或删除图片和身份。
6. 管理员密钥位于工作区根目录的
   `artifacts/workspace_logs/quick_start/admin-key.txt`，仅保存在当前浏览器会话。管理员可
   创建后台批量任务、查看难例、导出 CSV，以及下载或合并恢复图库备份。
7. 批量测试选择普通图片时只统计处理结果和耗时；若选择目录，使用以下一级目录结构可
   同时计算 Top-1 准确率（图片的直接父目录名即身份 ID）：

   ```text
   test-set/
     pet-001/query-1.jpg
     pet-001/query-2.jpg
     pet-002/query-1.jpg
   ```

8. 停止 Java 或 Python 服务后刷新，页面应显示断线说明；恢复服务并点击“重试”后状态
   应恢复。

## 自动检查

```powershell
npm run build
npx tsc --noEmit
npm run lint
```

API 地址通过 `NEXT_PUBLIC_PET_REID_API_BASE_URL` 配置；推荐值 `/` 表示同源代理，代理目标
通过 `PET_REID_GATEWAY_PROXY_TARGET` 配置。也可在页面“手机连接”中为当前浏览器覆盖地址。
批量、难例和图库备份接口由
Java 校验 `X-Admin-Key`，但普通本地工作区没有用户登录；在加入完整身份认证和 HTTPS
反向代理之前不要公开部署。
