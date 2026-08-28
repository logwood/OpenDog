# Pawprint ID Web

这是 Semantic Fusion V3 的本地浏览器工作台。浏览器只访问 Java Spring 网关，不直接
访问 Python CUDA 服务。

## 启动

最简单的方式是在仓库根目录双击启动器：`start-pet-reid.cmd` 使用 CUDA ONNX，
`start-pet-reid-cpu.cmd` 使用纯 CPU ONNX。二者都会在健康检查通过后打开 3000 端口的
前端；`stop-pet-reid.cmd` 用于停止服务。

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

打开 `http://localhost:3000`。右上角变为“已连接”，并显示 ONNX provider、模型指纹和
图库数量，即表示浏览器、Java 与 Python 推理服务已全部连通。

## 人工验收

1. 在“图片比对”选择一张宠物图，点击“开始识别”。结果应显示 Top-1、margin、候选
   身份和融合路径；`鼻子 + 脸` 表示完整双分支，`单路回退` 会显示醒目提示。
2. 可使用仓库中的回归图：
   `data/local_pet_gallery_v1/images/validation/local-1/003_mmexport1787622883567.jpg`。
   使用随附 seed gallery 时，预期身份为 `local-1`。
3. 点击“录入”，填写只含字母、数字、点、下划线或短横线的身份 ID，选择 1–8
   张图片。成功后图库数量会自动刷新。
4. “比对历史”会保存查询图、模型指纹、图库快照、Top-K、耗时与人工复核结果；标记为
   “错误”或“不确定”的记录会自动进入管理员难例。
5. 点击宠物行右侧箭头可修改显示名称、补充参考图、查看质量提示或删除图片和身份。
6. 管理员密钥位于 `logs/quick_start/admin-key.txt`，仅保存在当前浏览器会话。管理员可
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

API 地址通过 `NEXT_PUBLIC_PET_REID_API_BASE_URL` 配置。批量、难例和图库备份接口由
Java 校验 `X-Admin-Key`，但普通本地工作区没有用户登录；在加入完整身份认证和 HTTPS
反向代理之前不要公开部署。
