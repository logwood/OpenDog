# Pawprint ID 前端与接入说明

实际前端工程位于 `frontend/pet-reid-web`，使用 React、TypeScript，以及由 Vinext
提供 Next.js App Router 兼容层的 Vite 构建链。
界面已经接入健康检查、图片比对、融合诊断、临时图库、图片录入以及带确认的删除操作。

前端统一访问 Java Spring 网关，不直接访问 Python CUDA 服务。开发环境 API 根地址为：

    http://127.0.0.1:8080

本项目开发服务器使用 3000 端口，启动 Java 前设置：

    $env:FRONTEND_ALLOWED_ORIGINS = "http://127.0.0.1:3000,http://localhost:3000"

Java 默认不返回跨域许可；生产环境建议让前端和 Java 经同一个反向代理域名提供服务。

## 建议页面

1. 系统状态：读取 `GET /v1/upstream-health`，显示 CUDA provider、融合模式、模型哈希、
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
- `query.inference.descriptor.branch_available`：`[true,true]` 才是真正鼻子+脸融合。
- `fusion_weights`：用于调试展示；若为 `[1,0]` 且 `detection=null`，说明进入了鼻子
  单路回退，不应标记为“双分支识别”。

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

当前 Java API 没有面向浏览器用户的登录系统，因此只能绑定本机开发，或放在带认证和
HTTPS 的反向代理后面。参考图片属于敏感数据，正式前端需要权限控制、删除确认和操作日志。

进入 `frontend/pet-reid-web` 后复制 `.env.example` 为 `.env.local`，执行 `npm install` 和
`npm run dev`，再访问 `http://localhost:3000`。完整的启动与人工验收步骤见该工程的
`README.md`。

当前准备状态：前端构建、TypeScript 与 lint 检查通过；Java 13/13 测试通过；允许 Origin
的真实预检返回 200，未授权 Origin 返回 403。验证记录位于工作区
`artifacts/runs/legacy/frontend_cors_smoke/result.json`。
