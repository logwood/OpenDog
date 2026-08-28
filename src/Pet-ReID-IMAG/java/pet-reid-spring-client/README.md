# Pet ReID Java API

这是一个可直接运行的 Spring Boot 项目。它向 Java 业务系统提供强类型客户端和 HTTP API，
并把图片录入、识别、查询和删除请求转发给仓库中的 Python ONNX/CUDA 推理服务。

```text
业务系统 / curl
      |
      v
Java Spring Boot :8080
      |
      v
Python FastAPI :8000 -> AnyFace + SAM2 + ONNX Runtime CUDA -> SQLite 图片库
```

当前默认 ONNX 文件是已验收的 `dogfacenet_semantic_v3_v1` 联合融合网络，输出 512
维 embedding。它在全部 800 个开发身份上重训，并用锁定后的 64 个全新身份、128 个
查询做过一次性盲测；这些身份都不是固定线上类别。线上身份仍由图片录入 API 动态建立。

Java 层没有重复实现 AnyFace 检测和 SAM2 分割，因此推理结果与已验证的 Python GPU
流水线一致；部署时需要同时运行两个进程。

## 环境

- JDK 21
- Maven 3.9+
- 已按工作区 `docs/PET_API.md` 配好的 Python ONNX Runtime CUDA 环境

本机已安装的 Maven 可直接这样调用：

```powershell
D:\Maven\apache-maven-3.9.16\bin\mvn.cmd --version
```

## 1. 启动 Python GPU 服务

从工作区的 `src/Pet-ReID-IMAG` 运行：

```powershell
$env:PET_REID_API_KEY = "replace-with-a-long-random-secret"

python tools\serve_pet_api.py `
  --backend onnx `
  --onnx-provider cuda `
  --config-file ..\..\models\selected\dogfacenet_semantic_v3_v1\config.yaml `
  --identity-weights ..\..\models\selected\dogfacenet_semantic_v3_v1\model_final.pth `
  --onnx-model ..\..\models\selected\dogfacenet_semantic_v3_v1\onnx\pet_embedding.onnx `
  --storage-dir ..\..\data\gallery_store\pet_api_gallery_semantic_v3_v1 `
  --seed-gallery-model ..\..\models\selected\local_pet_gallery_semantic_v3_onnx_v1\gallery_model.json
```

先确认返回的 provider 是 CUDA：

```powershell
curl.exe http://127.0.0.1:8000/health
```

## 2. 启动 Java 服务

打开另一个 PowerShell，进入本目录：

```powershell
$env:PET_REID_BASE_URL = "http://127.0.0.1:8000"
$env:PET_REID_API_KEY = "replace-with-a-long-random-secret"

D:\Maven\apache-maven-3.9.16\bin\mvn.cmd spring-boot:run
```

默认监听 `127.0.0.1:8080`。以下环境变量可以覆盖默认配置：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `SERVER_ADDRESS` | `127.0.0.1` | Java 服务监听地址 |
| `SERVER_PORT` | `8080` | Java 服务端口 |
| `PET_REID_BASE_URL` | `http://127.0.0.1:8000` | Python 服务地址 |
| `PET_REID_API_KEY` | 空 | Java 调用 Python 时发送的 `X-API-Key` |
| `PET_REID_CONNECT_TIMEOUT` | `5s` | 连接超时 |
| `PET_REID_READ_TIMEOUT` | `60s` | GPU 请求读取超时 |
| `FRONTEND_ALLOWED_ORIGINS` | 空 | 允许访问 `/v1/**` 的浏览器 Origin，逗号分隔 |

项目固定使用标准 JDK HTTP/1.1 客户端，以兼容 Uvicorn 的 multipart 上传；不要删除
`spring.http.clients.imperative.factory: simple`，否则 JDK HTTP/2 的 h2c upgrade 可能导致上传失败。

Java 服务绑定到非本机地址前，应在反向代理或 Spring Security 中保护 Java 入口；
`PET_REID_API_KEY` 只用于 Java 到 Python 的连接，并不会自动鉴权 Java 的公开路由。

前端开发服务器与 Java 不同源时，显式开放它的 Origin。本仓库前端默认使用 3000 端口：

```powershell
$env:FRONTEND_ALLOWED_ORIGINS = "http://127.0.0.1:3000,http://localhost:3000"
```

默认值为空，即只允许同源访问。配置只接受不带路径、查询参数或账号信息的
`http://` 或 `https://` Origin，并且不会开放 Cookie 凭据。

## 3. 测试图片录入 API

建议同一只宠物录入至少 2 张清晰、单宠物图片：

```powershell
curl.exe -X POST "http://127.0.0.1:8080/v1/pets/dog-001/images" `
  -F "display_name=豆豆" `
  -F "files=@D:\pet-images\front.jpg" `
  -F "files=@D:\pet-images\left.jpg"
```

成功时 Java API 返回 HTTP `201`，并给出 `added_image_ids`。Java 调用方不需要再发送
Python 的 API key，Java 服务会自动添加它。

## 4. 测试识别 API

```powershell
curl.exe -X POST "http://127.0.0.1:8080/v1/identify?top_k=3" `
  -F "file=@D:\queries\query.jpg"
```

带未知宠物拒识参数的调用示例：

```powershell
curl.exe -X POST "http://127.0.0.1:8080/v1/identify?top_k=3&match_threshold=0.35&minimum_margin=0.08" `
  -F "file=@D:\queries\query.jpg"
```

`0.35` 和 `0.08` 只是接口示例，正式使用前应使用实际已知/未知宠物图片校准。

仓库内已有一张可做回归测试的图片。启动带 seed gallery 的 Python 服务后，下面的请求
预期返回 `predicted_pet_id: local-1`：

```powershell
curl.exe -X POST "http://127.0.0.1:8080/v1/identify?top_k=2" `
  -F "file=@..\..\..\..\data\processed\pet-reid-imag\local_pet_gallery_v1\images\validation\local-1\003_mmexport1787622883567.jpg"
```

## API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/v1/upstream-health` | 查看 Python backend、CUDA provider 和图库数量 |
| `GET` | `/actuator/health` | Java 与上游综合健康检查 |
| `POST` | `/v1/pets/{petId}/images` | 录入 1–8 张参考图 |
| `POST` | `/v1/identify` | 上传查询图并识别 |
| `GET` | `/v1/pets` | 宠物列表 |
| `GET` | `/v1/pets/{petId}` | 宠物与参考图详情 |
| `GET` | `/v1/pets/{petId}/images/{imageId}` | 下载参考图 |
| `DELETE` | `/v1/pets/{petId}/images/{imageId}` | 删除一张参考图 |
| `DELETE` | `/v1/pets/{petId}` | 删除宠物及所有参考图 |

上游的 `4xx` 状态和结构化错误码会原样映射；Python 服务不可达时返回 HTTP `502` 和
`upstream_unavailable`。

## 在 Java 代码中直接调用

项目内提供了 `PetReidClient` 接口，可以在 Spring Bean 中直接注入：

```java
@Service
public class PetLookupService {
    private final PetReidClient petReidClient;

    public PetLookupService(PetReidClient petReidClient) {
        this.petReidClient = petReidClient;
    }

    public PetListResponse allPets() {
        return petReidClient.listPets();
    }
}
```

图片录入和识别方法接收 Spring `MultipartFile`；非 Web 业务可以使用
`MockMultipartFile`，或者自行实现 `MultipartFile` 适配文件/字节数组。

## 自动化测试与打包

```powershell
# 13 个测试：HTTP 协议、multipart、查询参数、错误映射、CORS、控制器与完整上下文
D:\Maven\apache-maven-3.9.16\bin\mvn.cmd test

# 生成可执行 JAR
D:\Maven\apache-maven-3.9.16\bin\mvn.cmd clean package

# 运行打包结果
java -jar target\pet-reid-java-api-1.0.0.jar
```

单元测试不需要 GPU。最后的 curl 回归测试会实际经过 Java、Python 和 CUDA，适合验证
完整部署链路。

本机已完成一次真实链路验证：13/13 Java 测试通过；Java multipart API 录入 2 只宠物、
4 张参考图后，查询正确返回 pet-a（Top-1 0.8814、margin 0.8675），并透传
branch_available=[true,true] 与融合权重 [0.1177,0.8823]。后端健康信息确认
CUDAExecutionProvider、512 维和 semantic_residual_v3。审计结果位于
artifacts/evaluations/java_gateway_semantic_v3_e2e/e2e_result.json。
