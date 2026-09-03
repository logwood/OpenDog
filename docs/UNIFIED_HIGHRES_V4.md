# UnifiedPetReID V4 当前研发主线

V4 是当前模型研发代际，也是一个已通过 development、旧协议回归、CPU/CUDA ONNX
一致性以及唯一一次 blind 非劣比较的已验证候选。生产激活尚未执行，所以 V3 仍是
生产基线、默认部署和回滚点；V4 不覆盖 V3，也不复用 V3 图库。版本角色边界见
[`VERSIONING_CN.md`](VERSIONING_CN.md)。

## 模型契约

```text
float32 RGB [N, 3, H, W]，0..255
        ↓
单张动态 UnifiedPetReID V4 ONNX 图
        ↓
L2-normalized float32 [N, 512]
```

- 动态高度和宽度，最小边 64，最大长边 4096。
- 超过 4096 的图像先按长边等比缩小，不裁成固定的 800×800。
- 1280 全局视图以及脸部、鼻部高分辨率细节采样都在同一 ONNX 图内。
- 不加载外部检测器、分割器或第二个身份模型。
- 小于等于 1280 最大边的输入保持 V3 父模型锚点。

正式模型：

```text
models/selected/unified_pet_reid_v4_v1/onnx/unified_pet_reid_v4.onnx
SHA-256 dbd4448133efec28efb770a6ce77c749b4f8b0913c8f40273420be571fe7b000
```

## 验收结果

- 高分辨率 development：V4 和生产 V3 均为 Top-1/Top-5 `16/16`。
- 旧 V3 development：Top-1 `157/256`，Top-5 `211/256`，V4 低分辨率输出逐位保持 V3。
- 旧 clean 与鼻冲突：均为 Top-1 `70/72`、Top-5 `72/72`。
- 唯一一次 blind：V4 与生产 V3 均为 Top-1 `15/16`、Top-5 `16/16`，通过非劣条件。
- ONNX/PyTorch development 最小余弦：CPU 与 CUDA 均不低于 `0.99999988`。

blind 已经消费，结果不能再用于调参。完整记录位于
`models/selected/unified_pet_reid_v4_v1/deployment_record.json`。

当前状态快照见 [`SESSION_STATE_CN.md`](SESSION_STATE_CN.md)。V4 是当前研发后端，V3 仍是生产默认；两个统一模型图库都按 ONNX SHA-256 隔离。当前尚未创建
`data/gallery_store/pet_api_gallery_unified_v4_v1`，第一次启动 V4 时会自动建立空图库。

## 快速启动

CUDA：

```powershell
.\start-pet-reid-highres.cmd
```

CPU：

```powershell
.\start-pet-reid-highres-cpu.cmd
```

完整前端仍访问 <http://localhost:3000>；Python API 为
<http://127.0.0.1:8000>，OpenAPI 页面为 <http://127.0.0.1:8000/docs>。

只启动 Python API：

```powershell
Set-Location src\Pet-ReID-IMAG
$Python = "D:\CondaData\envs\torch312\python.exe"
& $Python tools\serve_pet_api.py `
  --backend onnx-highres `
  --onnx-provider cuda
```

未指定 `--storage-dir` 时，V4 自动使用独立目录：

```text
data/gallery_store/pet_api_gallery_unified_v4_v1
```

模型 SHA-256 会写入 Gallery。V3/V4 特征空间混用时，服务会以
`model_mismatch` 拒绝启动或导入。

## 录入与比对

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/pets/dog-001/images" `
  -F "display_name=豆豆" `
  -F "files=@D:\pet-images\front.jpg" `
  -F "files=@D:\pet-images\left.jpg"

curl.exe -X POST "http://127.0.0.1:8000/v1/identify?top_k=3" `
  -F "file=@D:\pet-images\query.jpg"
```

若设置了 `PET_REID_API_KEY`，请求中还要加入
`-H "X-API-Key: $env:PET_REID_API_KEY"`。

## 迁移既有 V3 图库

迁移读取 V3 图库保存的原始录入图片，用 V4 全部重算，在完整性检查通过后原子发布
新的 V4 目录。源图库只读，目标目录必须不存在：

```powershell
Set-Location src\Pet-ReID-IMAG
& $Python tools\migrate_unified_highres_gallery.py `
  --onnx-provider cuda
```

当前工作区尚未创建统一 V3/V4 持久图库；`data/gallery_store/` 中已有的 semantic、BIFOR、Agent 和 temporary 图库属于其他特征空间，不能直接迁入。首次使用 V4 时直接录入即可；只有在统一 V3 图库确实包含原始录入图片时，才运行上面的迁移命令。

## 测试

快速回归：

```powershell
$env:PYTHONPATH="$PWD\src\Pet-ReID-IMAG"
& $Python -m pytest -q `
  src\Pet-ReID-IMAG\tests\test_unified_highres.py `
  src\Pet-ReID-IMAG\tests\test_unified_highres_runtime.py `
  src\Pet-ReID-IMAG\tests\test_unified_highres_api_runtime.py `
  src\Pet-ReID-IMAG\tests\test_unified_highres_package.py
```

实际 ONNX 临时图库 smoke 测试：

```powershell
Set-Location src\Pet-ReID-IMAG
& $Python tools\smoke_test_pet_api.py `
  --backend onnx-highres `
  --onnx-provider cuda `
  --enroll dog-a=D:\test\dog-a-1.jpg `
  --enroll dog-a=D:\test\dog-a-2.jpg `
  --enroll dog-b=D:\test\dog-b-1.jpg `
  --enroll dog-b=D:\test\dog-b-2.jpg `
  --query D:\test\dog-a-query.jpg `
  --expected-pet-id dog-a
```

三层 Python → Java → Web 隔离测试：

```powershell
.\scripts\test-live-stack.ps1 -Provider cuda -Model candidate
```

旧值 `-Model unified-v4` 仍可用于复现历史命令，但新命令使用部署角色名。
