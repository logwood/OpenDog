# UnifiedPetReID V4 High Resolution — 当前研发主线（包修订 v1）

这是当前研发主线的已验证高分辨率候选包。这里的“包修订 v1”是文件包编号，不是
模型代际。它通过一次性 blind 非劣验证，但尚未执行生产激活；运行时仍然只加载一张
ONNX 图：

```text
RGB [N, 3, H, W] -> UnifiedPetReID V4 -> L2-normalized 512D
```

- 动态原图输入，长边上限 4096；更大的图像仅在进入 ONNX 前按长边等比缩小。
- 1280 全局视图、脸部细节采样、鼻部细节采样和融合计算全部位于同一张 ONNX 图内。
- 不加载 AnyFace、SAM2、身体检测器或第二个身份模型。
- development：V4 与生产 V3 都是 Top-1/Top-5 `16/16`。
- 唯一一次 blind：V4 与生产 V3 都是 Top-1 `15/16`、Top-5 `16/16`，非劣验证通过。
- blind 之后禁止继续用该 split 调参。

## 使用

CUDA 快速启动：仓库根目录 `start-pet-reid-highres.cmd`。

CPU 快速启动：仓库根目录 `start-pet-reid-highres-cpu.cmd`。

也可以直接启动 Python API：

```powershell
python tools/serve_pet_api.py --backend onnx-highres --onnx-provider cuda
```

V4 使用独立图库 `data/gallery_store/pet_api_gallery_unified_v4_v1`。V3 图库不能直接混用；
需要重新录入，或使用 `tools/migrate_unified_highres_gallery.py` 从原始录入图片安全重算。

V3 仍是默认部署和回滚模型，V4 不会覆盖它。运行时唯一必需的模型文件是
`onnx/unified_pet_reid_v4.onnx`；`model_final.pth` 仅保留用于来源追踪和复现导出。
