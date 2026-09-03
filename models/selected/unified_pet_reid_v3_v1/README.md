# UnifiedPetReID V3 — 生产基线（包修订 v1）

这是当前生产默认与回滚包。这里的“包修订 v1”是文件包编号，不是模型代际；
当前研发主线是 UnifiedPetReID V4，尚未替换这个生产指针。它把宠物图像直接送入
一个联合训练的 ONNX 图：

```text
raw RGB [N, 3, H, W] -> (letterbox + UnifiedPetReID + L2) ONNX -> 512D
```

运行时只创建这一张 ONNX 图，不再调用 AnyFace、SAM2、身体检测器或其他身份模型。

## 验收结果

- 开发集：候选 Top-1/Top-5 为 `157/211`，固定父模型为 `156/210`。
- 一次性 blind：候选 `158/204`，固定门槛 `69/104`，通过。
- 旧 clean 与鼻冲突回归：均为 Top-1/Top-5 `70/72`。
- 全 512 图 ONNX/PyTorch 最小余弦：CUDA `0.99999684`，CPU `0.99999690`；低于 `0.9999` 的样本为 0。

默认 E2E 模型文件的 SHA-256 为
`2db41b25d770eb285cd313f4e81a1f77c2017e70d827c0b9a1e48cf74edaf8a5`；旧固定方图的
锁定 hash 仍记录为 `2e278f6e4a6ef4086accfbae34c71a74059daa7fa75ba982fec0bcf0be28eef1`。
`candidate_lock_v3.json` 与 `blind_v3.json` 是原始验收文件的逐字节副本；发布记录见
`deployment_record.json`。

## 部署

- 唯一默认运行时模型：`onnx/e2e/unified_pet_reid.onnx`
- E2E 契约与验证：`onnx/e2e/metadata.json`、`onnx/e2e/validation.json`
- 固定方图回滚 artifact：`onnx/unified_pet_reid.onnx`
- 训练来源归档：`model_final.pth`（仅用于复现导出，运行时不读取也不要求存在）
- 默认图库：`data/gallery_store/pet_api_gallery_unified_v3_v1`
- 默认 CUDA 启动：仓库根目录 `start-pet-reid.cmd`
- 默认 CPU 启动：仓库根目录 `start-pet-reid-cpu.cmd`

该 512D 空间与 Semantic V3、BIFOR、Agent 及更早图库都不兼容。服务会用模型指纹拒绝混用；
需要在独立目录重新录入图片。旧模型仍保留为显式兼容启动项。
