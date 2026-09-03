# UnifiedPetReID V3 legacy ONNX

`unified_pet_reid.onnx` 是生产基线包中的历史锁定固定方图（输入已由调用方
letterbox 到 1280x1280），保留用于 rollback/compatibility。生产默认部署使用
`e2e/unified_pet_reid.onnx`，其契约如下：

- 输入：`rgb`, float32 `[N, 3, H, W]`, RGB 0..255；
- 输出：`embedding`, float32 `[N, 512]`, 已做 L2 归一化；
- batch/H/W：动态，已验证多种尺寸；
- 外部 tensor 文件：无；
- 运行时外部模型：无；
- E2E ONNX SHA-256：`2db41b25d770eb285cd313f4e81a1f77c2017e70d827c0b9a1e48cf74edaf8a5`。

详见 `e2e/README.md`、`e2e/metadata.json` 和 `e2e/validation.json`。
