# UnifiedPetReID V3 raw-RGB E2E ONNX

This is the deployment artifact for the graph-internal preprocessing path.
The graph has exactly one input, `rgb`, with `float32` RGB pixels in `0..255`
and dynamic shape `[N, 3, H, W]`. It performs centered black letterboxing,
ImageNet normalization, learned geometry/crops, identity fusion, and L2
normalization inside ONNX, then returns exactly one `float32` output,
`embedding`, shaped `[N, 512]`.

JPEG/PNG decoding, EXIF orientation, and the HTTP boundary's BGR-to-RGB
conversion are transport operations; no detector, segmenter, second model, or
Python output normalization is required at inference time.

See `metadata.json` for the machine-readable runtime contract and
`validation.json` for the CPU/CUDA ORT parity and development semantic guard.
The original fixed-1280 artifact in the parent `onnx/` directory is retained
as a rollback/compatibility artifact and is not the raw-input contract.
