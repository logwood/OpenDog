"""ONNX Runtime sessions with explicit, reported execution-provider fallback.

One model file runs on every target through ONNX Runtime execution providers:
TensorRT / CUDA (NVIDIA desktop and Jetson), DirectML (Windows GPUs), CoreML
(macOS / iOS), OpenVINO (Intel), NNAPI / XNNPACK / QNN (Android and ARM
boards) and CPU everywhere.  TensorRT engines are built by the TensorRT EP on
the machine that runs them and cached, so no GPU-specific engine files are
shipped.  The providers actually used are always reported, never assumed.
"""
from __future__ import annotations

import os
from pathlib import Path

ALIASES = {
    'cpu': 'CPUExecutionProvider', 'cuda': 'CUDAExecutionProvider',
    'tensorrt': 'TensorrtExecutionProvider', 'trt': 'TensorrtExecutionProvider',
    'directml': 'DmlExecutionProvider', 'dml': 'DmlExecutionProvider',
    'coreml': 'CoreMLExecutionProvider', 'openvino': 'OpenVINOExecutionProvider',
    'nnapi': 'NnapiExecutionProvider', 'xnnpack': 'XnnpackExecutionProvider',
    'qnn': 'QNNExecutionProvider', 'rocm': 'ROCMExecutionProvider',
}
AUTO_ORDER = ('TensorrtExecutionProvider', 'CUDAExecutionProvider', 'DmlExecutionProvider',
              'CoreMLExecutionProvider', 'OpenVINOExecutionProvider', 'CPUExecutionProvider')


def _ort():
    try:
        import onnxruntime
    except ImportError as exc:
        raise ImportError('ONNX models need onnxruntime: `pip install onnxruntime` '
                          '(or onnxruntime-gpu / onnxruntime-directml / onnxruntime-openvino)') from exc
    return onnxruntime


def available_providers():
    try:
        return list(_ort().get_available_providers())
    except ImportError:
        return []


def resolve_providers(spec='auto', cache_dir=''):
    """Turn 'auto' / 'cuda,cpu' / ['tensorrt', 'cpu'] into ORT provider entries."""
    available = available_providers()
    if isinstance(spec, str):
        names = [s.strip() for s in spec.split(',') if s.strip()] or ['auto']
    else:
        names = list(spec)
    if names == ['auto']:
        chosen = [p for p in AUTO_ORDER if p in available]
    else:
        chosen = []
        for name in names:
            provider = ALIASES.get(name.lower(), name)
            if provider not in available:
                raise RuntimeError(f'Execution provider {name!r} is not available in this onnxruntime build '
                                   f'(available: {available}); install the matching onnxruntime package')
            chosen.append(provider)
    if 'CPUExecutionProvider' not in chosen:
        chosen.append('CPUExecutionProvider')
    cache = Path(cache_dir or os.environ.get('DOG_CAPTURE_CACHE', Path.home() / '.cache' / 'dog_capture'))
    entries = []
    for provider in chosen:
        options = {}
        if provider == 'TensorrtExecutionProvider':
            cache.mkdir(parents=True, exist_ok=True)
            options = {'trt_fp16_enable': True, 'trt_engine_cache_enable': True,
                       'trt_engine_cache_path': str(cache / 'tensorrt'), 'trt_timing_cache_enable': True}
        elif provider == 'CUDAExecutionProvider':
            # TF32 changed a YOLOX score across the species-gate threshold in
            # the v1 video regression. Preserve FP32 inference by default;
            # reduced precision needs its own accuracy/latency acceptance.
            options = {'use_tf32': False}
        elif provider == 'CoreMLExecutionProvider':
            cache.mkdir(parents=True, exist_ok=True)
            options = {'ModelCacheDirectory': str(cache / 'coreml')}
        entries.append((provider, options))
    return entries


def create_session(path, providers='auto', threads=0, cache_dir=''):
    """Returns (session, info) where info records requested and used providers."""
    ort = _ort()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Model not found: {path}')
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        options.intra_op_num_threads = int(threads)
    options.log_severity_level = 3
    entries = resolve_providers(providers, cache_dir)
    errors = []
    session = None
    # Try the full list, then progressively drop the first (fastest) provider.
    for start in range(len(entries)):
        attempt = entries[start:]
        try:
            session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=[(name, opts) for name, opts in attempt])
            break
        except Exception as exc:  # provider libraries missing, unsupported op, ...
            errors.append(f'{attempt[0][0]}: {exc}'.splitlines()[0][:300])
    if session is None:
        raise RuntimeError(f'Could not create an ONNX Runtime session for {path}: {errors}')
    info = {'model': str(path), 'requested': [name for name, _ in entries],
            'used': session.get_providers(), 'provider_options': session.get_provider_options(),
            'fallback_errors': errors}
    return session, info
