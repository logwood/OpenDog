"""ONNX Runtime sessions with explicit, reported execution-provider fallback.

One model file runs on every target through ONNX Runtime execution providers:
TensorRT / CUDA (NVIDIA desktop and Jetson), DirectML (Windows GPUs), CoreML
(macOS / iOS), OpenVINO (Intel), NNAPI / XNNPACK / QNN (Android and ARM
boards) and CPU everywhere.  TensorRT engines are built by the TensorRT EP on
the machine that runs them and cached, so no GPU-specific engine files are
shipped.  The providers actually used are always reported, never assumed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re

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


def _session_options(ort, threads, thread_spinning, spin_duration_us=-1, spin_backoff_max=1):
    if spin_duration_us < -1 or spin_backoff_max < 1 or spin_backoff_max & (spin_backoff_max - 1):
        raise ValueError('spin_duration_us must be >= -1 and spin_backoff_max a positive power of two')
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        options.intra_op_num_threads = int(threads)
    if not thread_spinning:
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    elif spin_duration_us >= 0 or spin_backoff_max != 1:
        # Older ORT accepts unknown config keys silently. Do not report a
        # bounded policy on a build on which it has not been validated.
        version = tuple(int(v) for v in re.findall(r'\d+', ort.__version__)[:2])
        if version < (1, 26):
            raise RuntimeError('Bounded thread spinning requires onnxruntime >= 1.26; '
                               'upgrade ORT or use spin_duration_us=-1, spin_backoff_max=1')
        for pool in ('intra_op', 'inter_op'):
            if spin_duration_us >= 0:
                options.add_session_config_entry(f'session.{pool}.spin_duration_us', str(spin_duration_us))
            options.add_session_config_entry(f'session.{pool}.spin_backoff_max', str(spin_backoff_max))
    options.log_severity_level = 3
    return options


def _session_info(path, entries, session, errors, thread_spinning, threads=0,
                  spin_duration_us=-1, spin_backoff_max=1):
    return {'model': str(path), 'requested': [name for name, _ in entries],
            'used': session.get_providers(), 'provider_options': session.get_provider_options(),
            'fallback_errors': errors, 'thread_spinning': thread_spinning,
            'intra_op_threads': threads,
            'spin_duration_us': spin_duration_us if thread_spinning else 0,
            'spin_backoff_max': spin_backoff_max if thread_spinning else 1}


def create_session(path, providers='auto', threads=0, cache_dir='', *, thread_spinning=True,
                   spin_duration_us=-1, spin_backoff_max=1):
    """Returns (session, info) where info records requested and used providers."""
    ort = _ort()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Model not found: {path}')
    options = _session_options(ort, threads, thread_spinning, spin_duration_us, spin_backoff_max)
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
    return session, _session_info(path, entries, session, errors, thread_spinning, threads,
                                  spin_duration_us, spin_backoff_max)


class _CudaGraph:
    def __init__(self, ort, session, device_id):
        import numpy as np

        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or not outputs or any(
            value.type != 'tensor(float)' or value.shape is None
            or not all(isinstance(dim, int) and dim > 0 for dim in value.shape)
            for value in inputs + outputs
        ):
            raise ValueError('CUDA Graph requires one static FP32 input and static FP32 outputs')
        self.shape = tuple(inputs[0].shape)
        self._session = session
        self._binding = session.io_binding()
        self._input = ort.OrtValue.ortvalue_from_numpy(np.zeros(self.shape, np.float32), 'cuda', device_id)
        self._binding.bind_ortvalue_input(inputs[0].name, self._input)
        self._outputs = []
        for output in outputs:
            value = ort.OrtValue.ortvalue_from_shape_and_type(output.shape, np.float32, 'cuda', device_id)
            self._outputs.append(value)
            self._binding.bind_ortvalue_output(output.name, value)
        # ORT stores captured CUDA graphs per thread. Warm and replay on one worker,
        # even when callers change from startup to video or request threads.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dog-capture-graph')
        try:
            self._worker.submit(session.run_with_iobinding, self._binding).result()
        except BaseException:
            self.close()
            raise

    def _replay(self, tensor):
        import numpy as np

        self._input.update_inplace(np.ascontiguousarray(tensor))
        self._session.run_with_iobinding(self._binding)
        return self._binding.copy_outputs_to_cpu()

    def run(self, tensor):
        return self._worker.submit(self._replay, tensor).result()

    def close(self):
        self._worker.shutdown()


def create_cuda_graph_session(path, providers='auto', threads=0, cache_dir='', *, thread_spinning=True,
                              spin_duration_us=-1, spin_backoff_max=1):
    """Open and warm a CUDA Graph session, raising if ordinary inference is needed."""
    ort = _ort()
    entries = resolve_providers(providers, cache_dir)
    name, cuda_options = entries[0]
    if name != 'CUDAExecutionProvider':
        raise RuntimeError('CUDA Graph requires CUDA as the first provider')
    options = _session_options(ort, threads, thread_spinning, spin_duration_us, spin_backoff_max)
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=[(name, {**cuda_options, 'enable_cuda_graph': True}),
                                              'CPUExecutionProvider'])
    info = _session_info(path, entries, session, [], thread_spinning, threads, spin_duration_us, spin_backoff_max)
    actual = info['provider_options'].get(name, {})
    if info['used'][0] != name or actual.get('enable_cuda_graph') != '1' or actual.get('use_tf32') != '0':
        raise RuntimeError('ONNX Runtime did not activate FP32 CUDA Graph')
    session.disable_fallback()
    graph = _CudaGraph(ort, session, int(actual.get('device_id', '0')))
    return session, info, graph
