"""Optional TensorRT backend with persistent pinned buffers and CUDA Graphs.

Imported only when explicitly requested. It never changes drivers, builds an
engine during a video, or silently falls back to a different neural backend.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import platform
import threading

import numpy as np

_dll_handles = []


def load_cuda_dlls():
    """Find CUDA redistributables without changing system PATH or the driver."""
    if os.name != 'nt' or _dll_handles:
        return
    import site
    candidates = []
    configured = os.environ.get('DOG_CAPTURE_CUDA_DLL_DIR')
    if configured:
        candidates.append(Path(configured))
    for base in site.getsitepackages():
        base = Path(base)
        for pattern in ('nvidia/*/bin', 'nvidia/*/lib', 'tensorrt_libs', 'tensorrt_cu12_libs'):
            candidates.extend(base.glob(pattern))
    for path in candidates:
        if path.is_dir():
            _dll_handles.append(os.add_dll_directory(str(path.resolve())))
    # CUDA Python and TensorRT will resolve against these directories. Keep the
    # handles alive: closing add_dll_directory would undo the process-local path.


def cuda_check(result):
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f'CUDA call failed: {error}')
    return values[0] if len(values) == 1 else tuple(values)


class TrtModel:
    """Fixed batch=1, synchronous NumPy facade over an asynchronous CUDA graph.

    Buffers are reused. Returned output arrays are borrowed until the next call;
    the consuming pipeline extracts/copies its small results immediately.
    Each instance is owned by one inference worker at a time.
    """
    def __init__(self, engine_path, *, use_cuda_graph=True):
        load_cuda_dlls()
        import tensorrt as trt
        from cuda import cuda
        self.cuda = cuda
        self.closed = False
        self.host_pointers, self.device_pointers = [], []
        self.stream = self.graph = self.graph_exec = None
        self.context_handle = None
        self.lock = threading.Lock()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        path = Path(engine_path)
        metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
        self.metadata = metadata
        self.precision = metadata['precision']
        from .models import file_hash
        if file_hash(path) != metadata['engine_sha256']:
            raise ValueError(f'Engine checksum mismatch: {path}')
        onnx_path = path.with_suffix('.onnx')
        if onnx_path.exists() and file_hash(onnx_path) != metadata['onnx_sha256']:
            raise ValueError(f'ONNX graph changed after engine build: {path}; rebuild the engine')
        if metadata['tensorrt_version'] != trt.__version__:
            raise RuntimeError('TensorRT engine version mismatch; rebuild with prepare_dog_trt.py')
        if metadata.get('platform', platform.system()) != platform.system():
            raise RuntimeError('TensorRT engine platform mismatch; rebuild on this OS')
        try:
            cuda_check(cuda.cuInit(0))
            device = cuda_check(cuda.cuDeviceGet(0))
            self.context_handle = cuda_check(cuda.cuDevicePrimaryCtxRetain(device))
            self.device = device
            self._activate()
            major = cuda_check(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device))
            minor = cuda_check(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device))
            if metadata['compute_capability'] != [major, minor]:
                raise RuntimeError('TensorRT engine GPU architecture mismatch; rebuild for this GPU')
            gpu_name = cuda_check(cuda.cuDeviceGetName(256, device)).split(b'\0', 1)[0].decode()
            if metadata.get('gpu_name', gpu_name) != gpu_name:
                raise RuntimeError('TensorRT engine GPU model mismatch; rebuild for this GPU')
            self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
            if self.engine is None:
                raise RuntimeError(f'Cannot deserialize TensorRT engine: {path}')
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError('Cannot create TensorRT execution context')
            self.stream = cuda_check(cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING))
            self.inputs, self.outputs = [], []
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = tuple(self.engine.get_tensor_shape(name))
                if not shape or any(d <= 0 for d in shape):
                    raise ValueError('This runtime expects fixed, nonempty tensor shapes')
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                size = int(np.prod(shape)) * dtype.itemsize
                host = cuda_check(cuda.cuMemHostAlloc(size, 0))
                self.host_pointers.append(host)
                buffer = (ctypes.c_ubyte * size).from_address(int(host))
                array = np.frombuffer(buffer, dtype=dtype).reshape(shape)
                array.fill(0)
                pointer = cuda_check(cuda.cuMemAlloc(size))
                self.device_pointers.append(pointer)
                self.context.set_tensor_address(name, int(pointer))
                item = {'name': name, 'shape': shape, 'dtype': dtype, 'array': array,
                        'device': pointer, 'bytes': size}
                (self.inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.outputs).append(item)
            if len(self.inputs) != 1:
                raise ValueError('Expected one image input')
            # Preserve the declared semantic output order, not an engine's
            # implementation-dependent I/O ordering.
            output_order = metadata['output_names']
            self.outputs.sort(key=lambda x: output_order.index(x['name']))
            self.input = {'shape': self.inputs[0]['shape'], 'dtype': self.inputs[0]['dtype']}
            for _ in range(3):
                self._enqueue()
                cuda_check(cuda.cuStreamSynchronize(self.stream))
            self.cuda_graph_enabled = False
            if use_cuda_graph:
                cuda_check(cuda.cuStreamBeginCapture(self.stream, cuda.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL))
                self._enqueue()
                self.graph = cuda_check(cuda.cuStreamEndCapture(self.stream))
                self.graph_exec = cuda_check(cuda.cuGraphInstantiate(self.graph, 0))
                self.cuda_graph_enabled = True
        except BaseException:
            self.close()
            raise

    def _activate(self):
        # Another CUDA consumer may change the current context on this thread.
        cuda_check(self.cuda.cuCtxSetCurrent(self.context_handle))

    def _enqueue(self):
        for item in self.inputs:
            cuda_check(self.cuda.cuMemcpyHtoDAsync(item['device'], item['array'].ctypes.data, item['bytes'], self.stream))
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError('TensorRT enqueue failed')
        for item in self.outputs:
            cuda_check(self.cuda.cuMemcpyDtoHAsync(item['array'].ctypes.data, item['device'], item['bytes'], self.stream))

    def __call__(self, value):
        if self.closed:
            raise RuntimeError('TensorRT model is closed')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('A TensorRT model instance cannot be used concurrently')
        try:
            self._activate()
            item = self.inputs[0]
            if value.shape != item['shape']:
                raise ValueError(f'Expected {item["shape"]}, received {value.shape}')
            np.copyto(item['array'], value, casting='same_kind')
            if self.graph_exec is not None:
                cuda_check(self.cuda.cuGraphLaunch(self.graph_exec, self.stream))
            else:
                self._enqueue()
            cuda_check(self.cuda.cuStreamSynchronize(self.stream))
            return [item['array'] for item in self.outputs]
        finally:
            self.lock.release()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.context_handle is None:
            return
        self._activate()
        if self.stream is not None:
            self.cuda.cuStreamSynchronize(self.stream)
        if self.graph_exec is not None:
            self.cuda.cuGraphExecDestroy(self.graph_exec)
        if self.graph is not None:
            self.cuda.cuGraphDestroy(self.graph)
        self.context = None
        self.engine = None
        for pointer in self.device_pointers:
            self.cuda.cuMemFree(pointer)
        for pointer in self.host_pointers:
            self.cuda.cuMemFreeHost(pointer)
        if self.stream is not None:
            self.cuda.cuStreamDestroy(self.stream)
        self.cuda.cuDevicePrimaryCtxRelease(self.device)
        self.context_handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def fast_nms(boxes, scores, classes, *, threshold=.35, iou_threshold=.5, limit=25):
    """Class-agnostic fast NMS, then the caller filters the dog class.

    Do not NMS only the dog channel: another class can be the winning class at
    an anchor. No top-K truncation is applied before NMS.
    """
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    scores = np.asarray(scores, np.float32).reshape(-1)
    classes = np.asarray(classes, np.int32).reshape(-1)
    eligible = np.flatnonzero(np.isfinite(scores) & (scores >= threshold) & np.isfinite(boxes).all(axis=1))
    order = eligible[np.argsort(-scores[eligible], kind='stable')]
    selected = []
    while len(order) and len(selected) < limit:
        index = int(order[0])
        selected.append(index)
        others = order[1:]
        if not len(others):
            break
        a, b = boxes[index], boxes[others]
        low = np.maximum(a[:2], b[:, :2])
        high = np.minimum(a[2:], b[:, 2:])
        inter = np.prod(np.maximum(high - low, 0), axis=1)
        area_a = np.prod(np.maximum(a[2:] - a[:2], 0))
        area_b = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0), axis=1)
        iou = inter / np.maximum(area_a + area_b - inter, 1e-12)
        order = others[iou <= iou_threshold]
    result_boxes = np.zeros((1, limit, 4), np.float32)
    result_classes = np.zeros((1, limit), np.float32)
    result_scores = np.zeros((1, limit), np.float32)
    n = len(selected)
    result_boxes[0, :n] = boxes[selected]
    result_classes[0, :n] = classes[selected]
    result_scores[0, :n] = scores[selected]
    return [result_boxes, result_classes, result_scores, np.array([n], np.float32)]


class TrtDetector:
    """TensorRT image network + tiny CPU NMS; no CPU neural inference."""
    def __init__(self, directory, threshold=.35, use_cuda_graph=True):
        directory = Path(directory)
        self.spec = json.loads((directory / 'detector_contract.json').read_text(encoding='utf8'))
        if self.spec['input_domain'] not in ('pixel_0_255', 'tflite_real'):
            raise ValueError('Unsupported TensorRT detector input domain')
        metadata = json.loads((directory / 'detector.json').read_text(encoding='utf8'))
        from .models import file_hash
        if metadata.get('contract_sha256') and file_hash(directory / 'detector_contract.json') != metadata['contract_sha256']:
            raise ValueError('Detector postprocessing contract checksum mismatch')
        self.model = TrtModel(directory / 'detector.engine', use_cuda_graph=use_cuda_graph)
        self.threshold = threshold
        self.input = {'shape': (1, 320, 320, 3), 'dtype': np.uint8}

    def __call__(self, value):
        real = value.astype(np.float32)
        if self.spec['input_domain'] == 'tflite_real':
            real = (real - self.spec['zero_point']) * self.spec['scale']
        boxes, scores, classes = self.model(real)
        return fast_nms(boxes, scores, classes, threshold=self.threshold,
                        iou_threshold=self.spec['iou_threshold'], limit=self.spec['max_detections'])

    def close(self):
        self.model.close()
