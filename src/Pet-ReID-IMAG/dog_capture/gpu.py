"""Optional LiteRT WebGPU backend; same TFLite weights and output contract."""
import ctypes
import os
from pathlib import Path

import numpy as np

_dll_handles = []


def load_windows_dxc():
    if os.name != 'nt' or _dll_handles:
        return
    candidates = [os.environ.get('DOG_CAPTURE_DXC_DIR', ''),
                  str(Path(__file__).resolve().parents[3] / '.runtime/dxc/bin/x64'),
                  str(Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32/Microsoft-Edge-WebView')]
    for directory in candidates:
        if not directory or not (Path(directory) / 'dxcompiler.dll').is_file():
            continue
        try:
            _dll_handles.append(os.add_dll_directory(str(Path(directory).resolve())))
            for name in ('dxil.dll', 'dxcompiler.dll'):
                _dll_handles.append(ctypes.WinDLL(str(Path(directory) / name)))
            return
        except OSError:
            _dll_handles.clear()


class GpuModel:
    def __init__(self, path=None, threads=4, content=None):
        from ai_edge_litert import schema_py_generated as schema
        from ai_edge_litert.compiled_model import CompiledModel, HardwareAccelerator, Options, CpuOptions, GpuOptions
        load_windows_dxc()
        self.content = content or Path(path).read_bytes()
        graph = schema.Model.GetRootAsModel(self.content, 0).Subgraphs(0)
        self.shapes = [tuple(graph.Tensors(graph.Outputs(i)).ShapeAsNumpy()) for i in range(graph.OutputsLength())]
        if any(not shape or any(dim <= 0 for dim in shape) for shape in self.shapes):
            raise ValueError('GPU wrapper requires known static output shapes')
        input_shape = tuple(graph.Tensors(graph.Inputs(0)).ShapeAsNumpy())
        types = {schema.TensorType.FLOAT32: np.float32, schema.TensorType.UINT8: np.uint8,
                 schema.TensorType.INT8: np.int8, schema.TensorType.INT32: np.int32}
        dtype = types[graph.Tensors(graph.Inputs(0)).Type()]
        self.input = {'shape': input_shape, 'dtype': dtype}
        self.model = CompiledModel.from_buffer(self.content, options=Options(
            HardwareAccelerator.GPU | HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads),
            gpu_options=GpuOptions(allow_src_quantized_fc_conv_ops=True)))
        self.inputs = self.model.create_input_buffers(0)
        self.outputs = self.model.create_output_buffers(0)
        if len(self.outputs) != len(self.shapes):
            raise RuntimeError('Compiled output count differs from graph output count')
        self(np.zeros(input_shape, dtype))

    def __call__(self, value):
        self.inputs[0].write(np.ascontiguousarray(value, dtype=self.input['dtype']))
        self.model.run_by_index(0, self.inputs, self.outputs)
        return [buffer.read(int(np.prod(shape)), np.float32).reshape(shape)
                for buffer, shape in zip(self.outputs, self.shapes)]
