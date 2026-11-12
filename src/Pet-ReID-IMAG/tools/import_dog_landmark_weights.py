"""Recover trainable crop-landmarker weights from the supplied inference ONNX.

This is a warm start, not resumption of the original optimizer. ONNX-exported
Conv/BN folds are retained as biased Conv + Identity, so invented BN running
statistics cannot change the starting function. Reject unknown/unmapped weights
and require output parity before saving anything as a training checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def module_path(node_name):
    parts = node_name.strip('/').split('/')
    if parts[0] != 'm' or parts[-1] != 'Conv':
        raise ValueError(f'Unexpected Conv scope: {node_name}')
    path = []
    for part in parts[1:-1]:
        # Exporter repeats Sequential names: /lat5/lat5.0 -> lat5.0.
        if path and part.startswith(path[-1] + '.'):
            path[-1] = part
        else:
            path.append(part)
    return '.'.join(path)


def prepare_fused_conv(model, path):
    import torch
    import torch.nn as nn
    conv = model.get_submodule(path)
    if not isinstance(conv, nn.Conv2d):
        raise ValueError(f'{path} is not Conv2d')
    if conv.bias is None:
        conv.bias = nn.Parameter(torch.zeros(conv.out_channels, dtype=conv.weight.dtype))
    parent_path, child = path.rsplit('.', 1) if '.' in path else ('', path)
    parent = model.get_submodule(parent_path)
    if child == 'conv' and isinstance(getattr(parent, 'bn', None), nn.BatchNorm2d):
        parent.bn = nn.Identity()
    elif isinstance(parent, nn.Sequential) and child.isdigit():
        index = int(child) + 1
        if index < len(parent) and isinstance(parent[index], nn.BatchNorm2d):
            parent[index] = nn.Identity()


def packed(model, tensor):
    import torch
    xy, logits, face, _, turn = model(tensor)
    return torch.cat([(xy[:, :6] * 128).flatten(1), face * 128, logits, turn], 1)


def load_checkpoint(path, device='cpu'):
    import torch
    from train_dog_face_landmarks import build_model
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != 'dogface6_crop_yolo11n_v1':
        raise ValueError('Not a versioned dog crop-landmark checkpoint')
    model = build_model()
    for name in checkpoint['fused_convs']:
        prepare_fused_conv(model, name)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device), checkpoint


def recover(onnx_path, output, samples=12):
    import onnx
    import onnxruntime as ort
    import torch
    from train_dog_face_landmarks import build_model
    torch.set_num_threads(4)
    source = Path(onnx_path)
    target = Path(output)
    if target.exists():
        raise FileExistsError(f'Refusing to replace checkpoint: {target}')
    graph = onnx.load(source)
    onnx.checker.check_model(graph)
    values = {v.name: onnx.numpy_helper.to_array(v).copy() for v in graph.graph.initializer}
    model = build_model()
    fused, assigned = [], set()
    for node in graph.graph.node:
        if node.op_type != 'Conv':
            continue
        name = module_path(node.name)
        prepare_fused_conv(model, name)
        layer = model.get_submodule(name)
        if len(node.input) != 3:
            raise ValueError(f'Expected fused bias on {node.name}')
        for suffix, original in zip(('weight', 'bias'), node.input[1:]):
            parameter = getattr(layer, suffix)
            value = torch.from_numpy(values[original])
            if parameter.shape != value.shape:
                raise ValueError(f'Shape mismatch: {name}.{suffix}')
            parameter.data.copy_(value)
            assigned.add(f'{name}.{suffix}')
        fused.append(name)
    state = model.state_dict()
    for key, value in values.items():
        if not key.startswith('m.'):
            continue
        name = key[2:]
        if name not in state:
            raise ValueError(f'Unknown named initializer: {key}')
        state[name].copy_(torch.from_numpy(value))
        assigned.add(name)
    missing = set(dict(model.named_parameters())) - assigned
    if missing:
        raise ValueError(f'Unmapped trainable parameters: {sorted(missing)}')
    model.eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(source), options, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(20261112)
    maxima = []
    with torch.no_grad():
        for index in range(samples):
            x = rng.random((1, 3, 128, 128), dtype=np.float32)
            if index == 0:
                x.fill(0)
            elif index == 1:
                x.fill(114 / 255)
            want = session.run(None, {session.get_inputs()[0].name: x})[0]
            got = packed(model, torch.from_numpy(x)).numpy()
            np.testing.assert_allclose(got, want, atol=1e-3, rtol=1e-5)
            maxima.append(float(np.max(np.abs(got - want))))
    metadata = json.loads(source.with_suffix('.json').read_text(encoding='utf8'))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if metadata.get('sha256') != digest:
        raise ValueError('Source sidecar hash mismatch')
    parity = {'samples':samples,'max_abs_output_difference':max(maxima),'atol':1e-3,'rtol':1e-5}
    checkpoint = {'format_version':1,'architecture':'dogface6_crop_yolo11n_v1',
                  'state_dict':model.state_dict(),'fused_convs':fused,
                  'source_onnx_sha256':digest,'source_metadata':metadata,
                  'import_parity':parity,'optimizer_state_available':False}
    target.parent.mkdir(parents=True,exist_ok=True)
    torch.save(checkpoint, target)
    report = {k:v for k,v in checkpoint.items() if k != 'state_dict'}
    report['checkpoint_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.import.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(recover(args.onnx,args.output),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
