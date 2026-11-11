"""Model discovery, download and construction from one manifest.

Put model files in ``models/pretrained/dog_capture_v1/`` (or set
``DOG_CAPTURE_MODELS``).  ``detector.kind = auto`` uses the first available
face model in priority order; the manifest records source, license and hash
of every file, and ``python -m dog_capture models`` shows what is missing.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST = REPO_ROOT / 'assets' / 'dog_capture' / 'model_zoo_v1.json'
LEGACY_DIR = REPO_ROOT / 'models' / 'pretrained' / 'dog_face_landmarks_v1'


def model_dir():
    return Path(os.environ.get('DOG_CAPTURE_MODELS', REPO_ROOT / 'models' / 'pretrained' / 'dog_capture_v1'))


def manifest():
    return json.loads(MANIFEST.read_text(encoding='utf8'))


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify_model(path, expected=None, metadata=None):
    """Verify a pinned model and its sidecar before creating a runtime session.

    Explicit custom models use their own sidecar pin. Auto discovery additionally
    requires the catalog pin, so a corrupted default never silently falls back.
    """
    path = Path(path)
    if metadata is None:
        sidecar = path.with_suffix('.json')
        metadata = json.loads(sidecar.read_text(encoding='utf8')) if sidecar.is_file() else {}
    if not isinstance(metadata, dict):
        raise ValueError(f'{path}: model metadata must be a JSON object')
    digest = sha256(path)
    for pinned in (expected, metadata.get('sha256')):
        if pinned and digest != str(pinned).lower():
            raise ValueError(f'{path}: SHA-256 mismatch (expected {pinned}, got {digest})')
    return digest, metadata


def import_face(source, metadata=None):
    """Install the registered reference pair only after hash and runtime validation."""
    from .detectors import YoloPoseDetector

    source = Path(source)
    sidecar = Path(metadata) if metadata else source.with_suffix('.json')
    meta = json.loads(sidecar.read_text(encoding='utf8'))
    entry = next(e for e in manifest()['models'] if e['name'] == 'dog_face_pose')
    verify_model(source, entry['sha256'], meta)
    for key in ('kind', 'schema', 'kpt_shape', 'input_size', 'sha256'):
        if meta.get(key) != entry.get(key):
            raise ValueError(f'{sidecar}: {key} does not match the registered dog_face_pose reference')
    directory = model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    # Stage in the destination filesystem; bad metadata/graphs never replace a working pair.
    with tempfile.TemporaryDirectory(prefix='.import-', dir=directory) as temporary:
        candidate = Path(temporary) / entry['file']
        shutil.copyfile(source, candidate)
        shutil.copyfile(sidecar, candidate.with_suffix('.json'))
        detector = YoloPoseDetector(candidate, providers='cpu', threads=1)
        detector.warmup(1)
        del detector
        target = directory / entry['file']
        candidate.replace(target)
        candidate.with_suffix('.json').replace(target.with_suffix('.json'))
    return target


def status():
    """Every manifest entry with presence, hash check and how to obtain it."""
    rows = []
    for entry in manifest()['models']:
        path = model_dir() / entry['file'] if entry['kind'] != 'dogflw' else LEGACY_DIR
        present = path.exists() if entry['kind'] != 'dogflw' else (LEGACY_DIR / 'dog_face_landmarks_full.tflite').exists()
        row = {**entry, 'path': str(path), 'present': present}
        if present and entry.get('sha256') and path.is_file():
            row['hash_ok'] = sha256(path) == entry['sha256']
        rows.append(row)
    return rows


def download(name):
    entry = next((e for e in manifest()['models'] if e['name'] == name), None)
    if entry is None or not entry.get('url'):
        raise ValueError(f'{name!r} has no download URL; see `python -m dog_capture models` for how to get it')
    target = model_dir() / entry['file']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and (not entry.get('sha256') or sha256(target) == entry['sha256']):
        return target
    temporary = target.with_suffix('.download')
    request = urllib.request.Request(entry['url'], headers={'User-Agent': 'dog-capture'})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as f:
        while block := response.read(1 << 20):
            f.write(block)
    if entry.get('sha256') and sha256(temporary) != entry['sha256']:
        temporary.unlink()
        raise ValueError(f'Checksum mismatch for {name}; download discarded')
    temporary.replace(target)
    return target


def _guess_kind(path):
    name = Path(path).name.lower()
    sidecar = Path(path).with_suffix('.json')
    if sidecar.is_file():
        try:
            kind = json.loads(sidecar.read_text(encoding='utf8')).get('kind')
            if kind:
                return kind
        except ValueError:
            pass
    if 'scrfd' in name:
        return 'scrfd'
    if 'anyface' in name or 'yolov5' in name:
        return 'anyface'
    return 'yolo_pose'


def find_face_model(config):
    """(kind, path) for the configured or first available face detector."""
    d = config.detector
    if d.kind == 'external':
        return 'external', None
    if d.model:
        path = Path(d.model)
        if not path.exists():
            raise FileNotFoundError(f'detector.model not found: {path}')
        if path.is_file():
            verify_model(path)
        return (d.kind if d.kind != 'auto' else ('dogflw' if path.is_dir() else _guess_kind(path))), path
    entries = [e for e in manifest()['models'] if e['role'] == 'face']
    for entry in sorted(entries, key=lambda e: e['priority']):
        if d.kind not in ('auto', entry['kind']):
            continue
        if entry['kind'] == 'dogflw':
            if (LEGACY_DIR / 'dog_face_landmarks_full.tflite').exists() and importlib.util.find_spec('ai_edge_litert'):
                return 'dogflw', LEGACY_DIR
            continue
        path = model_dir() / entry['file']
        if path.is_file():
            verify_model(path, entry.get('sha256'))
            return entry['kind'], path
    raise FileNotFoundError(
        f'No dog face model found in {model_dir()} (kind={d.kind}). Options (`python -m dog_capture models`): '
        'import the reference with `models --import-face dog_face_pose.onnx --metadata dog_face_pose.json`; '
        'export the local AnyFace weights with tools/export_anyface_onnx.py; train the 6-point edge model with '
        'tools/train_dog_face_pose.py; or use the v0.x DogFLW models in models/pretrained/dog_face_landmarks_v1 '
        '(tools/prepare_dog_capture_assets.py --models, needs ai-edge-litert); or pass --model PATH.')


def build_detector(config, kind=None, path=None):
    from . import detectors
    if kind is None:
        kind, path = find_face_model(config)
    d, r = config.detector, config.runtime
    common = dict(input_size=d.input_size, score_threshold=d.score_threshold, nms_iou=d.nms_iou,
                  max_faces=d.max_faces, providers=r.providers, threads=r.threads, cache_dir=r.cache_dir)
    if kind == 'external':
        return None
    if kind == 'yolo_pose':
        return detectors.YoloPoseDetector(path, schema=d.schema, **common)
    if kind == 'anyface':
        return detectors.AnyFaceDetector(path, **common)
    if kind == 'scrfd':
        return detectors.ScrfdDetector(path, **common)
    if kind == 'dogflw':
        device = {'tensorrt': 'trt', 'trt': 'trt', 'cpu': 'cpu'}.get(r.providers, 'auto')
        return detectors.LegacyDogFlwDetector(path, threads=r.threads or 4, device=device)
    raise ValueError(f'Unknown detector kind {kind!r}')


def build_body_gate(config, detector_kind):
    from . import detectors
    g, r = config.body_gate, config.runtime
    if g.enabled == 'off' or detector_kind in ('dogflw',):
        return None  # the DogFLW cascade already gates on a dog detector
    path = Path(g.model) if g.model else None
    if path is None:
        # nano first: 3.7 MB and ~4x faster than tiny; per-track verification absorbs its lower recall
        for name in ('yolox_nano.onnx', 'yolox_tiny.onnx', 'yolox_s.onnx'):
            if (model_dir() / name).is_file():
                path = model_dir() / name
                break
    if path is None or not path.is_file():
        if g.enabled == 'on':
            raise FileNotFoundError('body_gate.enabled=on but no YOLOX model found; run '
                                    '`python -m dog_capture models --download yolox_nano`')
        return None
    entry = next((e for e in manifest()['models'] if e['file'] == path.name and e['role'] == 'body'), {})
    verify_model(path, entry.get('sha256') if not g.model else None)
    return detectors.YoloxBodyGate(path, class_id=g.class_id, input_size=g.input_size,
                                   score_threshold=g.score_threshold, providers=r.providers,
                                   threads=r.threads, cache_dir=r.cache_dir)


def build_engine(config):
    from .engine import CaptureEngine
    kind, path = find_face_model(config)
    detector = build_detector(config, kind, path)
    gate = build_body_gate(config, kind)
    return CaptureEngine(config, detector, gate)
