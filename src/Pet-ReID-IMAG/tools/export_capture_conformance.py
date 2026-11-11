"""Export conformance vectors: the Python engine is the reference for every port.

    python tools/export_capture_conformance.py      # writes dog_capture/protocol/conformance/*.json

Each file holds inputs and the exact outputs of the reference implementation
for one component.  The TypeScript SDK test-suite (frontend/dog-capture) loads
these files and must reproduce them within the stated tolerances; a Kotlin or
C++ port should do the same.  Re-export after any intentional behaviour change.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import sys
import zlib

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from dog_capture.engine import CaptureConfig  # noqa: E402
from dog_capture.engine.alignment import align_face, load_template, umeyama_similarity  # noqa: E402
from dog_capture.engine.guidance import guidance  # noqa: E402
from dog_capture.engine.pose import ViewClassifier, face_geometry  # noqa: E402
from dog_capture.engine.quality import QualityAssessor, _laplacian_energy, blur_effect  # noqa: E402
from dog_capture.engine.schema import SCHEMAS, Detection  # noqa: E402
from dog_capture.engine.selection import SETTLE_FACTOR, Shot, TrackShots  # noqa: E402
from dog_capture.engine.tracking import ByteTracker, OneEuroFilter  # noqa: E402
from dog_engine_synthetic import canonical_points, render  # noqa: E402

OUT = ROOT / 'dog_capture' / 'protocol' / 'conformance'


def r(value, digits=6):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [r(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: r(v, digits) for k, v in value.items()}
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    return round(float(value), digits)


def pack(array):
    array = np.ascontiguousarray(array, np.uint8)
    return {'shape': list(array.shape), 'deflate_b64': base64.b64encode(zlib.compress(array.tobytes(), 9)).decode()}


def face_points(center=(200, 200), iod=100, yaw=0.0, roll=0.0):
    a = math.radians(roll)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return canonical_points(yaw) * iod @ rot.T + center


def detection(points, schema='anyface5', box=(50, 50, 350, 350), score=0.9, kp=None):
    return Detection(np.asarray(box, np.float32), score, np.asarray(points, np.float32), schema, kp)


def geometry_cases(config):
    cases = []
    for yaw in (-0.6, -0.3, 0.0, 0.12, 0.45):
        for roll in (0, 37, 90, 180, 270):
            for swap in (False, True):
                p = face_points(yaw=yaw, roll=roll)
                if swap:
                    p[[0, 1]] = p[[1, 0]]
                    p[[3, 4]] = p[[4, 3]]
                cases.append(('anyface5', p, None))
    flat = face_points()
    flat[2, 1] = flat[0, 1] + 2
    cases.append(('anyface5', flat, None))
    odd = face_points(yaw=0.5)
    odd[3:, 0] -= 80
    cases.append(('anyface5', odd, None))
    cases.append(('anyface5', face_points(iod=10), None))
    p = face_points(yaw=0.35)[:3] * 1.0
    p6 = np.vstack([p, [p[2] + [0, 40]], p[0] + [-20, -60], p[1] + [20, -60]])
    cases.append(('dogface6', p6, [1, 1, 1, .9, .8, .7]))
    cases.append(('core3', face_points(yaw=-0.2)[:3], [.9, .9, .9]))
    out = []
    for schema, points, kp in cases:
        det = detection(points, schema, kp=kp)
        g = face_geometry(det, config.pose)
        out.append({'schema': schema, 'box': r(det.box), 'keypoints': r(points), 'kp_scores': kp,
                    'expected': r({'yaw': g.yaw, 'pitch': g.pitch, 'roll_deg': g.roll_deg, 'iod': g.iod,
                                   'lower_yaw': g.lower_yaw, 'approx_deg': g.approx_deg, 'valid': g.valid,
                                   'reason': g.reason, 'swapped': g.swapped,
                                   'points': {k: list(v) for k, v in g.points.items()}})})
    return out


def similarity_cases():
    rng = np.random.default_rng(5)
    out = []
    for _ in range(6):
        src = rng.random((5, 2)) * 300
        angle, scale = rng.uniform(-math.pi, math.pi), rng.uniform(0.2, 3)
        rot = scale * np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        dst = src @ rot.T + rng.uniform(-50, 50, 2) + rng.normal(0, 2, (5, 2))
        out.append({'src': r(src), 'dst': r(dst), 'matrix': r(umeyama_similarity(src, dst))})
    return out


def view_cases(config):
    sequences = {
        'dwell': [(i / 30, 0.45, 0, True) for i in range(8)],
        'hysteresis_and_break': ([(i / 30, 0.12, 0, True) for i in range(6)] + [(6 / 30, 0.16, 0, True),
                                 (7 / 30, 0.12, 0, False)] + [(i / 30, 0.0, 0, True) for i in range(8, 14)]
                                 + [(2.0, 0.0, 0, True)]),
        'turn': [(i / 30, float(y), 5, True) for i, y in enumerate(np.r_[np.zeros(6), np.linspace(0, -.5, 8),
                                                                           -.5 * np.ones(6)])],
        'roll_limit': [(i / 30, 0.0, 40, True) for i in range(5)],
    }
    out = {}
    for name, steps in sequences.items():
        classifier = ViewClassifier(config.pose)
        expected = []
        for t, yaw, roll, valid in steps:
            g = face_geometry(detection(face_points(yaw=yaw, roll=0)), config.pose)
            g.yaw, g.roll_deg, g.valid = yaw, roll, valid
            s = classifier.update(t, g)
            expected.append(r({'raw': s.raw, 'view': s.view, 'stable': s.stable, 'smoothed_yaw': s.smoothed_yaw,
                               'observations': s.observations}))
        out[name] = {'steps': [list(s) for s in steps], 'expected': expected}
    return out


def one_euro_cases():
    f = OneEuroFilter(1.5, 0.5)
    rng = np.random.default_rng(0)
    steps = [(i / 30, float(0.3 + rng.normal(0, .05)) if i < 30 else 1.0) for i in range(45)]
    return {'min_cutoff': 1.5, 'beta': 0.5, 'steps': r(steps), 'expected': r([float(f(t, x)) for t, x in steps])}


def tracking_cases(config):
    frames = []
    for i in range(40):
        t = i / 30
        if i in (8, 9):
            continue
        dets = [[100 + 6 * i, 100, 200 + 6 * i, 200, 0.3 if i == 12 else 0.9]]
        if 5 <= i < 30:
            dets.append([400, 60 - i, 480, 140 - i, 0.85])
        if i == 20:
            dets.append([10, 300, 40, 330, 0.4])  # low-score clutter never starts a track
        frames.append((t, dets))
    tracker = ByteTracker(config.tracker)
    expected = []
    for t, dets in frames:
        objects = [detection(face_points((b[0] + 50, b[1] + 50), 40), box=b[:4], score=b[4]) for b in dets]
        active = tracker.update(objects, t)
        expected.append([{'id': a.id, 'box': r(a.box, 3)} for a in sorted(active, key=lambda a: a.id)])
    return {'frames': [{'t': r(t), 'detections': r(d)} for t, d in frames], 'expected': expected}


class _State:
    def __init__(self, view, stable):
        self.view, self.stable = view, stable


def _run_selection(script, selection, shots):
    ts = TrackShots(1, ('front', 'left'), selection, {'front': shots})
    events, outstanding = [], []
    for action, view, stable, t, score in script:
        state = _State(view, stable)
        if action == 'observe':
            events += ts.observe(state, t)
        else:
            shot = Shot(view, 1, int(t * 30), t, score, None, None, None, np.zeros((1, 1, 3), np.uint8), None)
            new, _ = ts.offer(shot, state, t)
            events += new
        outstanding.append(ts.outstanding({'front'}))
    return {'expected_slots': {v: [r([s.time_s, s.score]) for s in ts.slots[v]] for v in ts.view_names},
            'expected_locked': sorted(ts.locked),
            'expected_events': [[e['event'], e['view'], r(e.get('time_s'))] for e in events],
            # guidance targets after each step, with 'front' required and 'left' optional
            'expected_outstanding': outstanding}


def selection_cases(config):
    script = [('observe', 'front', False, 0.0, None), ('offer', 'front', False, 0.0, 0.9),
              ('observe', 'front', True, 0.2, None), ('offer', 'front', True, 0.25, 0.5),
              ('offer', 'front', True, 0.9, 0.7), ('offer', 'front', True, 1.6, 0.95),
              ('observe', 'left', False, 1.7, None), ('offer', 'left', False, 1.7, 0.8),
              ('observe', None, False, 1.8, None), ('observe', 'left', True, 1.9, None),
              ('offer', 'left', True, 2.0, 0.6), ('offer', 'left', True, 4.5, 0.99)]
    # Only mediocre shots (below good_enough): full views must still settle.
    settle = [('offer', 'front', True, 0.0, 0.5), ('offer', 'front', True, 0.6, 0.55),
              ('observe', 'front', True, 1.5, None), ('observe', 'front', True, 2.1, None),
              ('offer', 'left', True, 2.2, 0.6), ('observe', 'left', True, 2.7, None),
              ('observe', 'left', True, 4.3, None)]
    out, settle_out = {}, {}
    for mode in ('fast', 'best', 'hybrid'):
        for shots in (1, 2):
            selection = config.merged({'selection': {'mode': mode, 'good_enough': .65, 'improve_window_s': 1.0,
                                                     'min_gap_s': .3}}).selection
            out[f'{mode}_front{shots}'] = {'mode': mode, 'front_shots': shots,
                                           **_run_selection(script, selection, shots)}
            settle_out[f'{mode}_front{shots}'] = {'mode': mode, 'front_shots': shots,
                                                  **_run_selection(settle, selection, shots)}
    return {'script': [list(s) for s in script], 'good_enough': .65, 'improve_window_s': 1.0, 'min_gap_s': .3,
            'settle_factor': SETTLE_FACTOR, 'required': ['front'], 'cases': out,
            'settle_script': [list(s) for s in settle], 'settle_cases': settle_out}


def quality_cases(config):
    assessor = QualityAssessor(config.quality, load_template('petface_dog'))
    views = {v.name: v for v in config.pose.views}
    out = {'frames': []}
    for name, kwargs in (('sharp', {}), ('soft', {'blur': 1.2}), ('dark', {'gain': .3}), ('side', {'yaw': .45})):
        image, dets = render(size=(240, 320), center=(160, 120), iod=70, **kwargs)
        det = dets[0]
        g = face_geometry(det, config.pose)
        target = views.get('right' if kwargs.get('yaw') else 'front')
        report, aligned = assessor.assess(image, det, g, target, 30.0)
        gray = cv2.cvtColor(aligned.image, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = assessor.mask_rect
        out['frames'].append({
            'name': name, 'frame_bgr': pack(image), 'box': r(det.box), 'keypoints': r(det.keypoints),
            'target': target.name, 'speed_px_s': 30.0,
            'expected_matrix': r(aligned.matrix), 'expected_residual': r(aligned.residual),
            'aligned_gray': pack(gray),
            'expected_metrics_on_aligned_gray': r({
                'laplacian': _laplacian_energy(gray, assessor.mask), 'blur_effect': blur_effect(gray[y1:y2, x1:x2]),
                'eye_laplacian': [_laplacian_energy(gray[b:d, a:c]) for a, b, c, d in assessor.eyes]}),
            'expected_report': r(report.to_dict())})
    out['mask_rect'] = [int(v) for v in assessor.mask_rect]
    out['mask'] = pack(assessor.mask.astype(np.uint8))
    out['eye_patches'] = [list(map(int, e)) for e in assessor.eyes]
    return out


def guidance_cases():
    class Face:
        def __init__(self, reasons, view=None, stable=False, valid=True):
            self.quality = type('Q', (), {'reasons': reasons})()
            self.view_state = type('S', (), {'view': view, 'stable': stable})()
            self.geometry = type('G', (), {'valid': valid})()
    cases = [({'face': None, 'missing': ['front'], 'rejected_species': False, 'dog_seen': False}, None),
             ({'face': None, 'missing': ['front'], 'rejected_species': True, 'dog_seen': False}, None),
             ({'face': None, 'missing': ['front'], 'rejected_species': False, 'dog_seen': True}, None),
             ({'face': {'reasons': ['blurred', 'too_dark']}, 'missing': ['front']}, None),
             ({'face': {'reasons': ['extreme_pose']}, 'missing': ['left']}, None),
             ({'face': {'reasons': []}, 'missing': ['left'], 'speed_iod': 5.0}, None),
             ({'face': {'reasons': [], 'view': 'left', 'stable': False}, 'missing': ['left', 'right']}, None),
             ({'face': {'reasons': [], 'view': 'front', 'stable': True}, 'missing': ['right']}, None),
             ({'face': {'reasons': [], 'view': 'front', 'stable': True}, 'missing': ['left', 'front']}, None),
             ({'face': {'reasons': []}, 'missing': []}, None)]
    out = []
    for spec, _ in cases:
        f = spec['face']
        face = None if f is None else Face(f['reasons'], f.get('view'), f.get('stable', False))
        result = guidance(face, spec['missing'], rejected_species=spec.get('rejected_species', False),
                          speed_iod=spec.get('speed_iod', 0.0), dog_seen=spec.get('dog_seen', False))
        out.append({'input': spec, 'expected': result})
    return out


def main():
    config = CaptureConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        'geometry.json': geometry_cases(config), 'similarity.json': similarity_cases(),
        'views.json': view_cases(config), 'one_euro.json': one_euro_cases(),
        'tracking.json': tracking_cases(config), 'selection.json': selection_cases(config),
        'quality.json': quality_cases(config), 'guidance.json': guidance_cases(),
        'config_default.json': config.to_dict(),
        'schemas.json': {n: {'names': list(s.names), 'pairs': [list(p) for p in s.pairs], 'yaw_scale': s.yaw_scale}
                         for n, s in SCHEMAS.items()},
    }
    meta = {'reference': 'dog_capture (Python) ' + __import__('dog_capture.engine', fromlist=['x']).__version__,
            'tolerances': {'geometry': 1e-4, 'similarity': 1e-6, 'views.smoothed_yaw': 1e-6, 'one_euro': 1e-9,
                           'tracking.box': 1e-2, 'quality.metrics_on_aligned_gray': '1% relative',
                           'quality.report.score': 0.03, 'quality.report.factors': 0.05},
            'files': sorted(files)}
    for name, value in {**files, 'index.json': meta}.items():
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')), encoding='utf8')
    print(json.dumps({name: (OUT / name).stat().st_size for name in sorted(files)}, indent=1))


if __name__ == '__main__':
    main()
