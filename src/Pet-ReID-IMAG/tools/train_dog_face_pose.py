"""Train a small one-stage dog *face* detector with 6 keypoints (edge model).

The face-recognition recipe: a single network that outputs face boxes and
facial keypoints (RetinaFace / SCRFD / YOLOv5-face style), small enough for
phones, trained on real annotated dogs.

Data: the Ultralytics dog-pose dataset (StanfordExtra annotations on Stanford
Dogs, 6773 train / 1703 val images, 24 keypoints with visibility).  ``prepare``
keeps the face subset -- both eyes, nose, chin, both ear bases -- and derives a
face box from the visible face points.  Instances without a visible nose and
at least two other face points are dropped (back of head, heavy occlusion);
images left without a face are kept as negatives.

The released labels contain **no eye annotations** (StanfordExtra keypoints
20-23 are empty).  ``pseudolabel`` therefore adds eyes from a large teacher --
the local AnyFace YOLOv5l6 ONNX export, or the DogFLW 46-point landmarker run
directly on the ground-truth face boxes -- the usual distillation route for
small face detectors.  A teacher face is used only if its nose agrees with
the ground-truth nose and its eye confidence is high, so teacher errors (often
in strong profiles) are discarded; eyes it cannot place stay unlabelled.

    # 1) convert (download https://github.com/ultralytics/assets/releases/download/v0.0.0/dog-pose.zip)
    python tools/train_dog_face_pose.py prepare --source datasets/dog-pose --output datasets/dog-face6
    # 2) add teacher eyes (AnyFace ONNX from tools/export_anyface_onnx.py, or a DogFLW directory)
    python tools/train_dog_face_pose.py pseudolabel --data datasets/dog-face6 --teacher models/pretrained/dog_capture_v1/anyface_yolov5l6_640.onnx
    # 3) train (RTX 4060: ~1-2 h for 100 epochs at 640) and 4) export ONNX + sidecar for the engine
    python tools/train_dog_face_pose.py train --data datasets/dog-face6/dog-face6.yaml --epochs 100 --device 0
    python tools/train_dog_face_pose.py export --weights runs/pose/dog_face6/weights/best.pt --imgsz 320

Export size: a 12-epoch CPU model trained at 320 gave the same validation
accuracy at 320 and 416 input (face recall 0.904 / 0.901) at 10.4 vs 17.0 ms,
so 320 is the default; use 416/640 only for distant dogs, and measure with
``python -m dog_capture bench``.

Licences: Ultralytics is AGPL-3.0 (models you train with it are covered unless
you hold an Enterprise licence); StanfordExtra / Stanford Dogs are research
datasets.  Check both before commercial deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]

# dog-pose (StanfordExtra) indices; "left/right" there are anatomical.  The
# engine's names follow the RetinaFace convention (image-left in a frontal
# view = the dog's right), so the dog's right eye becomes ``left_eye``.
FACE_FROM_DOGPOSE = {'left_eye': 21, 'right_eye': 20, 'nose': 16, 'chin': 17,
                     'left_ear_base': 15, 'right_ear_base': 14}
FACE_ORDER = ('left_eye', 'right_eye', 'nose', 'chin', 'left_ear_base', 'right_ear_base')
FLIP_IDX = [1, 0, 2, 3, 5, 4]


def face_instance(values, margin=0.18, min_points=3):
    """values: one dog-pose label row (class, box, 24 x (x, y, v)), normalised.

    Returns the converted row or None.  The box is the bounding box of the
    visible face points expanded by ``margin`` of its larger side, so the
    face box is defined the same way in frontal and profile views.
    """
    kps = [values[5 + 3 * i: 8 + 3 * i] for i in range(24)]
    points = []
    for name in FACE_ORDER:
        x, y, v = kps[FACE_FROM_DOGPOSE[name]]
        visible = v > 0 and 0 <= x <= 1 and 0 <= y <= 1 and (x > 0 or y > 0)
        points.append((x, y, 2.0 if visible else 0.0))
    visible = [(x, y) for x, y, v in points if v > 0]
    if points[2][2] == 0 or len(visible) < min_points:
        return None
    xs, ys = [p[0] for p in visible], [p[1] for p in visible]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    side = max(w, h, 0.02)
    pad = margin * side
    x1, x2 = max(0.0, min(xs) - pad), min(1.0, max(xs) + pad)
    y1, y2 = max(0.0, min(ys) - pad), min(1.0, max(ys) + pad)
    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
        return None
    row = [0, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    for x, y, v in points:
        row += [x, y, v] if v > 0 else [0.0, 0.0, 0.0]
    return row


def prepare(args):
    source, output = Path(args.source), Path(args.output)
    stats = {}
    for split in ('train', 'val'):
        images = sorted((source / 'images' / split).glob('*.jpg'))
        if args.limit:
            images = images[:args.limit]
        (output / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output / 'labels' / split).mkdir(parents=True, exist_ok=True)
        kept = faces = negatives = 0
        for image in images:
            label = source / 'labels' / split / (image.stem + '.txt')
            rows = []
            if label.is_file():
                for line in label.read_text().splitlines():
                    values = [float(v) for v in line.split()]
                    if len(values) >= 5 + 72:
                        row = face_instance(values)
                        if row is not None:
                            rows.append(row)
            target = output / 'images' / split / image.name
            if not target.exists():
                try:
                    target.hardlink_to(image)
                except OSError:
                    shutil.copy2(image, target)
            (output / 'labels' / split / (image.stem + '.txt')).write_text(
                ''.join(' '.join(f'{v:.6g}' for v in row) + '\n' for row in rows))
            kept += 1
            faces += len(rows)
            negatives += not rows
        stats[split] = {'images': kept, 'faces': faces, 'images_without_face': negatives}
    data = {'path': str(output.resolve()), 'train': 'images/train', 'val': 'images/val',
            'kpt_shape': [6, 3], 'flip_idx': FLIP_IDX, 'names': {0: 'dog_face'},
            'kpt_names': {0: list(FACE_ORDER)}}
    try:
        import yaml
        (output / 'dog-face6.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
    except ImportError:
        (output / 'dog-face6.yaml').write_text(json.dumps(data, indent=2))
    (output / 'conversion.json').write_text(json.dumps({'source': str(source), 'stats': stats,
                                                        'face_from_dogpose': FACE_FROM_DOGPOSE}, indent=2))
    print(json.dumps(stats, indent=2))


def match_teacher(row, teacher, width, height, tolerance=0.6):
    """Copy teacher eyes into one converted label row if the noses agree.

    ``row``: class, box, 6 x (x, y, v) normalised; ``teacher``: Detection list
    in pixels.  Returns (row, matched).  The teacher face must contain the
    ground-truth nose, and its nose must lie within ``tolerance`` x its eye
    distance of the ground-truth nose.  0.6 allows for the different nose
    definitions (DogFLW nose-pad centroid vs StanfordExtra nose tip: median
    0.35 eye distances apart even on frontal faces) while rejecting a teacher
    that found another face.
    """
    gx, gy = row[5 + 3 * 2] * width, row[5 + 3 * 2 + 1] * height
    best, best_error = None, float('inf')
    for det in teacher:
        names = det.schema.names
        if 'left_eye' not in names or 'nose' not in names:
            continue
        left, right, nose = (det.keypoints[names.index(k)] for k in ('left_eye', 'right_eye', 'nose'))
        iod = float(((left - right) ** 2).sum() ** 0.5)
        x1, y1, x2, y2 = det.box
        if iod < 4 or not (x1 <= gx <= x2 and y1 <= gy <= y2):
            continue
        error = float(((nose - (gx, gy)) ** 2).sum() ** 0.5) / iod
        if error < tolerance and error < best_error:
            best, best_error = (det, left, right), error
    if best is None:
        return row, False
    det, left, right = best
    row = list(row)
    scores = det.kp_scores
    for slot, point, name in ((0, left, 'left_eye'), (1, right, 'right_eye')):
        confident = scores is None or scores[det.schema.names.index(name)] >= 0.5
        if confident and 0 <= point[0] < width and 0 <= point[1] < height:
            row[5 + 3 * slot: 8 + 3 * slot] = [float(point[0]) / width, float(point[1]) / height, 2.0]
    return row, True


def pseudolabel(args):
    import sys
    sys.path.insert(0, str(ROOT))
    import cv2
    import numpy as np
    from dog_capture.engine import CaptureConfig
    from dog_capture.engine.zoo import _guess_kind, build_detector
    teacher_path = Path(args.teacher)
    kind = 'dogflw' if teacher_path.is_dir() else _guess_kind(teacher_path)
    config = CaptureConfig().merged({'detector': {'kind': kind, 'model': str(teacher_path),
                                                  'score_threshold': args.score, 'input_size': args.imgsz},
                                     'runtime': {'providers': args.providers}})
    teacher = build_detector(config, kind, teacher_path)
    data = Path(args.data)
    stats = {'instances': 0, 'matched': 0, 'eyes_labelled': 0}
    for split in ('train', 'val'):
        for label in sorted((data / 'labels' / split).glob('*.txt')):
            rows = [[float(v) for v in line.split()] for line in label.read_text().splitlines() if line.strip()]
            if not rows:
                continue
            image = cv2.imdecode(np.fromfile(str(data / 'images' / split / (label.stem + '.jpg')), np.uint8),
                                 cv2.IMREAD_COLOR)
            if image is None:
                continue
            h, w = image.shape[:2]
            if hasattr(teacher, 'landmarks_in_box'):  # landmark-only teacher on ground-truth boxes
                found = None
            else:
                found = teacher.detect(image)
            updated = []
            for row in rows:
                if found is None:
                    cx, cy, bw, bh = row[1] * w, row[2] * h, row[3] * w, row[4] * h
                    one = teacher.landmarks_in_box(image, (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
                    candidates = [one] if one is not None else []
                else:
                    candidates = found
                row, matched = match_teacher(row, candidates, w, h)
                stats['instances'] += 1
                stats['matched'] += matched
                stats['eyes_labelled'] += sum(row[5 + 3 * i + 2] > 0 for i in (0, 1))
                updated.append(row)
            label.write_text(''.join(' '.join(f'{v:.6g}' for v in row) + '\n' for row in updated))
    stats['teacher'] = str(teacher_path)
    (data / 'pseudolabel.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def train(args):
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                workers=args.workers, name=args.name, fraction=args.fraction, exist_ok=True,
                degrees=args.degrees, fliplr=0.5, mosaic=1.0, close_mosaic=min(10, max(args.epochs // 5, 0)),
                patience=max(20, args.epochs // 4), plots=False)


def export(args):
    from ultralytics import YOLO
    model = YOLO(args.weights)
    path = Path(model.export(format='onnx', imgsz=args.imgsz, opset=args.opset, simplify=True, dynamic=args.dynamic))
    target = Path(args.output) if args.output else REPO / 'models' / 'pretrained' / 'dog_capture_v1' / 'dog_face_pose.onnx'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    with target.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    meta = {'kind': 'yolo_pose', 'schema': 'dogface6', 'kpt_shape': [6, 3], 'input_size': args.imgsz,
            'source_weights': str(args.weights), 'sha256': digest,
            'license': 'Trained with Ultralytics (AGPL-3.0) on StanfordExtra/Stanford Dogs (research use)'}
    if args.dynamic:   # free input size: full frames at input_size, refinement crops at refine.crop_size
        meta['dynamic'] = True
    target.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({'onnx': str(target), **meta}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--source', required=True, help='Ultralytics dog-pose root (images/, labels/)')
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=0, help='first N images per split (smoke tests)')
    p.set_defaults(func=prepare)
    q = sub.add_parser('pseudolabel', help='add teacher eye positions to a prepared dataset (in place)')
    q.add_argument('--data', required=True, help='output directory of prepare')
    q.add_argument('--teacher', required=True, help='AnyFace/YOLO-pose ONNX file or DogFLW directory')
    q.add_argument('--score', type=float, default=0.3)
    q.add_argument('--imgsz', type=int, default=640)
    q.add_argument('--providers', default='auto')
    q.set_defaults(func=pseudolabel)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--model', default='yolo11n-pose.pt', help='COCO-pose pretrained start point')
    t.add_argument('--epochs', type=int, default=100)
    t.add_argument('--imgsz', type=int, default=416)
    t.add_argument('--batch', type=int, default=32)
    t.add_argument('--device', default='')
    t.add_argument('--workers', type=int, default=8)
    t.add_argument('--fraction', type=float, default=1.0)
    t.add_argument('--degrees', type=float, default=15.0, help='rotation augmentation (tilted heads / phones)')
    t.add_argument('--name', default='dog_face6')
    t.set_defaults(func=train)
    e = sub.add_parser('export')
    e.add_argument('--weights', required=True)
    e.add_argument('--imgsz', type=int, default=320, help='network input; 320 = fastest at equal accuracy (measured)')
    e.add_argument('--opset', type=int, default=17)
    e.add_argument('--dynamic', action='store_true',
                   help='free input height/width (same weights; lets refine.crop_size run smaller crops)')
    e.add_argument('--output')
    e.set_defaults(func=export)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
