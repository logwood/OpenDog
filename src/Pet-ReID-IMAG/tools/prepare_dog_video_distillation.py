"""Build reviewable detector pseudo labels from a fixed, video-disjoint manifest.

The frozen 320px detector is the only teacher; the crop tracker is never used for
labels. These labels measure teacher agreement, NOT independent ground truth.
Uncertain frames are rejected, not converted into negative training examples.
All paths in the source manifest are relative to --root; all generated image
paths are relative to --out. Outputs are immutable: use a new directory to rerun.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.engine.config import PoseConfig, RefineConfig
from dog_capture.engine.detectors import YoloPoseDetector
from dog_capture.engine.pose import face_geometry
from dog_capture.engine.refine import FaceRefiner, apply, crop_matrix, eye_contrast, flip_index, warp_crop


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path, root, verify=True):
    manifest = json.loads(Path(path).read_text(encoding='utf8'))
    groups, hashes, identifiers = {}, {}, set()
    sources = []
    protected = set(manifest.get('protected_sha256', []))
    for row in manifest['sources']:
        if row['split'] == 'excluded':
            continue
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError(f"Invalid split: {row['split']}")
        if row['id'] in identifiers:
            raise ValueError(f"Duplicate video id: {row['id']}")
        identifiers.add(row['id'])
        group = row['dog_group']
        if group in groups and groups[group] != row['split']:
            raise ValueError(f'Dog/group leakage: {group}')
        groups[group] = row['split']
        digest = row['sha256']
        if digest in hashes:
            raise ValueError(f'Duplicate video content: {row["id"]}')
        hashes[digest] = row['split']
        if digest in protected and row['split'] != 'test':
            raise ValueError(f'Protected regression video entered {row["split"]}: {row["id"]}')
        source = Path(root) / row['path']
        if verify and sha256(source) != digest:
            raise ValueError(f'Video hash mismatch: {source}')
        sources.append(row)
    if not any(s['split'] == 'train' for s in sources) or not any(s['split'] == 'val' for s in sources):
        raise ValueError('Need separate training and validation videos')
    return manifest, sources


def pose_bin(yaw):
    return 'left' if yaw <= -.25 else 'right' if yaw >= .25 else 'front'


def strict_label(frame, detection, refiner):
    """Filter globally, then mask individual unstable/unseen keypoints.

    The thresholds deliberately do not call the detector's confidence a
    visibility annotation. Visibility training is restricted to supported eyes
    and nose; unknown/invisible points receive no coordinate or visibility loss.
    """
    det = detection
    geometry = face_geometry(det, PoseConfig())
    if not det.extra.get('refined') or det.score < .55:
        return None, 'weak_refinement'
    if not geometry.valid or geometry.iod < 14 or abs(geometry.yaw) > .95:
        return None, 'geometry_or_resolution'
    check = refiner.verify(frame, det)
    if check['score'] < .5 or check['mirror'] > .25 or check['agreement'] > .3:
        return None, 'open_loop_disagreement'
    contrast = np.array(eye_contrast(frame, det))
    if max(contrast) < .2:
        return None, 'no_eye_evidence'
    if abs(geometry.yaw) < .2 and min(contrast) < .15:
        return None, 'front_eye_evidence'
    region = np.asarray(det.extra['region'], np.float64)
    angle = det.extra['upright_angle']
    image, matrix = refiner.crop(frame, region, angle, precise=True)
    mirrored = refiner._look(np.ascontiguousarray(image[:, ::-1]), precise=True)
    if mirrored is None:
        return None, 'mirror_missing'
    mirrored_points = np.asarray(mirrored.keypoints).copy()
    mirrored_points[:, 0] = image.shape[1] - 1 - mirrored_points[:, 0]
    order = flip_index(det.schema)
    mirrored_points = apply(np.linalg.inv(matrix), mirrored_points[order])
    side = region[2] - region[0]
    jitter = region + np.array([.04, -.03, .04, -.03]) * side
    shifted = refiner.examine(frame, jitter, angle, precise=True)
    if shifted is None:
        return None, 'jitter_missing'
    own = np.asarray(det.keypoints)
    disagreement = np.maximum(np.linalg.norm(own - mirrored_points, axis=1),
                              np.linalg.norm(own - shifted.keypoints, axis=1)) / geometry.iod
    confidence = np.minimum.reduce([np.asarray(det.kp_scores), np.asarray(mirrored.kp_scores)[order],
                                    np.asarray(shifted.kp_scores)])
    stable = (confidence >= .65) & (disagreement <= .18)
    # A far eye with no local image evidence cannot anchor a yaw target.
    stable[:2] &= contrast >= .15
    height, width = frame.shape[:2]
    stable &= (own[:, 0] >= 0) & (own[:, 0] < width) & (own[:, 1] >= 0) & (own[:, 1] < height)
    if not stable[2] or not np.any(stable[:2]):
        return None, 'unstable_nose_or_eyes'
    # Require a usable core; partial profiles may supervise the visible eye/nose
    # but cannot supply a yaw or eye-midpoint target.
    weight = np.where(stable, np.clip(1 - disagreement / .36, .5, 1), 0)
    return {'points':own.tolist(), 'point_weights':weight.tolist(), 'box':region.tolist(),
            'upright_angle':float(angle), 'yaw':float(geometry.yaw), 'bin':pose_bin(geometry.yaw),
            'iod':float(geometry.iod), 'yaw_supported':bool(np.all(stable[:3])),
            'teacher_score':float(det.score), 'verify':check, 'eye_contrast':contrast.tolist(),
            'point_disagreement':disagreement.tolist(), 'point_confidence':confidence.tolist()}, None


def contact_sheet(rows, output, root, limit=48):
    """A deterministic spread of examples, sorted by video and pose for review."""
    if not rows:
        return
    ordered = sorted(rows, key=lambda row: (row['video_id'], row['bin'], row['timestamp_s']))
    picks = np.linspace(0, len(ordered) - 1, min(limit, len(ordered))).round().astype(int)
    tile, columns = 256, 6
    sheet = np.full(((len(picks) + columns - 1) // columns * (tile + 45), columns * tile, 3), 240, np.uint8)
    for j, pick in enumerate(picks):
        row = ordered[pick]
        frame = cv2.imread(str(Path(root) / row['image']))
        matrix = crop_matrix(row['box'], row['upright_angle'], .7, tile)
        crop = warp_crop(frame, matrix, tile)
        points = apply(matrix, row['points'])
        for k, (point, weight) in enumerate(zip(points, row['point_weights'])):
            p = tuple(np.rint(point).astype(int))
            cv2.circle(crop, p, 3, (0, 230, 0) if weight else (0, 0, 255), -1)
            cv2.putText(crop, str(k), (p[0]+4, p[1]), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 0), 1)
        x, y = j % columns * tile, j // columns * (tile + 45)
        sheet[y+45:y+45+tile, x:x+tile] = crop
        for line, message in enumerate([f"{row['video_id']} {row['timestamp_s']:.1f}s", f"{row['split']} {row['bin']} yaw={row['yaw']:.2f}"]):
            cv2.putText(sheet, message, (x+3, y+17+18*line), cv2.FONT_HERSHEY_SIMPLEX, .4, (20,20,20), 1)
    cv2.imwrite(str(output), sheet)


def prepare(args):
    manifest, sources = read_manifest(args.manifest, args.root)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'Use a fresh dataset directory: {out}')
    out.mkdir(parents=True)
    (out / 'images').mkdir()
    cv2.setNumThreads(1)
    teacher = YoloPoseDetector(args.teacher, providers=args.provider, threads=args.threads,
                               thread_spinning=False, score_threshold=.35)
    if teacher.size != 320 or teacher.schema.name != 'dogface6':
        raise ValueError('This recipe requires the dogface6 320px teacher')
    refiner = FaceRefiner(teacher, RefineConfig(landmarks='off', antialias='tracker'))
    labels, reports = [], []
    started = time.perf_counter()
    for source in sources:
        cap = cv2.VideoCapture(str(Path(args.root) / source['path']))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'Cannot decode {source["path"]}')
        interval = max(1, int(round(fps / args.sample_fps)))
        index, accepted = 0, 0
        counts = Counter()
        previous = {}
        while cap.grab():
            current = index
            index += 1
            if current % interval or current / fps < source.get('start_s', 0):
                continue
            if current / fps > source.get('end_s', float('inf')):
                break
            if accepted >= args.max_per_video:
                break
            ok, frame = cap.retrieve()
            if not ok:
                counts['decode_failed'] += 1
                continue
            counts['sampled_frames'] += 1
            raw = teacher.detect(frame)
            detections = refiner.refine(frame, raw, current/fps, still=True)
            if not detections:
                counts['no_face'] += 1
                continue
            # One primary, largest face per source frame. Never allow the same
            # pixels to create several differently named training examples.
            det = max(detections, key=lambda d: max(0,d.width)*max(0,d.height))
            row, reason = strict_label(frame, det, refiner)
            if row is None:
                counts[reason] += 1
                continue
            matrix = crop_matrix(row['box'], row['upright_angle'], .5, 96)
            thumb = cv2.cvtColor(warp_crop(frame, matrix, 96), cv2.COLOR_BGR2GRAY).astype(np.float32)
            old = previous.get(row['bin'])
            if old is not None and np.abs(thumb-old[0]).mean() < 2 and abs(row['yaw']-old[1]) < .03:
                counts['near_duplicate'] += 1
                continue
            previous[row['bin']] = thumb, row['yaw']
            key = f"{source['id']}_{current:06d}"
            image = f'images/{key}.jpg'
            if not cv2.imwrite(str(out/image), frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
                raise OSError(f'Cannot save {image}')
            row.update({'id':key, 'video_id':source['id'], 'dog_group':source['dog_group'],
                        'split':source['split'], 'frame_index':current, 'timestamp_s':current/fps,
                        'image':image, 'image_sha256':sha256(out/image), 'source_sha256':source['sha256']})
            labels.append(row)
            counts['accepted'] += 1
            counts[f"accepted_{row['bin']}"] += 1
            counts['yaw_supported'] += int(row['yaw_supported'])
            accepted += 1
        cap.release()
        report = {'id':source['id'], 'split':source['split'], 'decoded_frames':index, **counts}
        reports.append(report)
        print(json.dumps(report), flush=True)
        (out/'labels.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in labels),encoding='utf8')
    result = {'format_version':1, 'label_kind':'detector_pseudo_labels_not_ground_truth',
              'teacher':teacher.describe(), 'teacher_sha256':sha256(args.teacher),
              'source_manifest_sha256':sha256(args.manifest), 'source_manifest':manifest,
              'sample_fps':args.sample_fps, 'max_per_video':args.max_per_video,
              'seconds':time.perf_counter()-started, 'sources':reports,
              'labels_sha256':sha256(out/'labels.jsonl')}
    (out/'dataset.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    for split in ('train','val','test'):
        contact_sheet([r for r in labels if r['split']==split], out/f'audit_{split}.jpg', out)
    print(json.dumps({'out':str(out),'labels':len(labels),'seconds':result['seconds']}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--teacher', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--sample-fps', type=float, default=4)
    parser.add_argument('--max-per-video', type=int, default=240)
    parser.add_argument('--provider', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.sample_fps <= 0 or args.max_per_video <= 0:
        parser.error('Sampling limits must be positive')
    prepare(args)


if __name__ == '__main__':
    main()
