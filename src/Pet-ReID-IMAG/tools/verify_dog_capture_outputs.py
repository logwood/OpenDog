"""Verify archived input hashes and exact original pixels in selected outputs."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.cli import ROOT
from dog_capture.models import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results')
    args = parser.parse_args()
    directory = Path(args.results).resolve()
    sources = {}
    for name in ('sources.json', 'fixtures.json'):
        sources.update(json.loads((ROOT / 'assets/dog_capture' / name).read_text(encoding='utf8')))
    by_hash = {asset['sha256']: ROOT / asset['path'] for asset in sources.values()}
    checks = []
    for result in directory.rglob('result.json'):
        report = json.loads(result.read_text(encoding='utf8'))
        source = by_hash.get(report['input_sha256'], Path(report['input']))
        assert file_hash(source) == report['input_sha256'], source
        chosen = report.get('selected', {})
        indices = {item['frame_index'] for item in chosen.values()}
        frame_images = {}
        cap = cv2.VideoCapture(str(source))
        for index in range(max(indices, default=-1) + 1):
            ok, frame = cap.read()
            assert ok, (source, index)
            if index in indices:
                frame_images[index] = frame
        cap.release()
        for state, item in chosen.items():
            original = frame_images[item['frame_index']]
            if report.get('parameters', {}).get('mirror'):
                original = cv2.flip(original, 1)
            saved = cv2.imdecode(np.fromfile(result.parent / item['frame_file'], np.uint8), cv2.IMREAD_COLOR)
            assert np.array_equal(original, saved), (result, state, 'original pixels differ')
            if 'box_xyxy' in item:
                x1, y1, x2, y2 = item['box_xyxy']
                crop = np.rot90(original[y1:y2, x1:x2], item.get('rotation_quarters', 0))
                saved_crop = cv2.imdecode(np.fromfile(result.parent / item['face_file'], np.uint8), cv2.IMREAD_COLOR)
                assert np.array_equal(crop, saved_crop), (result, state, 'crop pixels differ')
        checks.append({'result': str(result.relative_to(directory)), 'sha256_verified': True,
                       'original_frames_verified': len(chosen), 'passed': True})
    if not checks:
        raise ValueError('No result.json files found')
    (directory / 'pixel_checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
