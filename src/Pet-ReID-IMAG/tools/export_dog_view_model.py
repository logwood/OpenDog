"""Export the compact-confidence graph for a later native/mobile consumer."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.cli import ROOT
from dog_capture.landmarks import confidence_model
from dog_capture.models import ASSETS, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default=str(ROOT / 'models/pretrained/dog_face_landmarks_v1/dog_face_landmarks_full.tflite'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base, destination = Path(args.base), Path(args.output)
    if file_hash(base) != ASSETS['dog_face_landmarks_full.tflite'][1]:
        raise ValueError('Expected pinned upstream landmark weights')
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(confidence_model(base, compact=True))
    report = {'source_sha256': file_hash(base), 'sha256': file_hash(destination),
              'bytes': destination.stat().st_size, 'license': 'CC BY-NC 4.0 (inherits upstream weights)',
              'input': {'shape': [1, 384, 384, 3], 'dtype': 'float32', 'normalization': 'RGB / 255'},
              'outputs': [{'name': 'landmarks', 'shape': [1, 92], 'reshape': [46, 2], 'coordinates': 'normalized crop x,y'},
                          {'name': 'landmark_entropy', 'shape': [1, 46], 'epsilon': 1e-6}],
              'changes': 'Append MAXIMUM, LOG, MUL, SUM, NEG to reduce entropy; original coordinate graph and weights unchanged.',
              'validation': 'CPU parity in tests/test_dog_views.py; GPU parity and timing in tools/benchmark_dog_views.py. Mobile delegate not yet measured.'}
    destination.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
