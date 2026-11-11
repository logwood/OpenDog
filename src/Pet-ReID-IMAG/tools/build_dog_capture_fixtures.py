"""Explicitly edited fixtures: workflow order, frozen head, blur and empty scene.

These are engineering checks, not independent natural-video accuracy evidence.
"""
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.cli import ROOT
from dog_capture.models import file_hash


def main():
    target = ROOT / 'data/dog_capture/fixtures'
    target.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/dog_capture/public/wolfhound_panting.ogv'
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 24)
    ok, front = cap.read()
    if not ok:
        raise RuntimeError('Cannot decode source fixture frame')
    height, width = front.shape[:2]
    intervals = [(24, 60, 'front'), (240, 280, 'left'), (576, 612, 'front'),
                 (133, 158, 'right'), (624, 660, 'front')]
    examples = {
        'ordered_turn_fixture.mp4': None,
        'frozen_front_fixture.mp4': [front] * 72,
        'blurred_front_fixture.mp4': [cv2.GaussianBlur(front, (0, 0), 20)] * 72,
        'empty_scene_fixture.mp4': [np.full_like(front, 90)] * 72,
    }
    manifest = {}
    for name, sequence in examples.items():
        path = target / name
        if path.exists():
            raise FileExistsError(path)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError('Fixture encoder failed')
        if sequence is None:
            for start, end, _ in intervals:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for _ in range(start, end):
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError('Unexpected end of fixture source')
                    writer.write(frame)
        else:
            for frame in sequence:
                writer.write(frame)
        writer.release()
        manifest[name] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': file_hash(path),
                          'source': 'wolfhound_panting.ogv' if not name.startswith('empty_') else 'generated uniform gray',
                          'author': 'project-generated' if name.startswith('empty_') else 'Sage Ross; engineering transformations by this project',
                          'license': 'CC BY-SA 3.0' if not name.startswith('empty_') else 'project-generated',
                          'source_url': None if name.startswith('empty_') else 'https://commons.wikimedia.org/wiki/File:Irish_Wolfhound_mix,_panting.ogv',
                          'license_url': None if name.startswith('empty_') else 'https://creativecommons.org/licenses/by-sa/3.0/',
                          'changes': ('Temporal montage; source frame intervals [start,end), labels: ' + str(intervals)) if name.startswith('ordered') else ('Repeated frame 24, Gaussian sigma 20' if name.startswith('blurred') else 'Repeated frame 24' if name.startswith('frozen') else 'Uniform gray frames'),
                          'expected_sequence_complete': name.startswith('ordered'),
                          'expected_primary_image': name.startswith(('ordered', 'frozen')),
                          'scope': 'Constructed regression fixture, not a natural capture or liveness demonstration'}
    (ROOT / 'assets/dog_capture/fixtures.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    cap.release()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
