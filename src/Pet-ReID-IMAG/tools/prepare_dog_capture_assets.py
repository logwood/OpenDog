"""Fetch pinned public examples/models, or archive a new user video unchanged."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.cli import ROOT
from dog_capture.models import ensure_models, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--public', action='store_true')
    parser.add_argument('--models', action='store_true')
    parser.add_argument('--user-video')
    parser.add_argument('--benchmark-inputs', action='store_true', help='Recreate fixed model benchmark crops from archived videos')
    args = parser.parse_args()
    manifest_path = ROOT / 'assets/dog_capture/sources.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if args.models:
        ensure_models(ROOT / 'models/pretrained/dog_face_landmarks_v1', download=True)
    if args.public:
        for name, asset in manifest.items():
            if not asset.get('download_url'):
                continue
            path = ROOT / asset['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temp = path.with_suffix('.download')
                request = urllib.request.Request(asset['download_url'], headers={'User-Agent': 'DogCaptureResearch/0.1'})
                with urllib.request.urlopen(request, timeout=60) as response, temp.open('wb') as f:
                    shutil.copyfileobj(response, f)
                if file_hash(temp) != asset['sha256']:
                    raise ValueError(f'Checksum mismatch: {name}')
                temp.replace(path)
            if file_hash(path) != asset['sha256']:
                raise ValueError(f'Checksum mismatch: {name}')
            print(f'OK: {name}', flush=True)
    if args.user_video:
        source = Path(args.user_video).resolve(strict=True)
        digest = file_hash(source)
        target = ROOT / 'data/dog_capture/user' / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_hash(target) != digest:
            raise FileExistsError(f'Different video already archived as {target.name}; use a distinct source name')
        if source != target and not target.exists():
            shutil.copy2(source, target)
        if file_hash(target) != digest:
            raise ValueError('Archived video differs from original')
        manifest[source.name] = {'path': target.relative_to(ROOT).as_posix(), 'original_path': str(source),
                                 'sha256': digest, 'bytes': target.stat().st_size, 'source': 'user-provided',
                                 'license': 'private user asset; no public redistribution permission supplied',
                                 'modified': False}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
        print(f'Archived unchanged: {target}')
    if args.benchmark_inputs:
        import cv2
        import numpy as np
        from dog_capture.cli import save_image
        crop_root = ROOT / 'assets/dog_capture/benchmark_inputs'
        crops = json.loads((crop_root / 'sources.json').read_text(encoding='utf8'))
        for name, entry in crops.items():
            destination = crop_root / (name + '.png')
            if destination.exists():
                if file_hash(destination) != entry['sha256']:
                    raise ValueError(f'Benchmark input checksum mismatch: {destination}')
                continue
            video = ROOT / entry['video_path']
            if file_hash(video) != entry['video_sha256']:
                raise ValueError(f'Benchmark source checksum mismatch: {video}')
            capture = cv2.VideoCapture(str(video))
            try:
                for index in range(entry['frame_index'] + 1):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f'Cannot read benchmark frame: {video}, {index}')
            finally:
                capture.release()
            x1, y1, x2, y2 = entry['box_xyxy']
            save_image(destination, np.rot90(frame[y1:y2, x1:x2], entry['rotation_quarters']))
            if file_hash(destination) != entry['sha256']:
                raise ValueError(f'Recreated benchmark differs; check decoder/OpenCV version: {destination}')
            print(f'OK: {destination}', flush=True)


if __name__ == '__main__':
    main()
