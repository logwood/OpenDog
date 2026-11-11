"""Standalone capture/verification host. Mount onto serve_pet_api.py for enrollment/identification."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from fastapi import FastAPI
    import uvicorn
    from dog_capture.engine import CaptureConfig
    from dog_capture.engine.zoo import REPO_ROOT, model_dir
    from dog_capture.service import mount_capture
    from dog_capture.service.router import H5_DIST

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--models', type=Path, default=model_dir())
    parser.add_argument('--config', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--api-key', default=os.environ.get('PET_REID_API_KEY'))
    args = parser.parse_args()
    if args.host not in ('127.0.0.1', 'localhost', '::1') and not args.api_key:
        parser.error('Remote hosting requires --api-key or PET_REID_API_KEY')
    if not (H5_DIST / 'index.html').is_file():
        parser.error('Build H5 first: scripts/setup-dog-capture-v1.ps1 -Web')
    config = CaptureConfig.load(args.config or REPO_ROOT / 'assets/dog_capture/v1_reference/capture_quality.json')
    config = config.merged({'runtime': {'threads': args.threads}})
    app = FastAPI(title='Dog capture v1')
    mount_capture(app, api_key=args.api_key, models_dir=args.models, config=config, allow_client_commit=False)
    print(f'Capture UI: http://{args.host}:{args.port}/capture/', flush=True)
    print('Verified JPEGs: GET /v1/capture/sessions/{id}/shots/{shot_id} (backend API key).', flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
