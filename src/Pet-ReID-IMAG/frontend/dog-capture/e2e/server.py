"""E2E server: the real capture protocol + real capture models, a stand-in recognition service.

    DOG_CAPTURE_MODELS=<dir with dog_face_pose.onnx, yolox_nano.onnx> python e2e/server.py --port 8765

The stand-in encoder embeds a 16x16 grey thumbnail of the face region, so the
recognition-readiness plumbing (embedding, consistency, stability, commit to
enroll/identify) is exercised end to end without the production Re-ID model.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from fastapi import FastAPI  # noqa: E402
from dog_capture.service import POLICIES, mount_capture  # noqa: E402
from dog_capture.service.protocol import _LEFT, _RIGHT  # noqa: E402
from dog_capture.engine import CaptureConfig  # noqa: E402

# test-only policy: both sides required (mirror pair of one real dog)
POLICIES['sides_only'] = {'views': [{**_LEFT, 'shots': 1, 'required': True}, {**_RIGHT, 'shots': 1, 'required': True}],
                          'selection': {'mode': 'hybrid', 'good_enough': 0.6, 'improve_window_s': 1.0,
                                        'min_gap_s': 0.4, 'min_level': 'medium'},
                          'max_duration_s': 30.0, 'max_attempts': 5, 'randomise_side_order': False}


class Encoded:
    def __init__(self, fused):
        self.fused = fused


class StandInService:
    def __init__(self):
        self.store = type('Store', (), {'staging_root': Path(tempfile.mkdtemp(prefix='e2e-'))})()
        self.encoder = self
        self.calls = []

    def encode_file(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        thumb = cv2.resize(image[h // 4: 3 * h // 4, w // 4: 3 * w // 4], (16, 16), interpolation=cv2.INTER_AREA)
        vector = thumb.astype(np.float32).ravel() - thumb.mean()
        return Encoded(vector / max(np.linalg.norm(vector), 1e-6))

    def enroll(self, pet_id, uploads, display_name=None):
        self.calls.append(['enroll', pet_id, len(uploads)])
        return {'pet_id': pet_id, 'added': len(uploads), 'stand_in': True}

    def identify(self, payload, top_k=5):
        self.calls.append(['identify', payload.filename])
        return {'matches': [], 'stand_in': True, 'query_bytes': len(payload.data)}


def real_service(model, gallery_path, seed):
    """Opt-in integration with the existing deployed recognizer, in a NEW test gallery."""
    from pet_id.gallery_service import (MultimodalPipelineEncoder, PetGalleryStore,
                                         PetIdentificationService, UploadPayload)
    from pet_id.unified_runtime import UnifiedONNXRuntimePipeline
    gallery_path = Path(gallery_path).resolve()
    if gallery_path.exists():
        raise ValueError('E2E_GALLERY must be a new directory; production galleries are never reused')
    pipeline = UnifiedONNXRuntimePipeline(model, provider='cpu', device='cpu', warmup_batches=[1])
    encoder = MultimodalPipelineEncoder(pipeline)
    delegate = PetIdentificationService(PetGalleryStore(gallery_path), encoder,
                                         model_fingerprint=encoder.backend_info()['model_sha256'])
    data = Path(seed).read_bytes()
    delegate.enroll('e2e-dog', [UploadPayload(filename=Path(seed).name, content_type='image/png', data=data)])

    class RecordingService:
        store = delegate.store
        encoder = delegate.encoder
        _inference_lock = delegate._inference_lock

        def __init__(self):
            self.calls = []

        def enroll(self, pet_id, uploads, display_name=None):
            value = delegate.enroll(pet_id, uploads, display_name=display_name)
            self.calls.append(['enroll', pet_id, len(uploads)])
            return value

        def identify(self, payload, top_k=5):
            value = delegate.identify(payload, top_k=top_k)
            self.calls.append(['identify', payload.filename])
            return value

    return RecordingService()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    model = os.environ.get('E2E_REID_MODEL')
    service = real_service(model, os.environ['E2E_GALLERY'], os.environ['E2E_SEED']) if model else StandInService()
    app = FastAPI()
    config_path = os.environ.get('E2E_CAPTURE_CONFIG')
    config = CaptureConfig.load(config_path) if config_path else CaptureConfig().merged(
        {'runtime': {'providers': 'cpu', 'threads': 4}, 'body_gate': {'enabled': 'on'}})
    mount_capture(app, service, config=config)

    @app.get('/e2e/calls')
    def calls():
        return service.calls

    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
