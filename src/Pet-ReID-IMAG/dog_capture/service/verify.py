"""Server-side re-check of a capture package; the client is never trusted.

1. Transport: file present, SHA-256 matches the manifest, decodes, sane size.
2. Capture: the server's own engine re-detects the face, re-checks species,
   view band and quality on the uploaded pixels (``CaptureEngine.verify_shot``).
3. Recognition readiness, with the deployed recognition model itself:
   - consistency: every shot must look like the same dog as the others
     (median cosine to the other shots), the session-level analogue of a face
     service's "same person across frames" check;
   - stability: the embedding of the shot vs. a slightly cropped copy (an
     SER-FIQ-style robustness probe): unstable embeddings mean the model
     cannot read this image reliably.
   Both are always reported; they reject only when the policy sets a
   threshold (calibrate with ``tools/calibrate_capture_consistency.py``).
4. Coverage: every required view keeps at least one accepted shot.
"""
from __future__ import annotations

import hashlib
import time

import cv2
import numpy as np

from .protocol import UPLOAD, ProtocolError


def decode_jpeg(data):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ProtocolError('bad_image', 'image does not decode')
    return image


def stability_crop(image, fraction=0.04):
    h, w = image.shape[:2]
    dy, dx = int(round(h * fraction)), int(round(w * fraction))
    return image[dy:h - dy, dx:w - dx]


def encode_jpeg(image, quality=95):
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return data.tobytes()


class PackageVerifier:
    def __init__(self, engine_factory, embedder=None, consistency_threshold=None, stability_threshold=None):
        """``engine_factory(config_dict)`` -> CaptureEngine with detector (+ gate);
        ``embedder(jpeg_bytes)`` -> L2-normalised recognition embedding, or None."""
        self.engine_factory = engine_factory
        self.embedder = embedder
        self.consistency_threshold = consistency_threshold
        self.stability_threshold = stability_threshold

    def verify(self, session, shots, files):
        started = time.perf_counter()
        engine = self.engine_factory(session.config)
        results, accepted = [], []
        for shot in shots:
            entry = {'id': shot['id'], 'view': shot['view'], 'accepted': False, 'reasons': []}
            data = files.get(shot['file'])
            if data is None:
                entry['reasons'].append('missing_file')
            elif len(data) > UPLOAD['max_bytes_per_image']:
                entry['reasons'].append('file_too_large')
            elif hashlib.sha256(data).hexdigest() != shot['sha256'].lower():
                entry['reasons'].append('hash_mismatch')
            else:
                try:
                    image = decode_jpeg(data)
                except ProtocolError:
                    entry['reasons'].append('bad_image')
                    image = None
                if image is not None:
                    h, w = image.shape[:2]
                    entry['size'] = [w, h]
                    if max(h, w) > UPLOAD['max_long_side'] or min(h, w) < 64:
                        entry['reasons'].append('image_size_out_of_range')
                    else:
                        claimed = (shot.get('detection') or {}).get('box')
                        check = engine.verify_shot(image, claimed, shot['view'])
                        entry['server'] = {k: check.get(k) for k in ('view', 'box_iou', 'faces')}
                        entry['server']['quality'] = (check.get('quality') or {}).get('score')
                        entry['server']['level'] = (check.get('quality') or {}).get('level')
                        entry['reasons'] += check['reasons']
                        if check['ok']:
                            accepted.append((shot, data, image, check))
            results.append(entry)
        by_id = {r['id']: r for r in results}
        recognition = self._recognition(accepted, by_id)
        accepted = [item for item in accepted if not by_id[item[0]['id']]['reasons']]
        for shot, _, _, _ in accepted:
            by_id[shot['id']]['accepted'] = True
        required = [v['name'] for v in session.policy['views'] if v.get('required', True)]
        covered = {shot['view'] for shot, _, _, _ in accepted}
        missing = [v for v in required if v not in covered]
        status = 'accepted' if not missing and accepted else 'retake'
        retake = sorted({r['view'] for r in results if not r['accepted']} | set(missing))
        return {'status': status, 'shots': results, 'missing_views': missing,
                'retake_views': retake if status != 'accepted' else [],
                'recognition': recognition, 'elapsed_ms': round((time.perf_counter() - started) * 1000, 1),
                'accepted': [(shot, data) for shot, data, _, _ in accepted]}

    def _recognition(self, accepted, by_id):
        if self.embedder is None or not accepted:
            return {'available': False, 'reason': 'no recognition model attached' if self.embedder is None else
                    'no shot passed the capture checks'}
        vectors, report = [], {'available': True, 'per_shot': {},
                               'consistency_threshold': self.consistency_threshold,
                               'stability_threshold': self.stability_threshold}
        for shot, data, image, _ in accepted:
            vector = self.embedder(data)
            crop = self.embedder(encode_jpeg(stability_crop(image)))
            vectors.append(vector)
            stability = float(np.dot(vector, crop))
            report['per_shot'][shot['id']] = {'stability': round(stability, 4)}
            if self.stability_threshold is not None and stability < self.stability_threshold:
                by_id[shot['id']]['reasons'].append('recognition_unstable')
        if len(vectors) >= 2:
            matrix = np.stack(vectors) @ np.stack(vectors).T
            report['pairwise_min'] = round(float(matrix[np.triu_indices(len(vectors), 1)].min()), 4)
            for i, (shot, _, _, _) in enumerate(accepted):
                others = np.delete(matrix[i], i)
                median = float(np.median(others))
                report['per_shot'][shot['id']]['consistency'] = round(median, 4)
                if self.consistency_threshold is not None and median < self.consistency_threshold:
                    by_id[shot['id']]['reasons'].append('identity_inconsistent')
        return report
