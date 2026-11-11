"""Capture protocol service: sessions, server re-check, tokens, commit (no model files)."""
import gzip
import hashlib
import json

import cv2
import numpy as np
import pytest

fastapi = pytest.importorskip('fastapi')
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from dog_capture.engine import CaptureConfig, CaptureEngine  # noqa: E402
from dog_capture.service import PROTOCOL, TokenSigner, build_capture_router  # noqa: E402
from dog_engine_synthetic import render  # noqa: E402


class QueueDetector:
    """Returns the detection queued for each uploaded shot, in upload order."""

    def __init__(self):
        self.queue = []

    def detect(self, frame):
        return [self.queue.pop(0)] if self.queue else []

    def describe(self):
        return {'kind': 'queue'}


class FakeGallery:
    def __init__(self):
        self.calls = []

    def enroll(self, pet_id, uploads, display_name=None):
        self.calls.append(('enroll', pet_id, [u.filename for u in uploads]))
        return {'pet_id': pet_id, 'added': len(uploads)}

    def identify(self, payload, top_k=5):
        self.calls.append(('identify', payload.filename))
        return {'matches': [{'pet_id': 'dog-1', 'score': .9}]}


def color_embedder(data):
    """Same synthetic dog -> same vector; a tinted image -> a different 'dog'."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).astype(np.float32)
    vector = image.reshape(-1, 3).mean(0) - 120.0  # colour signature, centred
    return vector / np.linalg.norm(vector)


def shot_image(yaw, tint=None):
    image, dets = render(size=(480, 640), center=(320, 240), iod=110, yaw=yaw)
    if tint is not None:
        image = np.clip(image.astype(np.int16) + tint, 0, 255).astype(np.uint8)
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return data.tobytes(), dets[0]


@pytest.fixture
def app_parts():
    detector = QueueDetector()
    gallery = FakeGallery()
    base = CaptureConfig().merged({'runtime': {'still_images': True, 'rotate': '0'}})

    def factory(config_dict):
        return CaptureEngine(CaptureConfig.from_dict(config_dict), detector, None)

    router, store = build_capture_router(base_config=base, engine_factory=factory, model_files={},
                                         gallery=gallery, embedder=color_embedder, api_key='k',
                                         signer=TokenSigner('test-secret'), consistency_threshold=0.9)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), detector, gallery, store


def start(client, purpose='enroll', **extra):
    response = client.post('/v1/capture/sessions', json={'purpose': purpose, 'pet_id': 'dog-1', **extra},
                           headers={'x-api-key': 'k'})
    assert response.status_code == 200, response.text
    return response.json()


def package(session, shots, detector, tamper=None):
    manifest = {'protocol': PROTOCOL, 'session_id': session['session_id'], 'nonce': session['nonce'],
                'shots': [], 'signals': {'single_track': True}}
    files = []
    detector.queue = []  # shots rejected before detection must not shift later ones
    for i, (view, yaw, tint) in enumerate(shots):
        data, det = shot_image(yaw, tint)
        if tamper != f'{view}-{i}.jpg':
            detector.queue.append(det)
        name = f'{view}-{i}.jpg'
        manifest['shots'].append({'id': f'{view}-{i}', 'view': view, 'file': name,
                                  'sha256': hashlib.sha256(data).hexdigest(),
                                  'detection': det.to_dict(), 'quality': {'score': .8}})
        files.append(('files', (name, data if tamper != name else data[:-10] + b'0123456789', 'image/jpeg')))
    return {'manifest': json.dumps(manifest)}, files


FULL = [('front', 0.0, None), ('front', 0.02, None), ('left', -0.45, None), ('right', 0.45, None)]


def test_full_enrollment_flow_with_token_and_commit(app_parts):
    client, detector, gallery, _ = app_parts
    session = start(client)
    config = client.get(f'/v1/capture/sessions/{session["session_id"]}/config',
                        params={'nonce': session['nonce']}).json()
    assert config['policy']['name'] == 'enroll_full' and config['policy']['view_order'][0] == 'front'
    assert [v['shots'] for v in config['policy']['views']] == [2, 1, 1]
    assert config['engine']['pose']['views'][0]['shots'] == 2 and not config['client_commit']
    data, files = package(session, FULL, detector)
    result = client.post(f'/v1/capture/sessions/{session["session_id"]}/package', data=data, files=files).json()
    assert result['status'] == 'accepted', result
    assert 'capture_token' not in result  # the H5 never sees the token in keyed mode
    assert result['recognition']['available'] and result['recognition']['pairwise_min'] > .99
    status = client.get(f'/v1/capture/sessions/{session["session_id"]}', headers={'x-api-key': 'k'}).json()
    token = status['capture_token']
    assert status['state'] == 'accepted' and status['signals_recorded']
    committed = client.post(f'/v1/capture/sessions/{session["session_id"]}/commit', json={'capture_token': token},
                            headers={'x-api-key': 'k'})
    assert committed.status_code == 200 and committed.json()['action'] == 'enroll'
    assert gallery.calls == [('enroll', 'dog-1', [f'{session["session_id"]}-{s}.jpg'
                                                  for s in ('front-0', 'front-1', 'left-2', 'right-3')])]
    again = client.post(f'/v1/capture/sessions/{session["session_id"]}/commit', json={'capture_token': token},
                        headers={'x-api-key': 'k'})
    assert again.status_code == 409 and again.json()['code'] == 'token_used'


def test_missing_required_view_asks_for_retake(app_parts):
    client, detector, _, _ = app_parts
    session = start(client)
    data, files = package(session, FULL[:3], detector)
    result = client.post(f'/v1/capture/sessions/{session["session_id"]}/package', data=data, files=files).json()
    assert result['status'] == 'retake' and result['missing_views'] == ['right']
    data, files = package(session, FULL, detector)  # a second attempt in the same session
    assert client.post(f'/v1/capture/sessions/{session["session_id"]}/package',
                       data=data, files=files).json()['status'] == 'accepted'


def test_identify_accepts_front_only_and_commits_best_front(app_parts):
    client, detector, gallery, _ = app_parts
    session = start(client, purpose='identify', client_commit=True)
    data, files = package(session, FULL[:2], detector)
    result = client.post(f'/v1/capture/sessions/{session["session_id"]}/package', data=data, files=files).json()
    assert result['status'] == 'accepted' and result['capture_token']
    committed = client.post(f'/v1/capture/sessions/{session["session_id"]}/commit',
                            json={'capture_token': result['capture_token']}).json()
    assert committed['action'] == 'identify' and committed['query_shot'].startswith('front')
    assert gallery.calls[0][0] == 'identify'


def test_server_rejects_tampered_wrong_view_and_other_dog(app_parts):
    client, detector, _, _ = app_parts
    session = start(client)
    data, files = package(session, FULL, detector, tamper='left-2.jpg')
    result = client.post(f'/v1/capture/sessions/{session["session_id"]}/package', data=data, files=files).json()
    reasons = {s['id']: s['reasons'] for s in result['shots']}
    assert result['status'] == 'retake' and reasons['left-2'] == ['hash_mismatch']
    wrong = [('front', 0.0, None), ('front', 0.0, None), ('left', 0.45, None), ('right', 0.45, None)]
    data, files = package(session, wrong, detector)  # client claims "left" for a right turn
    result = client.post(f'/v1/capture/sessions/{session["session_id"]}/package', data=data, files=files).json()
    assert 'view_mismatch' in {s['id']: s['reasons'] for s in result['shots']}['left-2']
    other = FULL[:3] + [('right', 0.45, (0, 0, 90))]  # a different "dog" for one view
    data, files = package(session, other, detector)
    result = client.post(f'/v1/capture/sessions/{session["session_id"]}/package', data=data, files=files).json()
    assert 'identity_inconsistent' in {s['id']: s['reasons'] for s in result['shots']}['right-3']
    assert result['retake_views'] == ['right']


def test_protocol_guards(app_parts):
    client, detector, _, store = app_parts
    assert client.post('/v1/capture/sessions', json={'purpose': 'enroll', 'pet_id': 'x'}).status_code == 401
    assert client.post('/v1/capture/sessions', json={'purpose': 'enroll'},
                       headers={'x-api-key': 'k'}).json()['code'] == 'missing_pet_id'
    session = start(client)
    bad = client.get(f'/v1/capture/sessions/{session["session_id"]}/config', params={'nonce': 'nope'})
    assert bad.status_code == 403
    data, files = package(session, FULL, detector)
    manifest = json.loads(data['manifest'])
    manifest['nonce'] = 'forged'
    response = client.post(f'/v1/capture/sessions/{session["session_id"]}/package',
                           data={'manifest': json.dumps(manifest)}, files=files)
    assert response.status_code == 403
    commit = client.post(f'/v1/capture/sessions/{session["session_id"]}/commit',
                         json={'capture_token': 'x.y'}, headers={'x-api-key': 'k'})
    assert commit.json()['code'] == 'bad_token'
    store.get(session['session_id']).attempts = 99
    data, files = package(session, FULL, detector)
    assert client.post(f'/v1/capture/sessions/{session["session_id"]}/package',
                       data=data, files=files).status_code == 429


def test_tokens_are_signed_and_expire():
    signer = TokenSigner('s')
    token = signer.sign({'sid': 'a', 'exp': 9e9})
    assert signer.verify(token)['sid'] == 'a'
    with pytest.raises(Exception):
        TokenSigner('other').verify(token)
    with pytest.raises(Exception):
        signer.verify(signer.sign({'sid': 'a', 'exp': 1}))


def test_h5_static_serves_precompressed_assets(tmp_path):
    from dog_capture.service.router import _static_files
    (tmp_path / 'ort').mkdir()
    raw = b'\0asm' + bytes(4096)
    (tmp_path / 'ort' / 'rt.wasm').write_bytes(raw)
    (tmp_path / 'ort' / 'rt.wasm.br').write_bytes(b'br-bytes')
    (tmp_path / 'ort' / 'rt.wasm.gz').write_bytes(gzip.compress(raw))
    (tmp_path / 'index.html').write_text('<!doctype html>', encoding='utf8')
    app = FastAPI()
    app.mount('/capture', _static_files(tmp_path), name='h5')
    client = TestClient(app)
    br = client.get('/capture/ort/rt.wasm', headers={'accept-encoding': 'gzip, br'})
    assert br.headers['content-encoding'] == 'br' and br.headers['content-type'] == 'application/wasm'
    assert br.headers['vary'] == 'Accept-Encoding'
    gz = client.get('/capture/ort/rt.wasm', headers={'accept-encoding': 'gzip'})
    assert gz.headers['content-encoding'] == 'gzip' and gz.content == raw  # client decodes back to the original
    plain = client.get('/capture/ort/rt.wasm', headers={'accept-encoding': 'identity'})
    assert 'content-encoding' not in plain.headers and plain.content == raw
    assert client.get('/capture/', headers={'accept-encoding': 'br'}).text == '<!doctype html>'
