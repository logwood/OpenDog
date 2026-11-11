"""FastAPI router for the capture protocol, mounted onto the existing Pet ReID app.

``mount_capture(app, service, ...)`` adds ``/v1/capture/*`` and serves the Vue
H5 build at ``/capture/`` (same origin, so the browser needs no CORS and the
phone can use the same HTTPS address).  Existing routes are untouched.
"""
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
import threading
import time
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .protocol import (DEFAULT_POLICY, PROTOCOL, UPLOAD, ProtocolError, check_manifest, engine_config, policy,
                       view_order)
from .sessions import SessionStore, TokenSigner
from .verify import PackageVerifier

REPO_ROOT = Path(__file__).resolve().parents[4]
H5_DIST = REPO_ROOT / 'src' / 'Pet-ReID-IMAG' / 'frontend' / 'dog-capture' / 'packages' / 'h5' / 'dist'


def recognition_embedder(service):
    """Embed JPEG bytes with the deployed recognizer exactly as uploads are encoded."""
    lock = getattr(service, '_inference_lock', None) or threading.Lock()
    staging = getattr(getattr(service, 'store', None), 'staging_root', None)

    def embed(data):
        with tempfile.TemporaryDirectory(prefix='capture-', dir=staging) as directory:
            path = Path(directory) / 'shot.jpg'
            path.write_bytes(data)
            with lock:
                encoded = service.encoder.encode_file(path)
        vector = np.asarray(encoded.fused, np.float32).ravel()
        return vector / max(float(np.linalg.norm(vector)), 1e-12)
    return embed


def _sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def build_capture_router(*, base_config, engine_factory, model_files, gallery=None, embedder=None, api_key=None,
                         store=None, signer=None, consistency_threshold=None, stability_threshold=None,
                         allow_client_commit=None):
    """``model_files``: {'face': Path, 'gate': Path|None} served to clients;
    ``gallery``: object with enroll(pet_id, uploads, display_name=) / identify(payload, top_k=)."""
    store = store or SessionStore()
    signer = signer or TokenSigner()
    verifier = PackageVerifier(engine_factory, embedder, consistency_threshold, stability_threshold)
    client_commit_default = (api_key is None) if allow_client_commit is None else bool(allow_client_commit)
    served = {}
    for role, path in model_files.items():
        if path is not None and Path(path).is_file():
            sidecar = Path(path).with_suffix('.json')
            meta = json.loads(sidecar.read_text(encoding='utf8')) if sidecar.is_file() else {}
            served[Path(path).name] = {'role': role, 'path': Path(path), 'sha256': _sha256(path),
                                       'bytes': Path(path).stat().st_size, 'meta': meta}
    router = APIRouter(prefix='/v1/capture', tags=['capture'])

    def require_key(x_api_key: Annotated[str | None, Header()] = None):
        if api_key is not None and not (x_api_key and secrets.compare_digest(x_api_key, api_key)):
            raise HTTPException(status_code=401, detail={'code': 'unauthorized', 'message': 'missing or bad API key'})

    def fail(error):
        return JSONResponse(status_code=error.status, content={'code': error.code, 'message': str(error)})

    @router.post('/sessions', dependencies=[Depends(require_key)])
    def create_session(body: dict):
        try:
            purpose = body.get('purpose', 'enroll')
            if purpose not in DEFAULT_POLICY:
                raise ProtocolError('bad_purpose', 'purpose must be enroll or identify')
            if purpose == 'enroll' and not body.get('pet_id'):
                raise ProtocolError('missing_pet_id', 'enroll sessions need pet_id')
            name = body.get('policy') or DEFAULT_POLICY[purpose]
            chosen = policy(name)
            config = engine_config(base_config, chosen).to_dict()
            session = store.create(purpose=purpose, policy_name=name, policy=chosen, view_order=view_order(chosen),
                                   config=config, pet_id=body.get('pet_id'), display_name=body.get('display_name'),
                                   client_commit=bool(body.get('client_commit', client_commit_default)))
        except ProtocolError as error:
            return fail(error)
        return {'protocol': PROTOCOL, 'session_id': session.id, 'nonce': session.nonce,
                'expires_at': session.expires_at,
                'h5_path': f'/capture/?session={session.id}&nonce={session.nonce}',
                'config_path': f'/v1/capture/sessions/{session.id}/config?nonce={session.nonce}'}

    @router.get('/sessions/{session_id}/config')
    def session_config(session_id: str, nonce: str):
        try:
            session = store.get(session_id, nonce)
        except ProtocolError as error:
            return fail(error)
        if session.state == 'created':
            session.state = 'capturing'
        models = {item['role']: {'url': f'/v1/capture/models/{name}', 'sha256': item['sha256'],
                                 'bytes': item['bytes'], **{k: item['meta'][k] for k in
                                                            ('kind', 'schema', 'input_size', 'kpt_shape')
                                                            if k in item['meta']}}
                  for name, item in served.items()}
        return {'protocol': PROTOCOL, 'session_id': session.id, 'purpose': session.purpose,
                'policy': {'name': session.policy_name, **session.policy, 'view_order': session.view_order},
                'engine': session.config, 'models': models, 'upload': UPLOAD, 'expires_at': session.expires_at,
                'client_commit': session.client_commit, 'state': session.state}

    @router.get('/models/{name}')
    def model_file(name: str):
        item = served.get(name)
        if item is None:
            raise HTTPException(status_code=404, detail={'code': 'model_not_found', 'message': name})
        return FileResponse(item['path'], media_type='application/octet-stream',
                            headers={'Cache-Control': 'public, max-age=86400, immutable', 'ETag': item['sha256']})

    @router.post('/sessions/{session_id}/package')
    async def upload_package(session_id: str,
                             manifest: Annotated[str, Form()],
                             files: Annotated[list[UploadFile], File()]):
        try:
            data = json.loads(manifest)
        except ValueError:
            return fail(ProtocolError('bad_manifest', 'manifest is not JSON'))
        try:
            session = store.get(session_id, data.get('nonce') if isinstance(data, dict) else None)
            if session.state in ('accepted', 'committed'):
                raise ProtocolError('already_accepted', 'this session already has an accepted package', 409)
            session.attempts += 1
            if session.attempts > session.policy.get('max_attempts', 5):
                session.state = 'expired'
                raise ProtocolError('too_many_attempts', 'too many capture attempts; start a new session', 429)
            uploads = {}
            for upload in files[:UPLOAD['max_shots']]:
                uploads[upload.filename] = await upload.read(UPLOAD['max_bytes_per_image'] + 1)
            shots = check_manifest(data, session, set(uploads))
        except ProtocolError as error:
            return fail(error)
        from starlette.concurrency import run_in_threadpool
        result = await run_in_threadpool(verifier.verify, session, shots, uploads)
        accepted = result.pop('accepted')
        session.signals = data.get('signals')
        session.result = result
        response = {'protocol': PROTOCOL, 'session_id': session.id, **result}
        if result['status'] == 'accepted':
            session.state, session.accepted = 'accepted', accepted
            exp = time.time() + 600
            session.token = signer.sign({'sid': session.id, 'purpose': session.purpose, 'pet': session.pet_id,
                                         'shots': [[s['id'], s['sha256']] for s, _ in accepted], 'exp': exp})
            response['token_expires_at'] = exp
            if session.client_commit:
                response['capture_token'] = session.token
        else:
            session.state = 'retake'
        return response

    @router.get('/sessions/{session_id}', dependencies=[Depends(require_key)])
    def session_status(session_id: str):
        try:
            session = store.get(session_id)
        except ProtocolError as error:
            return fail(error)
        return {**session.public(), 'result': session.result, 'capture_token': session.token,
                'signals_recorded': session.signals is not None, 'commit': session.commit_result}

    @router.post('/sessions/{session_id}/commit')
    async def commit(session_id: str, body: dict, x_api_key: Annotated[str | None, Header()] = None):
        try:
            session = store.get(session_id)
            keyed = api_key is None or (x_api_key and secrets.compare_digest(x_api_key, api_key))
            if not keyed and not session.client_commit:
                raise ProtocolError('unauthorized', 'commit needs the API key for this session', 401)
            payload = signer.verify(str(body.get('capture_token', '')))
            if payload.get('sid') != session.id:
                raise ProtocolError('bad_token', 'token does not belong to this session', 403)
            if session.token_used:
                raise ProtocolError('token_used', 'capture token was already committed', 409)
            if session.state != 'accepted':
                raise ProtocolError('bad_token', 'session has no accepted package', 403)
            if gallery is None:
                raise ProtocolError('no_gallery', 'no recognition service is attached to this server', 503)
            session.token_used = True
        except ProtocolError as error:
            return fail(error)
        from starlette.concurrency import run_in_threadpool
        try:
            result = await run_in_threadpool(_commit, gallery, session, body)
        except Exception as error:  # surface gallery errors with their own message
            session.token_used = False
            status = getattr(error, 'status_code', 400)
            return JSONResponse(status_code=status, content={'code': getattr(error, 'code', 'commit_failed'),
                                                              'message': str(error)})
        session.state, session.commit_result = 'committed', result
        return result

    return router, store


class _Upload:
    """Duck-typed stand-in for pet_id.gallery_service.UploadPayload (filename, content_type, data)."""

    def __init__(self, filename, content_type, data):
        self.filename, self.content_type, self.data = filename, content_type, data


def _upload_type():
    try:
        from pet_id.gallery_service import UploadPayload  # the existing upload type
        return UploadPayload
    except Exception:  # recognition stack not importable (tests, other hosts)
        return _Upload


def _commit(gallery, session, body):
    UploadPayload = _upload_type()
    shots = [(shot, data) for shot, data in session.accepted]
    payloads = [UploadPayload(filename=f'{session.id}-{shot["id"]}.jpg', content_type='image/jpeg', data=data)
                for shot, data in shots]
    if session.purpose == 'enroll':
        pet_id = session.pet_id or body.get('pet_id')
        result = gallery.enroll(pet_id, payloads, display_name=session.display_name or body.get('display_name'))
        return {'action': 'enroll', 'pet_id': pet_id, 'images': [s['id'] for s, _ in shots], 'gallery': result}
    # Identification queries with the best frontal shot (one history entry per session).
    order = {name: i for i, name in enumerate(['front', 'left', 'right'])}
    best = min(range(len(shots)), key=lambda i: (order.get(shots[i][0]['view'], 9),
                                                 -(shots[i][0].get('quality') or {}).get('score', 0)))
    result = gallery.identify(payloads[best], top_k=int(body.get('top_k', 5)))
    return {'action': 'identify', 'query_shot': shots[best][0]['id'], 'gallery': result}


def mount_capture(app, service=None, *, api_key=None, models_dir=None, config=None, **kwargs):
    """Attach ``/v1/capture`` and the ``/capture/`` H5 to an existing FastAPI app."""
    from fastapi.staticfiles import StaticFiles
    from ..engine import CaptureConfig
    from ..engine.zoo import build_body_gate, build_detector, find_face_model

    if models_dir is not None:
        os.environ['DOG_CAPTURE_MODELS'] = str(models_dir)
    base = config or CaptureConfig().merged({'runtime': {'still_images': True, 'rotate': '0'}})
    kind, path = find_face_model(base)
    detector = build_detector(base, kind, path)
    gate = build_body_gate(base, kind)
    from ..engine import CaptureEngine

    def engine_factory(config_dict):
        return CaptureEngine(CaptureConfig.from_dict(config_dict).merged(
            {'runtime': {'still_images': True, 'rotate': '0'}}), detector, gate)

    staging = getattr(getattr(service, 'store', None), 'staging_root', None)
    signer = TokenSigner(secret_file=Path(staging) / 'capture_secret.txt' if staging else None)
    router, store = build_capture_router(
        base_config=base, engine_factory=engine_factory,
        model_files={'face': path if kind != 'dogflw' else None, 'gate': getattr(gate, 'model_path', None)},
        gallery=service, embedder=recognition_embedder(service) if service is not None else None,
        api_key=api_key, signer=signer, **kwargs)
    app.include_router(router)
    if H5_DIST.is_dir():
        app.mount('/capture', StaticFiles(directory=str(H5_DIST), html=True), name='dog-capture-h5')
    app.state.capture_store = store
    return store
