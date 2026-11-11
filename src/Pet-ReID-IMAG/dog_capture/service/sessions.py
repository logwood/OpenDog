"""Capture sessions and signed one-time capture tokens (in-memory, bounded, thread-safe)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import threading
import time
from dataclasses import dataclass, field

from .protocol import ProtocolError


@dataclass
class CaptureSession:
    id: str
    nonce: str
    purpose: str                 # enroll | identify
    policy_name: str
    policy: dict
    view_order: list
    config: dict                 # full CaptureConfig dict sent to the client
    pet_id: str | None = None
    display_name: str | None = None
    client_commit: bool = False  # may the H5 client commit itself (dev / LAN mode)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    state: str = 'created'       # created -> capturing -> retake / accepted -> committed | expired
    attempts: int = 0
    result: dict | None = None
    accepted: list = field(default_factory=list)  # [(shot dict, jpeg bytes)]
    token: str | None = None
    token_used: bool = False
    commit_result: dict | None = None
    signals: dict | None = None

    def public(self):
        return {'session_id': self.id, 'purpose': self.purpose, 'policy': self.policy_name, 'state': self.state,
                'attempts': self.attempts, 'expires_at': self.expires_at, 'pet_id': self.pet_id,
                'view_order': self.view_order}


class SessionStore:
    def __init__(self, ttl_s=900, max_sessions=256):
        self.ttl, self.max_sessions = float(ttl_s), int(max_sessions)
        self.sessions: dict[str, CaptureSession] = {}
        self.lock = threading.Lock()

    def create(self, **kwargs):
        with self.lock:
            self._expire()
            if len(self.sessions) >= self.max_sessions:
                oldest = min(self.sessions.values(), key=lambda s: s.created_at)
                del self.sessions[oldest.id]
            session = CaptureSession(id='cap_' + secrets.token_urlsafe(12), nonce=secrets.token_urlsafe(16),
                                     **kwargs)
            session.expires_at = session.created_at + self.ttl
            self.sessions[session.id] = session
            return session

    def get(self, session_id, nonce=None):
        with self.lock:
            self._expire()
            session = self.sessions.get(session_id)
        if session is None:
            raise ProtocolError('session_not_found', 'unknown or expired capture session', 404)
        if nonce is not None and not hmac.compare_digest(str(nonce), session.nonce):
            raise ProtocolError('session_mismatch', 'wrong session nonce', 403)
        return session

    def _expire(self):
        now = time.time()
        for key in [k for k, s in self.sessions.items() if s.expires_at < now]:
            del self.sessions[key]


class TokenSigner:
    """HMAC-SHA256 capture tokens: base64url(payload).base64url(signature).

    The secret comes from ``DOG_CAPTURE_SECRET`` or a file created next to the
    gallery staging data, so tokens survive a service restart.
    """

    def __init__(self, secret=None, secret_file=None):
        if secret is None:
            secret = os.environ.get('DOG_CAPTURE_SECRET')
        if secret is None and secret_file is not None:
            path = Path(secret_file)
            if path.is_file():
                secret = path.read_text(encoding='utf8').strip()
            else:
                secret = secrets.token_urlsafe(32)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(secret, encoding='utf8')
        self.key = (secret or secrets.token_urlsafe(32)).encode()

    @staticmethod
    def _b64(data):
        return base64.urlsafe_b64encode(data).rstrip(b'=').decode()

    @staticmethod
    def _unb64(text):
        return base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))

    def sign(self, payload):
        body = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
        signature = hmac.new(self.key, body, hashlib.sha256).digest()
        return self._b64(body) + '.' + self._b64(signature)

    def verify(self, token):
        try:
            body_text, signature_text = token.split('.')
            body = self._unb64(body_text)
            expected = hmac.new(self.key, body, hashlib.sha256).digest()
            if not hmac.compare_digest(expected, self._unb64(signature_text)):
                raise ValueError('signature')
            payload = json.loads(body)
        except (ValueError, TypeError) as exc:
            raise ProtocolError('bad_token', 'capture token is invalid', 403) from exc
        if payload.get('exp', 0) < time.time():
            raise ProtocolError('token_expired', 'capture token has expired', 403)
        return payload
