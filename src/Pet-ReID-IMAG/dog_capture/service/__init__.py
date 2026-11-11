"""Capture protocol service (``dog-capture/1``): sessions, server re-check, tokens, commit.

Mount onto the existing Pet ReID FastAPI app with ``mount_capture(app, service)``
(``tools/serve_pet_api.py --enable-capture``).  See docs/DOG_CAPTURE_PROTOCOL_CN.md.
"""
from .protocol import POLICIES, PROTOCOL, ProtocolError
from .sessions import SessionStore, TokenSigner
from .verify import PackageVerifier

try:  # the HTTP layer needs FastAPI; protocol/verification do not
    from .router import build_capture_router, mount_capture, recognition_embedder
except ImportError:  # pragma: no cover
    build_capture_router = mount_capture = recognition_embedder = None

__all__ = ['POLICIES', 'PROTOCOL', 'ProtocolError', 'build_capture_router', 'mount_capture',
           'recognition_embedder', 'SessionStore', 'TokenSigner', 'PackageVerifier']
