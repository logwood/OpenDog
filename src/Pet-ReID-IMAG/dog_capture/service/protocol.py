"""Capture protocol ``dog-capture/1``: policies, session configuration, manifest checks.

The flow mirrors face-authentication services (business backend creates a
session -> H5/SDK captures with the session's configuration -> uploads a
capture package -> server re-checks -> business backend commits the token):

    POST /v1/capture/sessions                 (backend, API key)  -> session_id, nonce
    GET  /v1/capture/sessions/{id}/config     (client, nonce)     -> policy + engine config + models
    POST /v1/capture/sessions/{id}/package    (client, nonce)     -> verification result
    GET  /v1/capture/sessions/{id}            (backend, API key)  -> state, result, capture_token
    POST /v1/capture/sessions/{id}/commit     (backend, token)    -> existing enroll / identify

JSON Schemas for the config, the manifest and the result live next to this
file (``dog_capture/protocol/*.schema.json``) for non-Python clients.
"""
from __future__ import annotations

import copy
import secrets

PROTOCOL = 'dog-capture/1'

# Bands are in reference (DogFLW nose-centre) yaw units; detectors normalise
# their own yaw with the keypoint schema's yaw_scale, so these stay fixed.
_FRONT = {'name': 'front', 'center': 0.0, 'lower': -0.13, 'upper': 0.13, 'max_roll_deg': 30.0}
_LEFT = {'name': 'left', 'center': -0.45, 'lower': -0.75, 'upper': -0.25, 'max_roll_deg': 30.0}
_RIGHT = {'name': 'right', 'center': 0.45, 'lower': 0.25, 'upper': 0.75, 'max_roll_deg': 30.0}

POLICIES = {
    # Enrollment builds the reference set: more views, all required.
    'enroll_full': {
        'views': [{**_FRONT, 'shots': 2, 'required': True},
                  {**_LEFT, 'shots': 1, 'required': True},
                  {**_RIGHT, 'shots': 1, 'required': True}],
        'selection': {'mode': 'hybrid', 'good_enough': 0.65, 'improve_window_s': 1.5, 'min_gap_s': 0.5,
                      'min_level': 'medium'},
        'max_duration_s': 90.0, 'max_attempts': 5, 'randomise_side_order': True,
    },
    # Identification needs a good frontal view fast; sides are a bonus.
    'identify_quick': {
        'views': [{**_FRONT, 'shots': 2, 'required': True},
                  {**_LEFT, 'shots': 1, 'required': False},
                  {**_RIGHT, 'shots': 1, 'required': False}],
        'selection': {'mode': 'hybrid', 'good_enough': 0.65, 'improve_window_s': 1.0, 'min_gap_s': 0.4,
                      'min_level': 'medium'},
        'max_duration_s': 45.0, 'max_attempts': 5, 'randomise_side_order': True,
    },
    'front_only': {
        'views': [{**_FRONT, 'shots': 2, 'required': True}],
        'selection': {'mode': 'hybrid', 'good_enough': 0.65, 'improve_window_s': 1.0, 'min_gap_s': 0.4,
                      'min_level': 'medium'},
        'max_duration_s': 45.0, 'max_attempts': 5, 'randomise_side_order': False,
    },
}
DEFAULT_POLICY = {'enroll': 'enroll_full', 'identify': 'identify_quick'}

UPLOAD = {'image_format': 'jpeg', 'jpeg_quality': 0.95, 'max_long_side': 4096, 'max_bytes_per_image': 8 << 20,
          'max_shots': 8}


class ProtocolError(ValueError):
    """Raised with a stable machine-readable ``code``."""

    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.status = code, status


def policy(name):
    if name not in POLICIES:
        raise ProtocolError('unknown_policy', f'unknown capture policy {name!r}; known: {sorted(POLICIES)}')
    return copy.deepcopy(POLICIES[name])


def engine_config(base_config, chosen):
    """CaptureConfig for this session: base thresholds + the policy's views/selection."""
    return base_config.merged({'pose': {'views': chosen['views']}, 'selection': chosen['selection']})


def view_order(chosen, rng=secrets.SystemRandom()):
    """Front first (natural for operators); side order randomised as a liveness signal."""
    names = [v['name'] for v in chosen['views']]
    sides = [n for n in names if n != 'front']
    if chosen.get('randomise_side_order') and len(sides) > 1:
        rng.shuffle(sides)
    return (['front'] if 'front' in names else []) + sides


def check_manifest(manifest, session, file_names):
    """Structural checks; returns the list of shot entries.  Pixels are checked later."""
    if not isinstance(manifest, dict):
        raise ProtocolError('bad_manifest', 'manifest must be a JSON object')
    if manifest.get('protocol') != PROTOCOL:
        raise ProtocolError('protocol_mismatch', f'expected protocol {PROTOCOL}')
    if manifest.get('session_id') != session.id or manifest.get('nonce') != session.nonce:
        raise ProtocolError('session_mismatch', 'manifest does not belong to this session', 403)
    shots = manifest.get('shots')
    if not isinstance(shots, list) or not shots:
        raise ProtocolError('no_shots', 'manifest has no shots')
    if len(shots) > UPLOAD['max_shots']:
        raise ProtocolError('too_many_shots', f'at most {UPLOAD["max_shots"]} shots per package')
    views = {v['name']: v for v in session.policy['views']}
    seen_ids, per_view = set(), {}
    for shot in shots:
        if not isinstance(shot, dict):
            raise ProtocolError('bad_shot', 'each shot must be an object')
        for key in ('id', 'view', 'file', 'sha256'):
            if not isinstance(shot.get(key), str) or not shot[key]:
                raise ProtocolError('bad_shot', f'shot is missing {key!r}')
        if shot['id'] in seen_ids:
            raise ProtocolError('duplicate_shot', f'duplicate shot id {shot["id"]}')
        seen_ids.add(shot['id'])
        if shot['view'] not in views:
            raise ProtocolError('unknown_view', f'shot {shot["id"]}: view {shot["view"]!r} not in this policy')
        if shot['file'] not in file_names:
            raise ProtocolError('missing_file', f'shot {shot["id"]}: file {shot["file"]!r} was not uploaded')
        per_view[shot['view']] = per_view.get(shot['view'], 0) + 1
    for name, count in per_view.items():
        if count > max(1, views[name].get('shots', 1)):
            raise ProtocolError('too_many_shots', f'view {name}: {count} shots, policy allows {views[name]["shots"]}')
    return shots
