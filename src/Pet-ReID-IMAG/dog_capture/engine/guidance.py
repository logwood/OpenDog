"""Operator guidance, as in face-enrolment UIs ("move closer", "hold still").

Codes are stable identifiers for apps to localise; the Chinese message is a
default.  Only the highest-priority problem is reported each frame.
"""
from __future__ import annotations

MESSAGES = {
    'no_face': '未检测到狗脸，请把镜头对准狗狗的脸',
    'not_dog': '检测到的脸不像狗，请只拍摄狗狗',
    'dog_far': '看到狗狗了，但脸太小或没有对着镜头，请靠近并让它面向镜头',
    'face_too_small': '狗脸太小，请靠近一些',
    'face_clipped': '狗脸不完整，请让整张脸进入画面',
    'too_dark': '光线太暗，请换到明亮的地方',
    'overexposed': '光线过强或逆光，请调整角度',
    'low_contrast': '画面发灰，请调整光线',
    'blurred': '画面模糊，请保持手机稳定并对焦',
    'moving_fast': '狗狗动得太快，请等它稳定一下',
    'eyes_or_nose_occluded': '眼睛或鼻子被遮挡，请让狗狗面向镜头',
    'pose_unresolved': '看不清脸部朝向，请让狗狗面向镜头',
    'hold_still': '很好，请保持这个角度',
    'capture_view': '请引导狗狗把头转向：{view}',
    'done': '所需角度已全部采集完成',
}
VIEW_LABELS = {'front': '正对镜头', 'left': '画面左侧', 'right': '画面右侧'}
QUALITY_CODES = ('face_too_small', 'face_clipped', 'too_dark', 'overexposed', 'low_contrast',
                 'blurred', 'eyes_or_nose_occluded')
POSE_REASONS = {'eyes_unresolved', 'implausible_eye_span', 'implausible_nose_position',
                'extreme_pose', 'inconsistent_landmarks', 'eyes_coincide', 'alignment_failed'}


def guidance(face, missing_views, *, rejected_species=False, speed_iod=0.0, motion_ref=3.0, dog_seen=False):
    """``face`` is the primary FaceResult (or None); returns {code, message, view}."""
    def result(code, view=None):
        message = MESSAGES[code]
        if view:
            message = message.format(view=VIEW_LABELS.get(view, view))
        return {'code': code, 'message': message, 'view': view}

    if not missing_views:
        return result('done')
    if face is None:
        if dog_seen:
            return result('dog_far')
        return result('not_dog' if rejected_species else 'no_face')
    reasons = set(face.quality.reasons) if face.quality else set()
    for code in QUALITY_CODES:
        if code in reasons:
            return result(code)
    if reasons & POSE_REASONS or (face.geometry is not None and not face.geometry.valid):
        return result('pose_unresolved')
    if speed_iod > motion_ref:
        return result('moving_fast')
    state = face.view_state
    if state is not None and state.view in missing_views:
        # Already in a view still being captured or improved: keep it there.
        return result('hold_still', state.view)
    return result('capture_view', missing_views[0])
