/** Operator guidance codes and default Chinese messages (port of guidance.py). */
export const MESSAGES: Record<string, string> = {
  no_face: '未检测到狗脸，请把镜头对准狗狗的脸',
  not_dog: '检测到的脸不像狗，请只拍摄狗狗',
  dog_far: '看到狗狗了，但脸太小或没有对着镜头，请靠近并让它面向镜头',
  face_too_small: '狗脸太小，请靠近一些',
  face_clipped: '狗脸不完整，请让整张脸进入画面',
  too_dark: '光线太暗，请换到明亮的地方',
  overexposed: '光线过强或逆光，请调整角度',
  low_contrast: '画面发灰，请调整光线',
  blurred: '画面模糊，请保持手机稳定并对焦',
  moving_fast: '狗狗动得太快，请等它稳定一下',
  eyes_or_nose_occluded: '眼睛或鼻子被遮挡，请让狗狗面向镜头',
  pose_unresolved: '看不清脸部朝向，请让狗狗面向镜头',
  hold_still: '很好，请保持这个角度',
  capture_view: '请引导狗狗把头转向：{view}',
  done: '所需角度已全部采集完成',
};
export const VIEW_LABELS: Record<string, string> = { front: '正对镜头', left: '画面左侧', right: '画面右侧' };
const QUALITY_CODES = ['face_too_small', 'face_clipped', 'too_dark', 'overexposed', 'low_contrast', 'blurred',
  'eyes_or_nose_occluded'];
const POSE_REASONS = new Set(['eyes_unresolved', 'implausible_eye_span', 'implausible_nose_position', 'extreme_pose',
  'inconsistent_landmarks', 'eyes_coincide', 'alignment_failed']);

export interface Guidance { code: string; message: string; view: string | null }

export interface GuidanceFace {
  quality: { reasons: string[] } | null;
  viewState: { view: string | null; stable: boolean } | null;
  geometry: { valid: boolean } | null;
}

export function guidance(face: GuidanceFace | null, missing: string[], opts: {
  rejectedSpecies?: boolean; speedIod?: number; motionRef?: number; dogSeen?: boolean;
} = {}): Guidance {
  const result = (code: string, view: string | null = null): Guidance => ({
    code, view, message: view ? MESSAGES[code].replace('{view}', VIEW_LABELS[view] ?? view) : MESSAGES[code],
  });
  if (!missing.length) return result('done');
  if (!face) return result(opts.dogSeen ? 'dog_far' : opts.rejectedSpecies ? 'not_dog' : 'no_face');
  const reasons = new Set(face.quality?.reasons ?? []);
  for (const code of QUALITY_CODES) if (reasons.has(code)) return result(code);
  if ([...reasons].some((r) => POSE_REASONS.has(r)) || (face.geometry && !face.geometry.valid)) return result('pose_unresolved');
  if ((opts.speedIod ?? 0) > (opts.motionRef ?? 3)) return result('moving_fast');
  const state = face.viewState;
  if (state && state.view && missing.includes(state.view) && !state.stable) return result('hold_still', state.view);
  return result('capture_view', missing[0]);
}
