/** Operator-facing texts for views, quality levels and server re-check reasons. */
export const VIEW_NAMES: Record<string, string> = { front: '正面', left: '朝左', right: '朝右' };
export const VIEW_HINTS: Record<string, string> = {
  front: '狗狗正对镜头，两只眼睛都看得见',
  left: '狗狗把头转向画面左侧（约 30°）',
  right: '狗狗把头转向画面右侧（约 30°）',
};
export const LEVELS: Record<string, string> = { high: '清晰', medium: '合格', low: '较差', reject: '不合格' };

export const REASONS: Record<string, string> = {
  hash_mismatch: '图片在上传中被改动', missing_file: '图片缺失', bad_image: '图片无法解码',
  image_size_out_of_range: '图片尺寸不合规', file_too_large: '图片过大', no_face: '没有检测到狗脸',
  not_dog: '不像狗脸', claimed_face_not_found: '与采集时的狗脸位置不一致', view_mismatch: '角度不符合要求',
  quality_too_low: '质量不足', face_too_small: '狗脸太小', face_clipped: '狗脸不完整', blurred: '模糊',
  too_dark: '太暗', overexposed: '过曝', low_contrast: '发灰', eyes_or_nose_occluded: '眼睛或鼻子被遮挡',
  alignment_failed: '关键点异常', recognition_unstable: '识别模型无法稳定读取', identity_inconsistent: '和其他照片不像同一只狗',
  eyes_unresolved: '看不清眼睛', extreme_pose: '角度过大', implausible_nose_position: '关键点异常',
  inconsistent_landmarks: '关键点异常', implausible_eye_span: '关键点异常',
};

export const reasonText = (codes: string[]) => [...new Set(codes.map((c) => REASONS[c] ?? c))].join('、');
