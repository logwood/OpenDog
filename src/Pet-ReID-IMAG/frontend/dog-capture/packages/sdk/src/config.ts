/**
 * CaptureConfig — same structure and defaults as dog_capture.engine.config (Python).
 * In production the client receives the complete configuration from the
 * session (`engine` field of the session config), so both sides use identical
 * thresholds; the defaults here only serve tests and offline use.
 */
export interface ViewSpec {
  name: string; center: number; lower: number; upper: number;
  max_roll_deg: number; shots: number; required: boolean;
}

export interface CaptureConfig {
  detector: {
    kind: string; model: string; input_size: number; score_threshold: number; nms_iou: number;
    max_faces: number; schema: string; roi_zoom: boolean; roi_min_zoom: number; roi_period_s: number;
  };
  refine: {
    enabled: boolean; landmarks: string; share: number; crop_size: number; antialias: 'off' | 'tracker' | 'all'; candidate_retry_s: number; search_period_s: number; confident_score: number; max_search_period_s: number;
    switch_margin: number; fine_min_deg: number;
    research_below: number; carry_frames: number; match_iou: number; full_detect_period_s: number;
    min_shot_score: number; max_mirror_disagreement: number; min_front_eye_conf: number; min_side_eye_conf: number;
    max_agreement: number; max_side_mirror_disagreement: number; max_side_agreement: number;
    min_eye_contrast: number; gate_window_s: number; gate_min_frames: number; gate_min_ratio: number;
    gate_min_passes: number; full_detect_min_frames: number; search_min_frames: number; backoff_margin: number;
    max_backoff: number; empty_search_period_s: number; empty_search_min_frames: number;
  };
  body_gate: {
    enabled: string; model: string; input_size: number; score_threshold: number; class_id: number;
    period_s: number; hold_s: number; margin: number; head_score_threshold: number;
  };
  tracker: {
    high_score: number; low_score: number; new_track_score: number; match_iou: number; low_match_iou: number;
    unconfirmed_match_iou: number; max_lost_s: number; max_tracks: number; nominal_fps: number;
  };
  pose: {
    views: ViewSpec[]; exit_margin: number; min_dwell_s: number; min_dwell_obs: number; max_gap_s: number;
    smooth_min_cutoff: number; smooth_beta: number; snout_factor: number; min_iod_box_ratio: number;
    max_iod_box_ratio: number; pitch_range: [number, number]; max_abs_yaw: number;
  };
  quality: {
    min_iod_px: number; good_iod_px: number; min_sharpness: number; min_brightness: number; max_brightness: number;
    max_clipped: number; min_contrast: number; border_margin: number; min_kp_score: number;
    max_alignment_residual: number; motion_ref_iod_per_s: number; weights: Record<string, number>;
    high_level: number; medium_level: number;
  };
  selection: {
    mode: 'best' | 'fast' | 'hybrid'; good_enough: number; improve_window_s: number; top_k: number;
    min_gap_s: number; min_level: 'low' | 'medium' | 'high';
  };
  output: { aligned_size: number; template: string; face_margin: number; keep_frames: boolean; image_format: string };
  runtime: {
    providers: string; threads: number; cache_dir: string; timestamps: 'drop' | 'raise'; rotate: string;
    mirror: boolean; still_images: boolean;
    /** native ONNX Runtime options (Python engine); carried in the session config, unused in the browser */
    cuda_graph: boolean; thread_spinning: boolean;
  };
}

const view = (name: string, center: number, lower: number, upper: number): ViewSpec =>
  ({ name, center, lower, upper, max_roll_deg: 30, shots: 0, required: true });

export function defaultConfig(): CaptureConfig {
  return {
    detector: { kind: 'auto', model: '', input_size: 0, score_threshold: 0.35, nms_iou: 0.45, max_faces: 8,
      schema: '', roi_zoom: true, roi_min_zoom: 1.5, roi_period_s: 0.1 },
    refine: { enabled: true, landmarks: '', share: 0.35, crop_size: 0, antialias: 'tracker', candidate_retry_s: 0.25, search_period_s: 0.5, confident_score: 1.0, max_search_period_s: 2.0,
      switch_margin: 0.05, fine_min_deg: 8.0, research_below: 0.4, carry_frames: 3, match_iou: 0.3,
      full_detect_period_s: 0.3, min_shot_score: 0.5, max_mirror_disagreement: 0.2, min_front_eye_conf: 0.7,
      min_side_eye_conf: 0.5, max_agreement: 0.3, max_side_mirror_disagreement: 0.3, max_side_agreement: 0.5,
      min_eye_contrast: 0.2, gate_window_s: 0.5, gate_min_frames: 8, gate_min_ratio: 0.35, gate_min_passes: 3,
      full_detect_min_frames: 3, search_min_frames: 6, backoff_margin: 0.1, max_backoff: 4,
      empty_search_period_s: 0.25, empty_search_min_frames: 2 },
    body_gate: { enabled: 'auto', model: '', input_size: 416, score_threshold: 0.3, class_id: 16, period_s: 0.25,
      hold_s: 0.75, margin: 0.15, head_score_threshold: 0.5 },
    tracker: { high_score: 0.5, low_score: 0.2, new_track_score: 0.55, match_iou: 0.2, low_match_iou: 0.5,
      unconfirmed_match_iou: 0.3, max_lost_s: 1.0, max_tracks: 16, nominal_fps: 30.0 },
    pose: { views: [view('front', 0, -0.13, 0.13), view('left', -0.45, -0.75, -0.25), view('right', 0.45, 0.25, 0.75)],
      exit_margin: 0.05, min_dwell_s: 0.12, min_dwell_obs: 3, max_gap_s: 0.3, smooth_min_cutoff: 1.5,
      smooth_beta: 0.5, snout_factor: 0.8, min_iod_box_ratio: 0.12, max_iod_box_ratio: 1.1,
      pitch_range: [0.05, 2.2], max_abs_yaw: 1.2 },
    quality: { min_iod_px: 18, good_iod_px: 64, min_sharpness: 0.15, min_brightness: 30, max_brightness: 230,
      max_clipped: 0.3, min_contrast: 16, border_margin: 0.005, min_kp_score: 0.35, max_alignment_residual: 0.45,
      motion_ref_iod_per_s: 3.0,
      weights: { sharpness: 3, eye_detail: 2, exposure: 1, resolution: 1.5, confidence: 1, motion: 1, pose_fit: 0.75,
        alignment: 0.5 },
      high_level: 0.7, medium_level: 0.5 },
    selection: { mode: 'hybrid', good_enough: 0.65, improve_window_s: 2.0, top_k: 1, min_gap_s: 0.3, min_level: 'medium' },
    output: { aligned_size: 224, template: 'petface_dog', face_margin: 0.2, keep_frames: true, image_format: 'png' },
    runtime: { providers: 'auto', threads: 0, cache_dir: '', timestamps: 'drop', rotate: 'auto', mirror: false,
      still_images: false, cuda_graph: false, thread_spinning: true },
  };
}

type Deep<T> = { [K in keyof T]?: T[K] extends object ? (T[K] extends unknown[] ? T[K] : Deep<T[K]>) : T[K] };

/** Deep-merge a partial configuration (e.g. the session's `engine` dict) over the defaults. */
export function mergeConfig(base: CaptureConfig, override: Deep<CaptureConfig> | Record<string, unknown>): CaptureConfig {
  const out = structuredClone(base) as unknown as Record<string, Record<string, unknown>>;
  for (const [section, values] of Object.entries(override ?? {})) {
    if (!(section in out) || typeof values !== 'object' || values === null) continue;
    for (const [key, value] of Object.entries(values as Record<string, unknown>)) {
      if (key === 'views' && Array.isArray(value)) {
        out[section][key] = value.map((v) => ({ max_roll_deg: 30, shots: 0, required: true, ...(v as object) }));
      } else if (key === 'weights' && typeof value === 'object') {
        out[section][key] = { ...(out[section][key] as object), ...(value as object) };
      } else {
        out[section][key] = value;
      }
    }
  }
  const config = out as unknown as CaptureConfig;
  config.runtime.rotate = String(config.runtime.rotate);
  validateConfig(config);
  return config;
}

export function validateConfig(c: CaptureConfig): void {
  const check = (ok: boolean, message: string) => { if (!ok) throw new Error(`Invalid capture config: ${message}`); };
  const names = c.pose.views.map((v) => v.name);
  check(names.length > 0 && new Set(names).size === names.length, 'views need unique names');
  for (const v of c.pose.views) check(v.lower < v.center && v.center < v.upper && v.max_roll_deg > 0, `view band ${v.name}`);
  const ordered = [...c.pose.views].sort((a, b) => a.lower - b.lower);
  for (let i = 1; i < ordered.length; i++) check(ordered[i - 1].upper < ordered[i].lower, 'view bands must not overlap');
  check(['best', 'fast', 'hybrid'].includes(c.selection.mode), 'selection.mode');
  const crop = c.refine.crop_size;
  check(crop === 0 || (crop >= 96 && crop <= 1024 && crop % 32 === 0), 'refine.crop_size must be 0 or a multiple of 32 in 96..1024');
}
