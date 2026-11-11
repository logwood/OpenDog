"""One configuration object for the whole engine, loadable from JSON or YAML.

Every threshold lives here with a documented default instead of being spread
through the code.  Unknown keys are rejected with the list of valid keys, so a
typo in a config file fails loudly instead of being silently ignored.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ViewSpec:
    """A capture target on the dimensionless yaw cue (nose offset / eye span).

    Negative = nose toward the image left (after upright alignment).  The cue
    is a 2-D landmark ratio, not a calibrated angle; ``approx_deg`` in the
    results converts it with a simple snout model for display only.
    """
    name: str
    center: float
    lower: float
    upper: float
    max_roll_deg: float = 30.0


DEFAULT_VIEWS = (ViewSpec('front', 0.0, -0.13, 0.13),
                 ViewSpec('left', -0.45, -0.75, -0.25),
                 ViewSpec('right', 0.45, 0.25, 0.75))


@dataclass
class DetectorConfig:
    kind: str = 'auto'            # auto | yolo_pose | anyface | scrfd | dogflw | external
    model: str = ''               # path; '' = look up the model directory
    input_size: int = 640         # square network input for ONNX detectors
    score_threshold: float = 0.35
    nms_iou: float = 0.45
    max_faces: int = 8
    schema: str = ''              # keypoint schema override for yolo_pose models


@dataclass
class BodyGateConfig:
    """Dog-body detector used as a species gate (face detectors also find people/cats)."""
    enabled: str = 'auto'         # auto (use if the model exists) | on | off
    model: str = ''               # YOLOX ONNX (COCO); '' = look up the model directory
    input_size: int = 416
    score_threshold: float = 0.30
    class_id: int = 16            # COCO "dog"
    period_s: float = 0.25        # body detection refresh period
    hold_s: float = 0.75          # a body box stays valid this long
    margin: float = 0.15          # body box expansion before the face-centre test
    head_score_threshold: float = 0.50  # stricter: a head crop alone is weaker evidence (cats)


@dataclass
class TrackerConfig:
    """ByteTrack association thresholds (score-split two-stage IoU matching)."""
    high_score: float = 0.50
    low_score: float = 0.20
    new_track_score: float = 0.55
    match_iou: float = 0.20
    low_match_iou: float = 0.50
    unconfirmed_match_iou: float = 0.30
    max_lost_s: float = 1.0
    max_tracks: int = 16
    nominal_fps: float = 30.0     # Kalman time unit; real timestamps scale it


@dataclass
class PoseConfig:
    views: tuple = DEFAULT_VIEWS
    exit_margin: float = 0.05     # hysteresis: leave a view only beyond its band + margin
    min_dwell_s: float = 0.12     # a view must persist this long ...
    min_dwell_obs: int = 3        # ... and over this many observations
    max_gap_s: float = 0.30       # longer gaps restart the dwell
    smooth_min_cutoff: float = 1.5  # One-Euro filter on the yaw cue (Hz)
    smooth_beta: float = 0.5
    snout_factor: float = 0.8     # display-only cue -> approximate degrees
    min_iod_box_ratio: float = 0.12
    max_iod_box_ratio: float = 1.10
    pitch_range: tuple = (0.10, 1.60)
    max_abs_yaw: float = 1.20


@dataclass
class QualityConfig:
    """Face-image-quality factors.  Hard limits reject; soft factors rank."""
    min_iod_px: float = 18.0      # inter-ocular distance in source pixels
    good_iod_px: float = 64.0
    min_sharpness: float = 0.15   # normalised, see quality.py (keeps ~97% of real sharp faces)
    min_brightness: float = 30.0
    max_brightness: float = 230.0
    max_clipped: float = 0.30
    min_contrast: float = 16.0
    border_margin: float = 0.005  # core keypoints must be inside the frame by this fraction
    min_kp_score: float = 0.35    # only for detectors that output keypoint confidence
    max_alignment_residual: float = 0.45
    motion_ref_iod_per_s: float = 3.0
    weights: dict = field(default_factory=lambda: {
        'sharpness': 3.0, 'eye_detail': 2.0, 'exposure': 1.0, 'resolution': 1.5,
        'confidence': 1.0, 'motion': 1.0, 'pose_fit': 0.75, 'alignment': 0.5})
    high_level: float = 0.70
    medium_level: float = 0.50


@dataclass
class SelectionConfig:
    """Per-track best-shot policy, after face-capture camera modes."""
    mode: str = 'hybrid'          # best | fast | hybrid
    good_enough: float = 0.65     # fast/hybrid: first accepted score that triggers capture
    improve_window_s: float = 2.0  # hybrid: keep improving this long after the first capture
    top_k: int = 1                # images kept per view
    min_gap_s: float = 0.30       # top_k > 1: diversity in time
    min_level: str = 'medium'     # lowest quality level that may be selected


@dataclass
class OutputConfig:
    aligned_size: int = 224
    template: str = 'petface_dog'
    face_margin: float = 0.20     # unaligned crop padding (fraction of box)
    keep_frames: bool = True      # keep the original full frame of every selected shot
    image_format: str = 'png'


@dataclass
class RuntimeConfig:
    providers: str = 'auto'       # auto | cpu | cuda | tensorrt | coreml | directml | openvino | ...
    threads: int = 0              # 0 = ONNX Runtime default
    cache_dir: str = ''           # TensorRT/CoreML compiled-model cache
    timestamps: str = 'drop'      # non-increasing timestamps: drop the frame | raise
    rotate: str = 'auto'          # auto | 0 | 90 | 180 | 270 (clockwise, applied before detection)
    mirror: bool = False          # selfie camera: flip before detection
    still_images: bool = False    # unrelated photos: no temporal confirmation


@dataclass
class CaptureConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    body_gate: BodyGateConfig = field(default_factory=BodyGateConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # ---- construction -------------------------------------------------
    @classmethod
    def from_dict(cls, value: dict | None):
        config = _build(cls, value or {}, 'config')
        config.validate()
        return config

    @classmethod
    def load(cls, path):
        path = Path(path)
        text = path.read_text(encoding='utf8')
        if path.suffix.lower() in ('.yaml', '.yml'):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError('YAML config needs PyYAML; use a .json config or `pip install pyyaml`') from exc
            value = yaml.safe_load(text) or {}
        else:
            value = json.loads(text)
        return cls.from_dict(value)

    def merged(self, overrides: dict):
        """Deep-merge a partial dict (e.g. CLI flags) into a copy."""
        base = self.to_dict()
        _deep_merge(base, overrides)
        return CaptureConfig.from_dict(base)

    def to_dict(self):
        value = asdict(self)
        value['pose']['views'] = [asdict(v) for v in self.pose.views]
        value['pose']['pitch_range'] = list(self.pose.pitch_range)
        return value

    # ---- validation ---------------------------------------------------
    def validate(self):
        def check(condition, message):
            if not condition:
                raise ValueError(f'Invalid capture config: {message}')

        d, t, p, q, s, o, r = (self.detector, self.tracker, self.pose, self.quality,
                               self.selection, self.output, self.runtime)
        check(d.kind in ('auto', 'yolo_pose', 'anyface', 'scrfd', 'dogflw', 'external'),
              f'detector.kind={d.kind!r}')
        check(32 <= d.input_size <= 2048 and d.input_size % 32 == 0, 'detector.input_size must be a multiple of 32')
        check(0 < d.score_threshold < 1 and 0 < d.nms_iou < 1 and d.max_faces >= 1, 'detector thresholds')
        check(self.body_gate.enabled in ('auto', 'on', 'off'), 'body_gate.enabled must be auto/on/off')
        check(0 < self.body_gate.score_threshold <= self.body_gate.head_score_threshold < 1,
              'body_gate thresholds (head_score_threshold >= score_threshold)')
        check(0 <= t.low_score <= t.high_score <= 1 and 0 < t.new_track_score <= 1, 'tracker score thresholds')
        check(all(0 < v < 1 for v in (t.match_iou, t.low_match_iou, t.unconfirmed_match_iou)), 'tracker IoU thresholds')
        check(t.max_lost_s > 0 and t.max_tracks >= 1 and t.nominal_fps > 0, 'tracker limits')
        names = [v.name for v in p.views]
        check(names and len(set(names)) == len(names), 'views need unique names')
        for v in p.views:
            check(v.name.isidentifier(), f'view name {v.name!r} must be an identifier')
            check(all(math.isfinite(x) for x in (v.lower, v.center, v.upper, v.max_roll_deg))
                  and v.lower < v.center < v.upper and v.max_roll_deg > 0, f'view band {v.name}')
        ordered = sorted(p.views, key=lambda v: v.lower)
        check(all(a.upper < b.lower for a, b in zip(ordered, ordered[1:])), 'view bands must not overlap')
        check(0 <= p.exit_margin < 0.5 and p.min_dwell_s >= 0 and p.min_dwell_obs >= 1
              and p.max_gap_s > 0, 'pose dwell / hysteresis')
        check(p.smooth_min_cutoff > 0 and p.smooth_beta >= 0 and p.snout_factor > 0, 'pose smoothing')
        check(0 < p.min_iod_box_ratio < p.max_iod_box_ratio and len(p.pitch_range) == 2
              and p.pitch_range[0] < p.pitch_range[1], 'pose plausibility ranges')
        check(0 < q.min_iod_px < q.good_iod_px, 'quality iod range')
        check(0 <= q.min_brightness < q.max_brightness <= 255, 'quality brightness range')
        check(all(w >= 0 for w in q.weights.values()) and sum(q.weights.values()) > 0, 'quality weights')
        unknown = set(q.weights) - {'sharpness', 'eye_detail', 'exposure', 'resolution', 'confidence',
                                    'motion', 'pose_fit', 'alignment'}
        check(not unknown, f'unknown quality weights {sorted(unknown)}')
        check(0 < q.medium_level <= q.high_level < 1, 'quality levels')
        check(s.mode in ('best', 'fast', 'hybrid'), 'selection.mode must be best/fast/hybrid')
        check(s.top_k >= 1 and s.improve_window_s >= 0 and s.min_gap_s >= 0, 'selection limits')
        check(s.min_level in ('low', 'medium', 'high'), 'selection.min_level must be low/medium/high')
        check(64 <= o.aligned_size <= 1024 and o.image_format in ('png', 'jpg'), 'output image settings')
        check(r.timestamps in ('drop', 'raise'), 'runtime.timestamps must be drop/raise')
        check(str(r.rotate) in ('auto', '0', '90', '180', '270'), 'runtime.rotate must be auto/0/90/180/270')


def _deep_merge(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict) and key != 'weights':
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _build(cls, value, where):
    if not isinstance(value, dict):
        raise ValueError(f'{where} must be a mapping, got {type(value).__name__}')
    known = {f.name: f for f in fields(cls)}
    unknown = set(value) - set(known)
    if unknown:
        raise ValueError(f'Unknown key(s) {sorted(unknown)} in {where}; valid keys: {sorted(known)}')
    if cls is ViewSpec:
        missing = {'name', 'center', 'lower', 'upper'} - set(value)
        if missing:
            raise ValueError(f'{where} is missing {sorted(missing)}')
        return ViewSpec(**{k: (str(v) if k == 'name' else float(v)) for k, v in value.items()})
    kwargs = {}
    defaults = cls()
    for name, item in value.items():
        current = getattr(defaults, name)
        if is_dataclass(current):
            kwargs[name] = _build(type(current), item, f'{where}.{name}')
        elif name == 'views':
            kwargs[name] = tuple(v if isinstance(v, ViewSpec) else _build(ViewSpec, v, f'{where}.views[]')
                                 for v in item)
        elif name == 'pitch_range':
            kwargs[name] = tuple(float(x) for x in item)
        elif name == 'weights':
            merged = dict(current)
            merged.update(item)
            kwargs[name] = merged
        elif isinstance(current, bool):
            if not isinstance(item, bool):
                raise ValueError(f'{where}.{name} must be true/false')
            kwargs[name] = item
        elif isinstance(current, (int, float)) and not isinstance(current, bool):
            if not isinstance(item, (int, float)) or isinstance(item, bool):
                raise ValueError(f'{where}.{name} must be a number')
            kwargs[name] = type(current)(item) if isinstance(current, float) else int(item)
        elif isinstance(current, str):
            kwargs[name] = str(item)
        else:
            kwargs[name] = item
    return replace(defaults, **kwargs)
