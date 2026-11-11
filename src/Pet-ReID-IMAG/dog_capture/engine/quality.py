"""Face-image quality assessment (FIQA) on the *aligned* crop.

Structure follows mature face SDKs (e.g. SeetaFace QualityAssessor, ISO/IEC
29794-5 quality components): independent, interpretable factors, each mapped
to [0, 1], hard limits that reject, and a single ranking score with a
LOW / MEDIUM / HIGH level.  Because every crop is similarity-aligned to the
same template at the same size, pixel factors are measured at a constant
face scale and on a fixed eyes-nose-mouth region.

Factors
    resolution  inter-ocular distance in *source* pixels (aligned crops of a
                tiny face are upsampled, so detail must come from the source)
    sharpness   Laplacian energy + Crete et al. 2007 no-reference re-blur
                metric, both on the fixed core region
    eye_detail  Laplacian energy around the eyes (the most identity-bearing
                and most often blurred/closed region)
    exposure    brightness, clipping and contrast of the core region
    confidence  detector score, and keypoint visibility where the model has it
    motion      Kalman centre speed in eye-distances per second (motion blur risk)
    pose_fit    closeness to the target view centre, and roll
    alignment   landmark residual after the similarity fit

The ranking score is a weighted geometric mean: one bad factor pulls the score
down instead of being averaged away.  Sharpness reference values were
calibrated on aligned dog faces (see docs/DOG_CAPTURE_ENGINE_CN.md).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import cv2
import numpy as np

from .alignment import align_face, core_mask, eye_patches

# Quality is always measured on a 224 x 224 aligned crop (the PetFace template
# size), independent of the output crop size; constants are calibrated for it.
QUALITY_SIZE = 224
# Calibrated with tools/calibrate_dog_quality.py on 300 real frontal dog faces
# (StanfordExtra val, eye distance >= 40 px) against Gaussian defocus and
# motion blur of 0.03-0.3 eye distances; see docs/DOG_CAPTURE_ENGINE_CN.md.
LAPLACIAN_REF = 80.0        # 1 - exp(-lap / ref); real faces: median lap 70
EYE_LAPLACIAN_REF = 100.0   # real faces: median eye-patch lap 92
CRETE_SHARP = 0.30          # re-blur metric, ~p10 of real sharp faces
CRETE_BLURRY = 0.70         # ~median of 0.15-eye-distance motion blur

LEVELS = ('reject', 'low', 'medium', 'high')


@dataclass
class QualityReport:
    score: float = 0.0
    level: str = 'reject'
    passed: bool = False
    reasons: list = field(default_factory=list)
    factors: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def to_dict(self):
        value = asdict(self)
        value['score'] = round(self.score, 4)
        value['factors'] = {k: round(float(v), 4) for k, v in self.factors.items()}
        value['metrics'] = {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                            for k, v in self.metrics.items()}
        return value


def blur_effect(gray, h_size=11):
    """Crete et al. 2007 re-blur perceptual blur metric: ~0 sharp, ~1 blurred."""
    gray = gray.astype(np.float32)
    values = []
    # cv2 ksize is (width, height): axis 0 differences need a vertical average.
    for axis, ksize in ((0, (1, h_size)), (1, (h_size, 1))):
        blurred = cv2.blur(gray, ksize, borderType=cv2.BORDER_REFLECT)
        d_f = np.abs(np.diff(gray, axis=axis))
        d_b = np.abs(np.diff(blurred, axis=axis))
        crop = (slice(2, -2), slice(2, -2))
        d_f, d_b = d_f[crop], d_b[crop]
        total = float(d_f.sum())
        if total <= 1e-6:
            values.append(1.0)
            continue
        variation = float(np.maximum(d_f - d_b, 0).sum())
        values.append(1.0 - variation / total)
    return max(values)


def _laplacian_energy(gray, mask=None):
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.6)
    lap = cv2.Laplacian(smooth, cv2.CV_32F)
    return float(lap[mask].var() if mask is not None else lap.var())


class QualityAssessor:
    def __init__(self, quality_config, template):
        self.config = quality_config
        self.size = QUALITY_SIZE
        self.template_size, self.template = template
        self.mask = core_mask(self.size, self.template, self.template_size)
        ys, xs = np.nonzero(self.mask)
        self.mask_rect = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        self.eyes = eye_patches(self.size, self.template, template_size=self.template_size)

    def align(self, frame, geometry):
        return align_face(frame, geometry.points, self.template_size, self.template, self.size)

    def assess(self, frame, detection, geometry, target=None, speed_px_s=0.0, aligned=None, full=True):
        """Return (QualityReport, AlignedFace | None).  Cheap checks run first.

        ``full=False`` stops after the geometric checks (no warp, no pixel
        metrics): used for faces that can neither be selected nor guide.
        """
        c = self.config
        report = QualityReport()
        metrics = report.metrics
        h, w = frame.shape[:2]
        metrics['iod_px'] = geometry.iod
        metrics['detector_score'] = detection.score
        if not geometry.valid:
            report.reasons.append(geometry.reason)
            return report, None
        if geometry.iod < c.min_iod_px:
            report.reasons.append('face_too_small')
        margin = c.border_margin * max(h, w)
        core = [geometry.points[k] for k in ('left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth')
                if k in geometry.points]
        core = np.asarray(core)
        if (core[:, 0].min() < margin or core[:, 1].min() < margin
                or core[:, 0].max() > w - 1 - margin or core[:, 1].max() > h - 1 - margin):
            report.reasons.append('face_clipped')
        front = target is not None and target.lower < 0 < target.upper
        kp = geometry.scores
        if kp:
            eyes = [kp.get('left_eye', 0.0), kp.get('right_eye', 0.0)]
            visible = (min(eyes) if front else max(eyes)) >= c.min_kp_score and kp.get('nose', 0.0) >= c.min_kp_score
            metrics['eye_visibility'] = min(eyes) if front else max(eyes)
            metrics['nose_visibility'] = kp.get('nose', 0.0)
            if not visible:
                report.reasons.append('eyes_or_nose_occluded')
        if report.reasons or not full:  # do not pay for a warp that cannot matter
            report.passed = full and not report.reasons
            return report, None
        if aligned is None:
            aligned = self.align(frame, geometry)
        metrics['alignment_residual'] = aligned.residual
        if aligned.residual > c.max_alignment_residual:
            report.reasons.append('alignment_failed')

        gray = cv2.cvtColor(aligned.image, cv2.COLOR_BGR2GRAY)
        values = gray[self.mask]
        brightness = float(values.mean())
        contrast = float(np.percentile(values, 95) - np.percentile(values, 5))
        clipped = float(np.mean((values >= 250) | (values <= 3)))
        lap = _laplacian_energy(gray, self.mask)
        x1, y1, x2, y2 = self.mask_rect
        crete = blur_effect(gray[y1:y2, x1:x2])
        eye_laps = [_laplacian_energy(gray[b:d, a:c_]) for a, b, c_, d in self.eyes]
        metrics.update(brightness=brightness, contrast=contrast, clipped=clipped, laplacian=lap,
                       blur_effect=crete, eye_laplacian=eye_laps, warp_scale=aligned.scale)

        s_lap = 1 - math.exp(-lap / LAPLACIAN_REF)
        s_crete = float(np.clip((CRETE_BLURRY - crete) / (CRETE_BLURRY - CRETE_SHARP), 0, 1))
        sharpness = 0.5 * (s_lap + s_crete)
        eye_lap = min(eye_laps) if front else max(eye_laps)
        eye_detail = 1 - math.exp(-eye_lap / EYE_LAPLACIAN_REF)
        lo, hi = c.min_brightness, c.max_brightness
        if brightness < 80:
            exposure = (brightness - lo) / max(80 - lo, 1)
        elif brightness > 170:
            exposure = (hi - brightness) / max(hi - 170, 1)
        else:
            exposure = 1.0
        exposure = float(np.clip(exposure, 0, 1)) * (1 - min(clipped / max(c.max_clipped, 1e-6), 1) * 0.5)
        exposure *= float(np.clip(contrast / (2.5 * c.min_contrast), 0.2, 1))
        if sharpness < c.min_sharpness:
            report.reasons.append('blurred')
        if brightness < lo:
            report.reasons.append('too_dark')
        elif brightness > hi or clipped > c.max_clipped:
            report.reasons.append('overexposed')
        elif contrast < c.min_contrast:
            report.reasons.append('low_contrast')

        resolution = (geometry.iod - c.min_iod_px) / (c.good_iod_px - c.min_iod_px)
        confidence = float(np.clip((detection.score - 0.2) / 0.6, 0, 1))
        if kp:
            confidence = math.sqrt(confidence * float(np.mean([metrics['eye_visibility'], metrics['nose_visibility']])))
        speed = speed_px_s / max(geometry.iod, 1e-6)
        metrics['speed_iod_per_s'] = speed
        motion = 1.0 / (1.0 + (speed / c.motion_ref_iod_per_s) ** 2)
        if target is not None:
            half = max(target.center - target.lower, target.upper - target.center)
            fit = 1 - 0.5 * min(abs(geometry.yaw - target.center) / half, 1)
            fit *= 1 - 0.3 * min(abs(geometry.roll_deg) / target.max_roll_deg, 1)
        else:
            fit = 0.5
        alignment = 1 - min(aligned.residual / c.max_alignment_residual, 1) * 0.8
        report.factors = {'sharpness': sharpness, 'eye_detail': eye_detail, 'exposure': exposure,
                          'resolution': float(np.clip(resolution, 0, 1)), 'confidence': confidence,
                          'motion': motion, 'pose_fit': fit, 'alignment': alignment}
        weights = c.weights
        total = sum(weights.get(k, 0) for k in report.factors)
        log_sum = sum(weights.get(k, 0) * math.log(max(v, 0.02)) for k, v in report.factors.items())
        report.score = math.exp(log_sum / total)
        report.passed = not report.reasons
        if report.passed:
            report.level = ('high' if report.score >= c.high_level else
                            'medium' if report.score >= c.medium_level else 'low')
        return report, aligned
