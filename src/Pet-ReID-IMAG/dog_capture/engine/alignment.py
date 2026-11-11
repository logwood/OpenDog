"""Five-point similarity alignment, as in ArcFace/InsightFace ``norm_crop``.

Keypoints are mapped to a fixed template with a least-squares similarity
transform (Umeyama 1991): rotation + uniform scale + translation, no shear.
Any in-plane roll, including a phone held sideways, is removed, and every
crop has the same scale, so pixel-level quality measures become comparable
between frames.

The default dog template is the 224 x 224 five-point template published with
PetFace (Shinoda & Shiohara, ECCV 2024; ``keypoints/dog.npy`` at commit
69e7be7a), defined for AnyFace's point order.  Crops therefore match the
convention of PetFace-trained re-identification models.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

PETFACE_DOG_224 = {
    'left_eye': (56.0, 94.75487264), 'right_eye': (168.0, 94.57845561),
    'nose': (112.2124176, 150.36327108),
    'left_mouth': (60.33797073, 165.05586751), 'right_mouth': (163.13275146, 165.04858398),
}
TEMPLATES = {'petface_dog': (224, PETFACE_DOG_224)}


def load_template(name_or_path):
    """Built-in template name, or a JSON file {"size": S, "points": {name: [x, y]}}."""
    if name_or_path in TEMPLATES:
        size, points = TEMPLATES[name_or_path]
        return size, {k: np.asarray(v, np.float64) for k, v in points.items()}
    value = json.loads(Path(name_or_path).read_text(encoding='utf8'))
    return int(value['size']), {k: np.asarray(v, np.float64) for k, v in value['points'].items()}


def umeyama_similarity(src, dst):
    """2x3 matrix M minimising ||M [src;1] - dst||^2 over similarity transforms."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    n = len(src)
    if n < 2:
        raise ValueError('Need at least two point pairs')
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    var_s = (s0 ** 2).sum() / n
    if var_s < 1e-12:
        raise ValueError('Degenerate source points')
    cov = d0.T @ s0 / n
    u, sigma, vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    rotation = u @ sign @ vt
    scale = np.trace(np.diag(sigma) @ sign) / var_s
    matrix = np.zeros((2, 3))
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = mu_d - matrix[:, :2] @ mu_s
    return matrix


@dataclass
class AlignedFace:
    image: np.ndarray        # (S, S, 3) uint8 BGR
    matrix: np.ndarray       # 2x3 frame -> aligned crop
    scale: float             # aligned pixels per source pixel
    residual: float          # mean landmark error / template eye distance
    points: np.ndarray       # the used keypoints in aligned coordinates
    names: tuple


def align_face(frame, points: dict, size=224, template=PETFACE_DOG_224, out_size=None):
    """Warp ``frame`` so the named ``points`` land on the template.

    ``points`` maps keypoint names to frame coordinates; names absent from the
    template (e.g. chin, ear bases) are ignored.  Large downscales are done
    with an area pre-resize so the crop is not aliased, which would otherwise
    make small sharpness differences depend on the face size.
    """
    out_size = out_size or size
    names = tuple(k for k in ('left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth')
                  if k in points and k in template)
    if len(names) < 3:
        raise ValueError('Alignment needs at least eyes and nose')
    factor = out_size / size
    src = np.asarray([points[k] for k in names], np.float64)
    dst = np.asarray([template[k] for k in names], np.float64) * factor
    matrix = umeyama_similarity(src, dst)
    scale = float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))
    projected = src @ matrix[:, :2].T + matrix[:, 2]
    eye_span = float(np.linalg.norm(dst[names.index('left_eye')] - dst[names.index('right_eye')]))
    residual = float(np.linalg.norm(projected - dst, axis=1).mean() / max(eye_span, 1e-6))
    warp_matrix, source = matrix, frame
    if scale < 0.5:
        # Area-resample a source ROI first; the remaining warp scale is ~0.8.
        inverse = cv2.invertAffineTransform(matrix)
        corners = np.array([[0, 0], [out_size, 0], [0, out_size], [out_size, out_size]], np.float64)
        mapped = corners @ inverse[:, :2].T + inverse[:, 2]
        x1, y1 = np.floor(mapped.min(0)).astype(int) - 2
        x2, y2 = np.ceil(mapped.max(0)).astype(int) + 2
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        if x2 - x1 > 1 and y2 - y1 > 1:
            f = scale / 0.8
            roi = frame[y1:y2, x1:x2]
            rw, rh = max(1, round((x2 - x1) * f)), max(1, round((y2 - y1) * f))
            source = cv2.resize(roi, (rw, rh), interpolation=cv2.INTER_AREA)
            fx, fy = rw / (x2 - x1), rh / (y2 - y1)
            pre = np.array([[fx, 0, -fx * x1], [0, fy, -fy * y1], [0, 0, 1]], np.float64)
            warp_matrix = (np.vstack([matrix, [0, 0, 1]]) @ np.linalg.inv(pre))[:2]
    image = cv2.warpAffine(source, warp_matrix, (out_size, out_size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return AlignedFace(image, matrix, scale, residual, projected.astype(np.float32), names)


def core_mask(size=224, template=PETFACE_DOG_224, template_size=224):
    """Fixed elliptical eyes-nose-mouth region in aligned coordinates.

    Pixels outside (ears, fur, background) never influence quality, so a
    sharp fence behind the dog cannot make a blurred face look sharp.
    """
    f = size / float(template_size)
    eyes_y = (template['left_eye'][1] + template['right_eye'][1]) / 2
    mouth_y = (template['left_mouth'][1] + template['right_mouth'][1]) / 2
    cx = (template['left_eye'][0] + template['right_eye'][0]) / 2
    half_w = (template['right_eye'][0] - template['left_eye'][0]) * 0.72
    top, bottom = eyes_y - 26, mouth_y + 16
    mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(mask, (round(cx * f), round((top + bottom) / 2 * f)),
                (round(half_w * f), round((bottom - top) / 2 * f)), 0, 0, 360, 255, -1)
    return mask > 0


def eye_patches(size=224, template=PETFACE_DOG_224, radius=18, template_size=224):
    """Square patches (x1, y1, x2, y2) around both template eyes, aligned coordinates."""
    f = size / float(template_size)
    patches = []
    for name in ('left_eye', 'right_eye'):
        x, y = template[name][0] * f, template[name][1] * f
        r = radius * f
        patches.append((max(0, round(x - r)), max(0, round(y - r)),
                        min(size, round(x + r)), min(size, round(y + r))))
    return patches
