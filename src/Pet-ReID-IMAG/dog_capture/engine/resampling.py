"""Exact area downsampling in two independent source strips.

OpenCV's area resizer schedules stripes by *output* pixel count. A large
close-up reduced to a small crop can therefore remain one task. Split only
at common source/destination boundaries, and check that OpenCV's FP32 area
coefficients stay identical. Ineligible shapes use the original call.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import math
import os
from threading import Lock

import cv2
import numpy as np

_pool = None
_lock = Lock()


def _after_fork():
    global _pool, _lock
    _pool, _lock = None, Lock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)


def _executor():
    global _pool
    with _lock:
        if _pool is None:
            # The calling thread handles the other half. Bounded across engines.
            _pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='capture-area')
        return _pool


def _coefficients(height, out, offset=0):
    """Canonical nonzero row range and FP32 edge/interior area weights.

Matches computeResizeAreaTab in OpenCV: its reciprocal scale, 1e-3 edge
cutoff and final-row clipping matter. Canonicalizing the last full-weight
row allows a strip boundary to have the same coefficients as an interior.
"""
    scale = 1 / (out / height)
    first = np.arange(out, dtype=np.float64) * scale
    last = first + scale
    cell = np.minimum(scale, height - first)
    hi = np.minimum(np.floor(last), height - 1)
    lo = np.minimum(np.ceil(first), hi)
    begin = np.where(lo - first > 1e-3, lo - 1, lo)
    end = np.where(last - hi > 1e-3, hi, hi - 1)
    a = np.where(begin < lo, (lo - first) / cell, 1 / cell)
    b = np.where(end == hi, np.minimum(np.minimum(last - hi, 1), cell) / cell, 1 / cell)
    return np.stack([begin + offset, end + offset, a.astype(np.float32),
                     b.astype(np.float32), (1 / cell).astype(np.float32)], axis=1)


@lru_cache(maxsize=512)
def _split(height, out):
    common = math.gcd(height, out)
    if common < 2:
        return None
    source = height // common * (common // 2)
    target = out // common * (common // 2)
    full = _coefficients(height, out)
    if (np.array_equal(full[:target], _coefficients(source, target))
            and np.array_equal(full[target:], _coefficients(height - source, out - target, source))):
        return source, target
    return None


def resize_area(image, size):
    """BGR uint8 shrink, preserving INTER_AREA pixels and source row strides.

No rounded resize dimensions, changed antialiasing, or cross-frame reuse.
One bounded helper thread overlaps independent strips; all work finishes before
return. Other formats, upscales, small images and unsafe cuts use OpenCV.
"""
    h, w = image.shape[:2]
    nw, nh = size
    eligible = (image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
                and image.size >= 600_000 and 0 < nw < w and 0 < nh < h
                and image.strides[0] > 0 and image.strides[1:] == (3, 1))
    rows = _split(h, nh) if eligible else None
    columns = _split(w, nw) if eligible and rows is None else None
    if rows is None and columns is None:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    output = np.empty((nh, nw, 3), np.uint8)
    if rows is not None:
        sy, dy = rows
        first, second = image[:sy], image[sy:]
        out_first, out_second = output[:dy], output[dy:]
    else:
        sx, dx = columns
        first, second = image[:, :sx], image[:, sx:]
        out_first, out_second = output[:, :dx], output[:, dx:]
    pending = _executor().submit(cv2.resize, first, out_first.shape[1::-1], dst=out_first,
                                 interpolation=cv2.INTER_AREA)
    try:
        cv2.resize(second, out_second.shape[1::-1], dst=out_second, interpolation=cv2.INTER_AREA)
    finally:
        pending.result()  # also wait on exceptions; no work outlives the input
    return output
