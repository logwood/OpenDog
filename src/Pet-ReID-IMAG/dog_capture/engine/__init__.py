"""Dog face best-shot capture engine (v1).

Mirrors a mature face-capture pipeline -- one-stage detector with keypoints,
species gate, ByteTrack, similarity alignment, FIQA-style quality, per-track
best-shot policy, operator guidance -- on top of portable ONNX Runtime models.

Only NumPy and OpenCV are needed by the engine itself; detectors add ONNX
Runtime.  Nothing here imports torch, TensorFlow, TensorRT or the Re-ID code.
"""
from .config import CaptureConfig, ViewSpec
from .engine import CaptureEngine, FaceResult, FrameResult
from .schema import ANYFACE5, CORE3, DOGFACE6, Detection, KeypointSchema

__all__ = ['CaptureConfig', 'ViewSpec', 'CaptureEngine', 'FaceResult', 'FrameResult',
           'Detection', 'KeypointSchema', 'ANYFACE5', 'DOGFACE6', 'CORE3']
__version__ = '1.0.0'
