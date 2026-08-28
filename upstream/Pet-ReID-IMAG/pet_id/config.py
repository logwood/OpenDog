# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

from fastreid.config import CfgNode as CN


def add_retri_config(cfg):
    _C = cfg

    _C.TEST.RECALLS = [1, 2, 4, 8, 16, 32]

    # Frozen geometry providers + locally end-to-end identity fusion. These
    # options are inert for the released single-branch trainer unless the
    # multimodal CLI/model is explicitly constructed.
    _C.MULTIMODAL = CN()
    _C.MULTIMODAL.ENABLED = False
    _C.MULTIMODAL.NOSE_CONFIG = "logs/s101_224/config.yaml"
    _C.MULTIMODAL.NOSE_WEIGHTS = "logs/s101_224/model_final.pth"
    _C.MULTIMODAL.IDENTITY_WEIGHTS = ""
    _C.MULTIMODAL.ARCFACE_WEIGHTS = "../../dog.pt"
    _C.MULTIMODAL.ANYFACE_ROOT = "third_party/AnyFace"
    _C.MULTIMODAL.ANYFACE_WEIGHTS = (
        "third_party/AnyFace/yolov5-face/weights/yolov5l6_best.pt"
    )
    _C.MULTIMODAL.SAM2_CHECKPOINT = (
        "third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"
    )
    _C.MULTIMODAL.SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
    _C.MULTIMODAL.ANYFACE_IMAGE_SIZE = 800
    _C.MULTIMODAL.ANYFACE_CONFIDENCE = 0.20
    _C.MULTIMODAL.MAX_LONG_SIDE = 1280
    _C.MULTIMODAL.NOSE_SIZE = [244, 244]
    _C.MULTIMODAL.FACE_SIZE = [224, 224]
    _C.MULTIMODAL.NOSE_PRIOR = 0.75
    _C.MULTIMODAL.FACE_PRIOR = 0.25
    _C.MULTIMODAL.NUM_CLASSES = 0
    _C.MULTIMODAL.NOSE_TRAINABLE_PARTS = []
    _C.MULTIMODAL.ARCFACE_TRAINABLE_PARTS = ["layer4", "fc"]
    _C.MULTIMODAL.JOINT_ENABLED = False
    _C.MULTIMODAL.FUSION_MODE = "legacy_concat"
    _C.MULTIMODAL.JOINT_DIM = 512
    _C.MULTIMODAL.ADAPTER_BOTTLENECK_DIM = 128
    _C.MULTIMODAL.JOINT_INITIAL_MIX = 0.0025
    _C.MULTIMODAL.MODALITY_DROPOUT = 0.0
    _C.MULTIMODAL.CROSS_VIEW_WEIGHT = 0.0
    _C.MULTIMODAL.CROSS_MODAL_WEIGHT = 0.0
    _C.MULTIMODAL.BRANCH_CONSISTENCY_WEIGHT = 0.0
    _C.MULTIMODAL.SEMANTIC_MAX_NOSE_WEIGHT = 0.35
    _C.MULTIMODAL.SEMANTIC_RESIDUAL_SCALE = 0.05
    _C.MULTIMODAL.SEMANTIC_CONFLICT_WEIGHT = 0.0
    _C.MULTIMODAL.SEMANTIC_CONFLICT_MARGIN = 0.05
    _C.MULTIMODAL.DOMINANCE_WEIGHT = 0.0
    _C.MULTIMODAL.DOMINANCE_TOLERANCE = 0.02
    _C.MULTIMODAL.CONTRASTIVE_TEMPERATURE = 0.10
    _C.MULTIMODAL.CONTRASTIVE_POSE_BOOST = 1.0
    _C.MULTIMODAL.VIEWPOINT_NOSE_PENALTY = 0.35
    _C.MULTIMODAL.VIEWPOINT_NOSE_FLOOR = 0.50
    _C.MULTIMODAL.ALLOW_RAW_NOSE_FALLBACK = True
    _C.MULTIMODAL.CACHE_DIR = "logs/multimodal_cache"
