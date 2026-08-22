from .config import add_retri_config
from .dataset import (
    PetID,
    PetIDFull,
    PetIDSmoke,
    PetIDTest,
    PetIDValidation,
    PetIDValidationSmoke,
)
from .evaluator import PetIDEvaluator, PetIDFeatureEvaluator, PetIDVerificationEvaluator
from .arcface import DogArcFaceEncoder, fuse_main_and_arcface, fuse_normalized_features
from .localization import (
    VIEWPOINT_DIM,
    AnyFaceDetector,
    FaceDetection,
    SAM2NoseSegmenter,
    viewpoint_signals,
)
from .multimodal import (
    DescriptorCache,
    FastReIDDescriptorEncoder,
    LocalEndToEndPetIDModel,
    MultimodalPetIDPipeline,
    PairSimilarity,
    PetDescriptor,
    QualityFusionGate,
    ResidualProjectionAdapter,
    CrossModalResidual,
    build_local_identity_model,
    build_multimodal_pipeline,
    compare_descriptors,
    viewpoint_supervised_contrastive_loss,
)
from .dogfacenet_alignment import (
    AlignmentIndexRecord,
    PKBatchSampler,
    PreparedDogFaceNetDataset,
    TargetDetectionMatch,
    build_alignment_index,
    collate_prepared_dogfacenet,
    dogfacenet_identity_from_filename,
    match_annotated_target,
    prepare_alignment_record,
)
