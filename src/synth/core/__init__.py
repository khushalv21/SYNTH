"""Core business logic for Synth."""

from synth.core.auth import (
    APIEndpointConfig,
    AuthResult,
    BaseAuthenticator,
    DetectorFactory,
    LocalHFAuthenticator,
    UniversalAPIAuthenticator,
)
from synth.core.device import (
    detect_device,
    estimate_available_vram,
    get_torch_device,
    supports_mixed_precision,
)
from synth.core.ensemble import DetectorVote, EnsembleAggregator, EnsembleResult
from synth.core.exceptions import ImageLoadError, NoTextFoundError, SynthError
from synth.core.manager import MultiDetectorManager
from synth.core.normalizer import ConfidenceNormalizer
from synth.core.ocr import DocumentScanner, PreprocessConfig
from synth.core.registry import (
    DetectorCapability,
    DetectorMetadata,
    DetectorRegistry,
)
from synth.core.router import AnalysisMode, AnalysisModeResolver
from synth.core.weights import WeightManager

__all__ = [
    # Auth
    "APIEndpointConfig",
    "AuthResult",
    "BaseAuthenticator",
    "DetectorFactory",
    "LocalHFAuthenticator",
    "UniversalAPIAuthenticator",
    # Device
    "detect_device",
    "estimate_available_vram",
    "get_torch_device",
    "supports_mixed_precision",
    # Ensemble
    "DetectorVote",
    "EnsembleAggregator",
    "EnsembleResult",
    # Exceptions
    "ImageLoadError",
    "NoTextFoundError",
    "SynthError",
    # Manager
    "MultiDetectorManager",
    # Normalizer
    "ConfidenceNormalizer",
    # OCR
    "DocumentScanner",
    "PreprocessConfig",
    # Registry
    "DetectorCapability",
    "DetectorMetadata",
    "DetectorRegistry",
    # Router
    "AnalysisMode",
    "AnalysisModeResolver",
    # Weights
    "WeightManager",
]
