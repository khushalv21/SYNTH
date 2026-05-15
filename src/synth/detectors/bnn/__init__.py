"""BNN image detector — auto-registration."""

from synth.core.registry import DetectorCapability, DetectorMetadata, DetectorRegistry

_META = DetectorMetadata(
    name="bnn",
    capability=DetectorCapability.IMAGE_FORENSICS,
    speed_tier="fast",
    requires_gpu=False,
    model_size_mb=25,
    description="Faster Than Lies — Binary Neural Network for ultra-fast deepfake detection",
    weight=1.0,
)


def _factory(**kwargs):  # type: ignore[no-untyped-def]
    from synth.detectors.bnn.detector import BNNDetector

    return BNNDetector(**kwargs)


DetectorRegistry.register(_META, _factory)
