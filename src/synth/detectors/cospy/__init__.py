"""CO-SPY image detector — auto-registration."""

from synth.core.registry import DetectorCapability, DetectorMetadata, DetectorRegistry

_META = DetectorMetadata(
    name="cospy",
    capability=DetectorCapability.IMAGE_FORENSICS,
    speed_tier="forensic",
    requires_gpu=True,
    model_size_mb=400,
    description="CO-SPY — semantic + pixel fusion for AI-generated image detection",
    weight=1.5,
)


def _factory(**kwargs):  # type: ignore[no-untyped-def]
    from synth.detectors.cospy.detector import COSPYDetector

    return COSPYDetector(**kwargs)


DetectorRegistry.register(_META, _factory)
