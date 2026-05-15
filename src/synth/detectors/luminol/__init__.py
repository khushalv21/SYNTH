"""Luminol-AI text detector — auto-registration (experimental)."""

from synth.core.registry import DetectorCapability, DetectorMetadata, DetectorRegistry

_META = DetectorMetadata(
    name="luminol",
    capability=DetectorCapability.STATISTICAL_TEXT,
    speed_tier="forensic",
    requires_gpu=False,
    model_size_mb=550,
    description="Luminol-AI — zero-shot perplexity-under-shuffling text detection (experimental)",
    experimental=True,
    weight=1.0,
)


def _factory(**kwargs):  # type: ignore[no-untyped-def]
    from synth.detectors.luminol.detector import LuminolDetector

    return LuminolDetector(**kwargs)


DetectorRegistry.register(_META, _factory)
