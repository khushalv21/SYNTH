"""DivEye text detector — auto-registration."""

from synth.core.registry import DetectorCapability, DetectorMetadata, DetectorRegistry

_META = DetectorMetadata(
    name="diveye",
    capability=DetectorCapability.TEXT_DETECTION,
    speed_tier="balanced",
    requires_gpu=False,
    model_size_mb=550,
    description="IBM DivEye — surprisal-based AI text detection with XGBoost classifier",
    weight=1.2,
)


def _factory(**kwargs):  # type: ignore[no-untyped-def]
    from synth.detectors.diveye.detector import DivEyeDetector

    return DivEyeDetector(**kwargs)


DetectorRegistry.register(_META, _factory)
