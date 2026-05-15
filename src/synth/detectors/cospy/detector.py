"""CO-SPY detector — semantic + pixel fusion for AI image detection.

Wraps the CO-SPY fusion model into a :class:`BaseVisionDetector`
for ensemble integration.  The forensic-tier detector combines
semantic understanding (CLIP-inspired) with pixel-level artifact
detection (SRM-inspired) for high-accuracy deepfake detection.

Architecture::

    image → test_transform → [semantic + artifact] → fusion → probability
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from synth.core.device import detect_device, get_torch_device
from synth.core.ensemble import DetectorVote
from synth.core.normalizer import ConfidenceNormalizer
from synth.core.registry import DetectorCapability, DetectorMetadata
from synth.detectors.base import BaseVisionDetector

logger = logging.getLogger(__name__)

_META = DetectorMetadata(
    name="cospy",
    capability=DetectorCapability.IMAGE_FORENSICS,
    speed_tier="forensic",
    requires_gpu=True,
    model_size_mb=400,
    description="CO-SPY — semantic + pixel fusion for AI image detection",
    weight=1.5,
)


class COSPYDetector(BaseVisionDetector):
    """CO-SPY AI image detector.

    Forensic-tier detector that fuses semantic and artifact features
    for high-accuracy deepfake detection.  Best with GPU; functional
    but slower on CPU.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._transform: Any = None
        self._device = get_torch_device()
        self._device_str = detect_device()

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load the fusion model on first use."""
        if self._model is not None:
            return

        logger.info("CO-SPY: loading fusion model on %s...", self._device_str)

        from synth.detectors.cospy.model import COSPYFusionModel

        self._model = COSPYFusionModel()
        self._model.to(self._device)
        self._model.eval()
        self._transform = self._model.test_transform

        logger.info("CO-SPY: ready (device=%s)", self._device_str)

    # ── Detection ─────────────────────────────────────────────────────────

    def _classify(self, image_rgb: Any) -> DetectorVote:
        """Run classification on a PIL Image."""
        self._ensure_loaded()

        tensor = self._transform(image_rgb).unsqueeze(0).to(self._device)
        probs = self._model.predict(tensor)
        ai_prob = ConfidenceNormalizer.from_probability(probs[0])

        verdict = "fake" if ai_prob >= 0.50 else "real"

        return DetectorVote(
            detector_name="cospy",
            score=round(ai_prob, 4),
            verdict=verdict,
            weight=_META.weight,
        )

    def detect_file(self, path: Path) -> DetectorVote:
        """Classify an image file."""
        from PIL import Image

        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path).convert("RGB")
        return self._classify(img)

    def detect_array(self, array: Any, *, label: str = "<array>") -> DetectorVote:
        """Classify a numpy BGR array."""
        from PIL import Image

        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        return self._classify(img)

    # ── Metadata & cleanup ────────────────────────────────────────────────

    @property
    def metadata(self) -> DetectorMetadata:
        return _META

    def cleanup(self) -> None:
        """Release model memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._transform = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("CO-SPY: resources released")
