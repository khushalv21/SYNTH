"""BNN (Faster Than Lies) detector — ultra-fast binary deepfake detection.

Combines forensic channel preprocessing (Sobel + FFT + LBP) with a
lightweight backbone for CPU-friendly image forensics.

Architecture::

    image → resize 224 → [RGB + Sobel + FFT + LBP]
                              ↓
                        6ch adapter → ResNet-18 → sigmoid → score
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
    name="bnn",
    capability=DetectorCapability.IMAGE_FORENSICS,
    speed_tier="fast",
    requires_gpu=False,
    model_size_mb=25,
    description="Faster Than Lies — Binary Neural Network deepfake detection",
    weight=1.0,
)

INPUT_SIZE = 224


class BNNDetector(BaseVisionDetector):
    """Ultra-fast binary deepfake detector.

    Uses forensic channel augmentation (Sobel, FFT, LBP) + a lightweight
    backbone.  Designed for CPU-first inference — runs in ~50ms.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._device = get_torch_device()
        self._device_str = detect_device()

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load the backbone model on first use."""
        if self._model is not None:
            return

        logger.info("BNN: loading forensic backbone on %s...", self._device_str)

        from synth.detectors.bnn.backbone import load_forensic_backbone

        self._model = load_forensic_backbone(device=self._device)
        logger.info("BNN: ready (device=%s)", self._device_str)

    # ── Image preprocessing ───────────────────────────────────────────────

    def _preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Preprocess an RGB image to a 6-channel tensor.

        Args:
            image_rgb: HxWx3 uint8 RGB image.

        Returns:
            (1, 6, 224, 224) float32 tensor.
        """
        from synth.detectors.bnn.preprocessing import compute_forensic_channels

        # Resize
        resized = cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE))

        # Compute forensic channels
        forensic = compute_forensic_channels(resized)  # HxWx3 float64

        # Normalise RGB to [0, 1]
        rgb_norm = resized.astype(np.float64) / 255.0

        # Stack: [RGB(3) + forensic(3)] = 6 channels
        combined = np.concatenate([rgb_norm, forensic], axis=-1)  # HxWx6

        # To tensor: (H, W, 6) → (1, 6, H, W)
        tensor = torch.from_numpy(combined).float().permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self._device)

    # ── Detection ─────────────────────────────────────────────────────────

    def _classify(self, image_rgb: np.ndarray) -> DetectorVote:
        """Run classification on an RGB numpy array."""
        self._ensure_loaded()

        tensor = self._preprocess(image_rgb)

        with torch.no_grad():
            logits = self._model(tensor)
            score = torch.sigmoid(logits).item()

        ai_prob = ConfidenceNormalizer.from_probability(score)
        verdict = "fake" if ai_prob >= 0.50 else "real"

        return DetectorVote(
            detector_name="bnn",
            score=round(ai_prob, 4),
            verdict=verdict,
            weight=_META.weight,
        )

    def detect_file(self, path: Path) -> DetectorVote:
        """Classify an image file as real or AI-generated."""
        from PIL import Image

        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path).convert("RGB")
        rgb = np.array(img)
        return self._classify(rgb)

    def detect_array(self, array: Any, *, label: str = "<array>") -> DetectorVote:
        """Classify a numpy BGR array as real or AI-generated."""
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        return self._classify(rgb)

    # ── Metadata & cleanup ────────────────────────────────────────────────

    @property
    def metadata(self) -> DetectorMetadata:
        return _META

    def cleanup(self) -> None:
        """Release model memory."""
        if self._model is not None:
            del self._model
            self._model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("BNN: resources released")
