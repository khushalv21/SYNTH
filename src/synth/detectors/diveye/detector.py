"""DivEye detector — IBM's surprisal-based AI text detection.

Wraps the DivEye feature extraction + XGBoost classification pipeline
into a :class:`BaseTextDetector` for ensemble integration.

Architecture::

    text → GPT-2 tokenise → log-likelihoods → surprisal stats
                                                    ↓
                                            10-feature vector
                                                    ↓
                                            XGBoost classify
                                                    ↓
                                           DetectorVote(score, verdict)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from synth.core.device import detect_device, get_torch_device
from synth.core.ensemble import DetectorVote
from synth.core.normalizer import ConfidenceNormalizer
from synth.core.registry import DetectorCapability, DetectorMetadata
from synth.detectors.base import BaseTextDetector

logger = logging.getLogger(__name__)

_META = DetectorMetadata(
    name="diveye",
    capability=DetectorCapability.TEXT_DETECTION,
    speed_tier="balanced",
    requires_gpu=False,
    model_size_mb=550,
    description="IBM DivEye — surprisal-based AI text detection",
    weight=1.2,
)

# Verdict thresholds
AI_THRESHOLD = 0.65
HUMAN_THRESHOLD = 0.35


class DivEyeDetector(BaseTextDetector):
    """DivEye text detector using GPT-2 surprisal + XGBoost.

    Lazy-loads GPT-2 on first use and caches the model for the
    session lifetime.
    """

    def __init__(self, model_name: str = "gpt2") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._tokenizer: Any = None
        self._feature_extractor: Any = None
        self._classifier: Any = None
        self._device = get_torch_device()
        self._device_str = detect_device()

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load GPT-2 and the XGBoost classifier on first use."""
        if self._model is not None:
            return

        logger.info(
            "DivEye: loading %s on %s...", self._model_name, self._device_str
        )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
        self._model.to(self._device)
        self._model.eval()

        from synth.detectors.diveye.features import DivEyeFeatureExtractor

        self._feature_extractor = DivEyeFeatureExtractor(
            self._model, self._tokenizer
        )

        from synth.detectors.diveye.classifier import DivEyeClassifier

        self._classifier = DivEyeClassifier()

        logger.info(
            "DivEye: ready (XGB fallback=%s, device=%s)",
            self._classifier.is_using_fallback,
            self._device_str,
        )

    # ── Detection ─────────────────────────────────────────────────────────

    def detect(self, text: str) -> DetectorVote:
        """Analyse *text* and return an ensemble-ready vote.

        Args:
            text: The content to analyse.

        Returns:
            :class:`DetectorVote` with normalised AI score.
        """
        self._ensure_loaded()

        features = self._feature_extractor.compute(text)

        if features is None:
            # Text too short — return uncertain score
            return DetectorVote(
                detector_name="diveye",
                score=0.5,
                verdict="mixed",
                weight=_META.weight,
            )

        ai_prob = self._classifier.predict_proba(features)
        ai_prob = ConfidenceNormalizer.from_probability(ai_prob)

        if ai_prob >= AI_THRESHOLD:
            verdict = "ai"
        elif ai_prob <= HUMAN_THRESHOLD:
            verdict = "human"
        else:
            verdict = "mixed"

        return DetectorVote(
            detector_name="diveye",
            score=round(ai_prob, 4),
            verdict=verdict,
            weight=_META.weight,
        )

    # ── Metadata & cleanup ────────────────────────────────────────────────

    @property
    def metadata(self) -> DetectorMetadata:
        return _META

    def cleanup(self) -> None:
        """Release GPU memory held by the GPT-2 model."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            del self._feature_extractor
            self._model = None
            self._tokenizer = None
            self._feature_extractor = None

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("DivEye: resources released")
