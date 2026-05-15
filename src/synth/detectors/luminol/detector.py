"""Luminol-AI detector — zero-shot perplexity-under-shuffling text detection.

Implements the full Luminol-AIDetect algorithm from the paper:

1. Shuffle the input text
2. Compute perplexity of both original and shuffled text
3. Extract 5 perplexity-feature types
4. Classify via density estimation against pre-fitted distributions
5. Ensemble vote across features

This is marked as **experimental** — the distribution parameters are
preliminary and should be calibrated with proper benchmark data.

Architecture::

    text → shuffle(text) → [ppl(text), ppl(shuffled)]
                                    ↓
                            feature_vector (5 dims)
                                    ↓
                            density estimation
                            (fitted Gamma PDFs)
                                    ↓
                            ensemble vote → verdict
"""

from __future__ import annotations

import logging
from typing import Any

from synth.core.device import detect_device, get_torch_device
from synth.core.ensemble import DetectorVote
from synth.core.normalizer import ConfidenceNormalizer
from synth.core.registry import DetectorCapability, DetectorMetadata
from synth.detectors.base import BaseTextDetector

logger = logging.getLogger(__name__)

_META = DetectorMetadata(
    name="luminol",
    capability=DetectorCapability.STATISTICAL_TEXT,
    speed_tier="forensic",
    requires_gpu=False,
    model_size_mb=550,
    description="Luminol-AI — zero-shot perplexity-under-shuffling detection (experimental)",
    experimental=True,
    weight=1.0,
)


class LuminolDetector(BaseTextDetector):
    """Luminol-AI zero-shot text detector.

    Uses perplexity-under-shuffling as a statistical discriminant
    between human and AI-generated text.  Shares GPT-2 with DivEye
    when both are loaded.

    .. note::

        This detector is marked as **experimental**.  It is excluded
        from default profiles and must be explicitly enabled via
        ``--profile forensic`` or ``--detector luminol``.
    """

    def __init__(self, model_name: str = "gpt2") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._tokenizer: Any = None
        self._device = get_torch_device()
        self._device_str = detect_device()
        self._distributions: Any = None

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load GPT-2 and distribution parameters on first use."""
        if self._model is not None:
            return

        logger.info(
            "Luminol: loading %s on %s...", self._model_name, self._device_str
        )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
        self._model.to(self._device)
        self._model.eval()

        from synth.detectors.luminol.features import load_distributions

        self._distributions = load_distributions()

        logger.info("Luminol: ready (device=%s)", self._device_str)

    # ── Detection ─────────────────────────────────────────────────────────

    def detect(self, text: str) -> DetectorVote:
        """Analyse *text* using perplexity-under-shuffling.

        Steps:
            1. Shuffle the text
            2. Compute perplexity of original and shuffled versions
            3. Extract perplexity features
            4. Classify via density estimation

        Args:
            text: The content to analyse.

        Returns:
            :class:`DetectorVote` with the zero-shot classification.
        """
        self._ensure_loaded()

        # Guard: text too short
        if len(text.split()) < 15:
            return DetectorVote(
                detector_name="luminol",
                score=0.5,
                verdict="mixed",
                weight=_META.weight,
            )

        # Step 1: Shuffle
        from synth.detectors.luminol.shuffler import shuffle_text

        text_shuffled = shuffle_text(text, seed=42)

        # Step 2: Compute perplexities
        from synth.detectors.luminol.perplexity import compute_perplexity

        ppl_original = compute_perplexity(
            text, self._model, self._tokenizer
        )
        ppl_shuffled = compute_perplexity(
            text_shuffled, self._model, self._tokenizer
        )

        # Step 3: Extract features
        from synth.detectors.luminol.features import (
            classify_with_density,
            extract_perplexity_features,
        )

        features = extract_perplexity_features(ppl_original, ppl_shuffled)

        # Step 4: Density-based classification
        ai_prob, verdict = classify_with_density(
            features, self._distributions
        )

        ai_prob = ConfidenceNormalizer.from_probability(ai_prob)

        return DetectorVote(
            detector_name="luminol",
            score=round(ai_prob, 4),
            verdict=verdict,
            weight=_META.weight,
        )

    # ── Metadata & cleanup ────────────────────────────────────────────────

    @property
    def metadata(self) -> DetectorMetadata:
        return _META

    def cleanup(self) -> None:
        """Release GPU memory held by GPT-2."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Luminol: resources released")
