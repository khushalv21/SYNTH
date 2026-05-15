"""Confidence score normalisation — unified 0 → 1 AI probability.

Different detectors emit wildly different output formats: raw logits,
softmax probabilities, XGBoost ``predict_proba`` vectors, or discrete
classification labels.  This module maps them all to a single ``float``
on ``[0.0, 1.0]`` where ``0.0`` = definitely human/real and
``1.0`` = definitely AI-generated.
"""

from __future__ import annotations

import math


class ConfidenceNormalizer:
    """Static utility class for normalising diverse model outputs."""

    @staticmethod
    def from_sigmoid(logit: float) -> float:
        """Map a raw logit through sigmoid → [0, 1].

        Useful for models that output a single un-activated score.
        """
        return 1.0 / (1.0 + math.exp(-logit))

    @staticmethod
    def from_probability(prob: float) -> float:
        """Clamp a probability value to [0, 1].

        No-op when the value is already valid; guards against
        floating-point drift.
        """
        return max(0.0, min(1.0, prob))

    @staticmethod
    def from_binary_proba(proba_vector: list[float]) -> float:
        """Extract AI probability from a binary ``[human, ai]`` vector.

        If the vector is ``[p_human, p_ai]``, returns ``p_ai``.
        Falls back to the second element for any 2-element vector.
        """
        if len(proba_vector) == 2:
            return max(0.0, min(1.0, proba_vector[1]))
        # Single-element: treat as AI probability directly
        if len(proba_vector) == 1:
            return max(0.0, min(1.0, proba_vector[0]))
        raise ValueError(
            f"Expected 1 or 2-element probability vector, got {len(proba_vector)}"
        )

    @staticmethod
    def from_classification(
        label: str,
        score: float,
        *,
        ai_labels: tuple[str, ...] = (
            "AI-generated",
            "Fake",
            "fake",
            "ai",
            "AI",
            "LABEL_1",
            "machine",
            "artificial",
        ),
    ) -> float:
        """Map a (label, score) pair to AI probability.

        If the label is an AI-positive label, returns the score directly.
        Otherwise returns ``1 - score`` (i.e. the complement).
        """
        if label in ai_labels:
            return max(0.0, min(1.0, score))
        return max(0.0, min(1.0, 1.0 - score))

    @staticmethod
    def invert(score: float) -> float:
        """Flip the polarity: human-probability ↔ AI-probability."""
        return 1.0 - max(0.0, min(1.0, score))

    @staticmethod
    def from_logit(logit: float) -> float:
        """Alias for :meth:`from_sigmoid` — map raw logit to [0, 1].

        Provided for readability when the input is explicitly a logit.
        """
        return 1.0 / (1.0 + math.exp(-logit))

    @staticmethod
    def from_binary_label(label: int) -> float:
        """Map a binary integer label (0 or 1) to AI probability.

        Args:
            label: ``1`` for AI / fake, ``0`` for human / real.
        """
        return float(max(0, min(1, label)))

    @staticmethod
    def from_string_label(label: str) -> float:
        """Map a string verdict to AI probability.

        Returns:
            ``1.0`` for AI-positive labels, ``0.0`` for human-positive,
            ``0.5`` for ambiguous / mixed.
        """
        low = label.lower().strip()
        if low in ("ai", "fake", "ai-generated", "machine", "artificial", "label_1"):
            return 1.0
        if low in ("human", "real", "label_0", "natural"):
            return 0.0
        return 0.5
