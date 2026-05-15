"""Ensemble aggregation — weighted consensus from multiple detectors.

Given a list of :class:`DetectorVote` instances (each with a normalised
score, verdict, and weight), the :class:`EnsembleAggregator` computes:

* **Consensus score** — weighted average of individual scores.
* **Consensus verdict** — weighted majority vote with tie-breaking.
* **Agreement ratio** — ``1 - std(scores)``, clamped to [0, 1].
* **Disagreement warning** — raised when agreement < threshold.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

VERDICT_AI = "ai"
VERDICT_HUMAN = "human"
VERDICT_MIXED = "mixed"
VERDICT_FAKE = "fake"
VERDICT_REAL = "real"

# Score thresholds for consensus verdict
AI_THRESHOLD = 0.70
HUMAN_THRESHOLD = 0.30

# Agreement threshold — below this we flag a disagreement warning
AGREEMENT_WARNING_THRESHOLD = 0.60


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DetectorVote:
    """A single detector's output, normalised for ensemble consumption.

    Attributes:
        detector_name: Registry name of the detector.
        score: Normalised AI probability (0.0 → 1.0).
        verdict: The detector's own verdict string.
        weight: Ensemble weight (default 1.0; higher = more influence).
        latency_ms: Wall-clock time the detector took, in milliseconds.
    """

    detector_name: str
    score: float
    verdict: str
    weight: float = 1.0
    latency_ms: float = 0.0


@dataclass(frozen=True)
class EnsembleResult:
    """Aggregated result from multiple detectors.

    Attributes:
        consensus_score: Weighted-average AI probability.
        consensus_verdict: Final verdict after weighted majority vote.
        individual_votes: Per-detector breakdown.
        agreement_ratio: 0 → 1, how much detectors agree (1 = unanimous).
        disagreement_warning: ``True`` when agreement is below threshold.
        reasoning: Human-readable summary of the ensemble decision.
        domain: ``"text"`` or ``"image"``.
    """

    consensus_score: float
    consensus_verdict: str
    individual_votes: list[DetectorVote]
    agreement_ratio: float
    disagreement_warning: bool
    reasoning: str
    domain: str = "text"


# ── Aggregator ────────────────────────────────────────────────────────────────


class EnsembleAggregator:
    """Compute weighted consensus from multiple :class:`DetectorVote` instances.

    Usage::

        agg = EnsembleAggregator()
        result = agg.aggregate(votes, domain="text")
    """

    def __init__(
        self,
        *,
        ai_threshold: float = AI_THRESHOLD,
        human_threshold: float = HUMAN_THRESHOLD,
        agreement_threshold: float = AGREEMENT_WARNING_THRESHOLD,
    ) -> None:
        self._ai_threshold = ai_threshold
        self._human_threshold = human_threshold
        self._agreement_threshold = agreement_threshold

    # ── Public API ────────────────────────────────────────────────────────

    def aggregate(
        self,
        votes: list[DetectorVote],
        *,
        domain: str = "text",
    ) -> EnsembleResult:
        """Aggregate a list of detector votes into a consensus result.

        Args:
            votes: One or more :class:`DetectorVote` instances.
            domain: ``"text"`` or ``"image"`` — affects verdict labels.

        Returns:
            :class:`EnsembleResult` with weighted consensus.

        Raises:
            ValueError: If *votes* is empty.
        """
        if not votes:
            raise ValueError("Cannot aggregate an empty list of votes")

        # Single detector — pass through
        if len(votes) == 1:
            v = votes[0]
            return EnsembleResult(
                consensus_score=round(v.score, 4),
                consensus_verdict=v.verdict,
                individual_votes=votes,
                agreement_ratio=1.0,
                disagreement_warning=False,
                reasoning=f"Single detector ({v.detector_name}): {v.verdict}",
                domain=domain,
            )

        # ── Weighted average score ────────────────────────────────────────
        total_weight = sum(v.weight for v in votes)
        if total_weight == 0:
            total_weight = len(votes)
            weighted_score = sum(v.score for v in votes) / total_weight
        else:
            weighted_score = (
                sum(v.score * v.weight for v in votes) / total_weight
            )

        # ── Agreement ratio ───────────────────────────────────────────────
        agreement = self._compute_agreement(votes)
        disagreement = agreement < self._agreement_threshold

        # ── Consensus verdict ─────────────────────────────────────────────
        verdict = self._score_to_verdict(weighted_score, domain)

        # ── Reasoning ─────────────────────────────────────────────────────
        reasoning = self._build_reasoning(
            votes, weighted_score, verdict, agreement, disagreement
        )

        if disagreement:
            logger.warning(
                "Ensemble disagreement detected (agreement=%.2f): %s",
                agreement,
                ", ".join(
                    f"{v.detector_name}={v.verdict}" for v in votes
                ),
            )

        return EnsembleResult(
            consensus_score=round(weighted_score, 4),
            consensus_verdict=verdict,
            individual_votes=votes,
            agreement_ratio=round(agreement, 4),
            disagreement_warning=disagreement,
            reasoning=reasoning,
            domain=domain,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _score_to_verdict(self, score: float, domain: str) -> str:
        """Map a consensus score to a verdict label."""
        if domain == "image":
            return VERDICT_FAKE if score >= 0.50 else VERDICT_REAL
        # Text domain
        if score >= self._ai_threshold:
            return VERDICT_AI
        if score <= self._human_threshold:
            return VERDICT_HUMAN
        return VERDICT_MIXED

    @staticmethod
    def _compute_agreement(votes: list[DetectorVote]) -> float:
        """Compute agreement ratio: ``1 - std(scores)``, clamped to [0, 1].

        A ratio of 1.0 means all detectors returned the same score.
        A ratio near 0.0 means maximal disagreement.
        """
        scores = [v.score for v in votes]
        n = len(scores)
        if n <= 1:
            return 1.0

        mean = sum(scores) / n
        variance = sum((s - mean) ** 2 for s in scores) / n
        std = math.sqrt(variance)

        # std of [0, 1] data is at most 0.5 (when split 50/50 between
        # 0 and 1), so we normalise by 0.5 to get a 0→1 disagreement
        # metric, then subtract from 1 for agreement.
        return max(0.0, min(1.0, 1.0 - (std / 0.5)))

    @staticmethod
    def _build_reasoning(
        votes: list[DetectorVote],
        score: float,
        verdict: str,
        agreement: float,
        disagreement: bool,
    ) -> str:
        """Build a human-readable ensemble reasoning string."""
        vote_strs = [
            f"{v.detector_name}={v.score:.2f}" for v in votes
        ]
        parts = [
            f"Ensemble consensus: {verdict} ({score * 100:.1f}%).",
            f"Votes: [{', '.join(vote_strs)}].",
            f"Agreement: {agreement * 100:.0f}%.",
        ]
        if disagreement:
            parts.append("⚠ Significant detector disagreement detected.")
        return " ".join(parts)
