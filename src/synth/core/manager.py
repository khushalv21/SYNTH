"""Multi-detector lifecycle manager.

Orchestrates lazy loading, caching, and ensemble execution across
multiple detectors.  Selects detectors based on the active profile
and delegates to the :class:`~synth.core.ensemble.EnsembleAggregator`
for consensus.

Usage::

    mgr = MultiDetectorManager(profile="balanced")
    result = mgr.detect_text("Some text to analyse...")
    result = mgr.detect_image(Path("photo.png"))
    mgr.unload_all()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from synth.core.ensemble import DetectorVote, EnsembleAggregator, EnsembleResult
from synth.core.registry import (
    DetectorCapability,
    DetectorRegistry,
    SPEED_BALANCED,
)

logger = logging.getLogger(__name__)


class MultiDetectorManager:
    """Lazy-load, cache, and orchestrate multiple detectors.

    Detectors are instantiated on first use and cached as singletons.
    The profile determines which detectors are loaded.

    Attributes:
        profile: Operating profile (``"fast"`` | ``"balanced"`` | ``"forensic"``).
    """

    def __init__(self, profile: str = SPEED_BALANCED) -> None:
        self._profile = profile
        self._loaded: dict[str, Any] = {}
        self._aggregator = EnsembleAggregator()
        self._registered = False

    # ── Lazy registration ─────────────────────────────────────────────────

    def _ensure_registered(self) -> None:
        """Trigger bulk detector registration on first use."""
        if self._registered:
            return
        from synth.detectors import register_all

        register_all()
        self._registered = True

    # ── Detector lifecycle ────────────────────────────────────────────────

    def _get_or_load(self, name: str) -> Any:
        """Return a cached detector instance, loading it if necessary."""
        if name not in self._loaded:
            logger.info("Loading detector: %s", name)
            instance = DetectorRegistry.create(name)
            self._loaded[name] = instance
        return self._loaded[name]

    def unload(self, name: str) -> None:
        """Unload a specific detector and free its resources."""
        if name in self._loaded:
            instance = self._loaded.pop(name)
            if hasattr(instance, "cleanup"):
                instance.cleanup()
            logger.info("Unloaded detector: %s", name)

    def unload_all(self) -> None:
        """Unload all cached detectors."""
        for name in list(self._loaded):
            self.unload(name)

    # ── Text detection ────────────────────────────────────────────────────

    def detect_text(self, text: str) -> EnsembleResult:
        """Run all text detectors for the active profile and return consensus.

        Args:
            text: The content to analyse.

        Returns:
            :class:`EnsembleResult` with weighted consensus from all
            text detectors in the profile.
        """
        self._ensure_registered()

        detector_names = DetectorRegistry.get_by_profile(
            self._profile,
            capability=DetectorCapability.TEXT_DETECTION,
        )

        # Also include statistical text detectors
        stat_names = DetectorRegistry.get_by_profile(
            self._profile,
            capability=DetectorCapability.STATISTICAL_TEXT,
        )
        detector_names.extend(stat_names)

        if not detector_names:
            raise RuntimeError(
                f"No text detectors available for profile '{self._profile}'"
            )

        votes: list[DetectorVote] = []
        for name in detector_names:
            try:
                detector = self._get_or_load(name)
                t0 = time.perf_counter()
                vote = detector.detect(text)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                # Inject latency
                vote = DetectorVote(
                    detector_name=vote.detector_name,
                    score=vote.score,
                    verdict=vote.verdict,
                    weight=vote.weight,
                    latency_ms=round(elapsed_ms, 1),
                )
                votes.append(vote)
                logger.info(
                    "Detector '%s': score=%.3f verdict=%s (%.0fms)",
                    name, vote.score, vote.verdict, elapsed_ms,
                )
            except Exception as exc:
                logger.warning(
                    "Detector '%s' failed, skipping: %s", name, exc
                )

        if not votes:
            raise RuntimeError(
                "All text detectors failed. Cannot produce a result."
            )

        return self._aggregator.aggregate(votes, domain="text")

    # ── Image detection ───────────────────────────────────────────────────

    def detect_image(self, image_path: Path) -> EnsembleResult:
        """Run all image detectors for the active profile on a file.

        Args:
            image_path: Path to the image file.

        Returns:
            :class:`EnsembleResult` with weighted consensus from all
            image detectors in the profile.
        """
        self._ensure_registered()

        detector_names = DetectorRegistry.get_by_profile(
            self._profile,
            capability=DetectorCapability.IMAGE_FORENSICS,
        )

        if not detector_names:
            raise RuntimeError(
                f"No image detectors available for profile '{self._profile}'"
            )

        votes: list[DetectorVote] = []
        for name in detector_names:
            try:
                detector = self._get_or_load(name)
                t0 = time.perf_counter()
                vote = detector.detect_file(image_path)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                vote = DetectorVote(
                    detector_name=vote.detector_name,
                    score=vote.score,
                    verdict=vote.verdict,
                    weight=vote.weight,
                    latency_ms=round(elapsed_ms, 1),
                )
                votes.append(vote)
                logger.info(
                    "Detector '%s': score=%.3f verdict=%s (%.0fms)",
                    name, vote.score, vote.verdict, elapsed_ms,
                )
            except Exception as exc:
                logger.warning(
                    "Detector '%s' failed, skipping: %s", name, exc
                )

        if not votes:
            raise RuntimeError(
                "All image detectors failed. Cannot produce a result."
            )

        return self._aggregator.aggregate(votes, domain="image")

    def detect_image_array(
        self, array: Any, *, label: str = "<array>"
    ) -> EnsembleResult:
        """Run all image detectors on a numpy BGR array.

        Args:
            array: A ``uint8`` HxWxC BGR numpy array.
            label: Human-readable label for logging.

        Returns:
            :class:`EnsembleResult`.
        """
        self._ensure_registered()

        detector_names = DetectorRegistry.get_by_profile(
            self._profile,
            capability=DetectorCapability.IMAGE_FORENSICS,
        )

        if not detector_names:
            raise RuntimeError(
                f"No image detectors available for profile '{self._profile}'"
            )

        votes: list[DetectorVote] = []
        for name in detector_names:
            try:
                detector = self._get_or_load(name)
                t0 = time.perf_counter()
                vote = detector.detect_array(array, label=label)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                vote = DetectorVote(
                    detector_name=vote.detector_name,
                    score=vote.score,
                    verdict=vote.verdict,
                    weight=vote.weight,
                    latency_ms=round(elapsed_ms, 1),
                )
                votes.append(vote)
            except Exception as exc:
                logger.warning(
                    "Detector '%s' failed on %s, skipping: %s",
                    name, label, exc,
                )

        if not votes:
            raise RuntimeError(
                "All image detectors failed. Cannot produce a result."
            )

        return self._aggregator.aggregate(votes, domain="image")

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def profile(self) -> str:
        """Return the active operating profile."""
        return self._profile

    @property
    def loaded_detectors(self) -> list[str]:
        """Return names of currently loaded (cached) detectors."""
        return list(self._loaded.keys())

    def available_detectors(self) -> dict[str, list[str]]:
        """Return detectors available for the current profile, by domain."""
        self._ensure_registered()
        return {
            "text": DetectorRegistry.get_by_profile(
                self._profile,
                capability=DetectorCapability.TEXT_DETECTION,
            )
            + DetectorRegistry.get_by_profile(
                self._profile,
                capability=DetectorCapability.STATISTICAL_TEXT,
            ),
            "image": DetectorRegistry.get_by_profile(
                self._profile,
                capability=DetectorCapability.IMAGE_FORENSICS,
            ),
        }
