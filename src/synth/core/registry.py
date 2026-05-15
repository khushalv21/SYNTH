"""Detector registry — capability-based model discovery and routing.

Every detector (built-in or third-party) registers itself here via
:meth:`DetectorRegistry.register`.  The registry is the **single source
of truth** for what detectors are available, their capabilities, and
which operating profile(s) they belong to.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Capability taxonomy ───────────────────────────────────────────────────────


class DetectorCapability(Enum):
    """Domain a detector operates in."""

    TEXT_DETECTION = "text"
    IMAGE_FORENSICS = "image"
    STATISTICAL_TEXT = "stat_text"


# ── Speed tiers ───────────────────────────────────────────────────────────────

SPEED_FAST = "fast"
SPEED_BALANCED = "balanced"
SPEED_FORENSIC = "forensic"

VALID_SPEED_TIERS = {SPEED_FAST, SPEED_BALANCED, SPEED_FORENSIC}

# Profiles → which tiers are loaded
PROFILE_TIERS: dict[str, set[str]] = {
    SPEED_FAST: {SPEED_FAST},
    SPEED_BALANCED: {SPEED_FAST, SPEED_BALANCED},
    SPEED_FORENSIC: {SPEED_FAST, SPEED_BALANCED, SPEED_FORENSIC},
}


# ── Detector metadata ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DetectorMetadata:
    """Immutable descriptor for a registered detector.

    Attributes:
        name: Unique identifier (e.g. ``"diveye"``, ``"bnn"``).
        capability: Domain the detector operates in.
        speed_tier: One of ``"fast"`` | ``"balanced"`` | ``"forensic"``.
        requires_gpu: ``True`` if the detector needs CUDA/MPS.
        model_size_mb: Approximate weight file size in megabytes.
        description: One-line human-readable summary.
        experimental: ``True`` for detectors not yet production-ready.
        weight: Default ensemble weight (higher = more influence).
    """

    name: str
    capability: DetectorCapability
    speed_tier: str
    requires_gpu: bool = False
    model_size_mb: int = 0
    description: str = ""
    experimental: bool = False
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.speed_tier not in VALID_SPEED_TIERS:
            raise ValueError(
                f"Invalid speed_tier '{self.speed_tier}'. "
                f"Must be one of: {VALID_SPEED_TIERS}"
            )


# ── Registry singleton ────────────────────────────────────────────────────────


class DetectorRegistry:
    """Global registry mapping detector names → metadata + factory callables.

    Detectors register themselves at import time::

        DetectorRegistry.register(
            DetectorMetadata(name="diveye", ...),
            factory=lambda: DivEyeDetector(),
        )

    The manager then queries the registry to build the detector set for
    a given profile::

        text_detectors = DetectorRegistry.get_by_capability(
            DetectorCapability.TEXT_DETECTION,
        )
    """

    _detectors: dict[str, tuple[DetectorMetadata, Callable[..., Any]]] = {}

    @classmethod
    def register(
        cls,
        meta: DetectorMetadata,
        factory: Callable[..., Any],
    ) -> None:
        """Register a detector with its metadata and factory callable."""
        cls._detectors[meta.name] = (meta, factory)
        logger.info(
            "DetectorRegistry: registered '%s' [%s / %s]",
            meta.name,
            meta.capability.value,
            meta.speed_tier,
        )

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a detector from the registry."""
        cls._detectors.pop(name, None)

    @classmethod
    def get(cls, name: str) -> tuple[DetectorMetadata, Callable[..., Any]]:
        """Look up a detector by name.

        Raises:
            KeyError: If no detector is registered under *name*.
        """
        if name not in cls._detectors:
            available = ", ".join(sorted(cls._detectors)) or "(none)"
            raise KeyError(
                f"Unknown detector '{name}'. Registered: {available}"
            )
        return cls._detectors[name]

    @classmethod
    def get_metadata(cls, name: str) -> DetectorMetadata:
        """Return metadata only for the named detector."""
        return cls.get(name)[0]

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Any:
        """Instantiate a detector by calling its registered factory."""
        _, factory = cls.get(name)
        return factory(**kwargs)

    # ── Query helpers ─────────────────────────────────────────────────────

    @classmethod
    def get_by_capability(
        cls,
        cap: DetectorCapability,
        *,
        include_experimental: bool = False,
    ) -> list[str]:
        """Return detector names that match a capability."""
        return [
            name
            for name, (meta, _) in cls._detectors.items()
            if meta.capability == cap
            and (include_experimental or not meta.experimental)
        ]

    @classmethod
    def get_by_profile(
        cls,
        profile: str,
        *,
        capability: DetectorCapability | None = None,
        include_experimental: bool = False,
    ) -> list[str]:
        """Return detector names that belong to a profile's tier set.

        A "balanced" profile includes all "fast" *and* "balanced" detectors.
        A "forensic" profile includes everything.
        """
        allowed_tiers = PROFILE_TIERS.get(profile)
        if allowed_tiers is None:
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Choose from: {sorted(PROFILE_TIERS)}"
            )

        results: list[str] = []
        for name, (meta, _) in cls._detectors.items():
            if meta.speed_tier not in allowed_tiers:
                continue
            if capability is not None and meta.capability != capability:
                continue
            if not include_experimental and meta.experimental:
                continue
            results.append(name)

        return results

    @classmethod
    def all_metadata(cls) -> list[DetectorMetadata]:
        """Return metadata for every registered detector."""
        return [meta for meta, _ in cls._detectors.values()]

    @classmethod
    def available(cls) -> list[str]:
        """Return sorted names of all registered detectors."""
        return sorted(cls._detectors)

    @classmethod
    def clear(cls) -> None:
        """Remove all registered detectors (for testing)."""
        cls._detectors.clear()
