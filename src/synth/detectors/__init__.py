"""Synth detector plugins — auto-registration package.

Importing this package triggers registration of all bundled detectors
with the :class:`~synth.core.registry.DetectorRegistry`.

Third-party detectors can register themselves by calling
``DetectorRegistry.register(...)`` directly.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register_all() -> None:
    """Import all bundled detector subpackages to trigger registration.

    Called lazily by the :class:`~synth.core.manager.MultiDetectorManager`
    on first use — not at package import time — to avoid loading heavy
    ML libraries until they're needed.
    """
    from synth.detectors import diveye, bnn, cospy, luminol  # noqa: F401

    logger.debug("All bundled detectors registered")
