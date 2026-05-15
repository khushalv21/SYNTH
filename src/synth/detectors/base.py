"""Abstract base classes for Synth detectors.

All new detectors inherit from :class:`BaseTextDetector` (for text-domain
models) or :class:`BaseVisionDetector` (for image-domain models).

These differ from the legacy :class:`~synth.core.auth.BaseAuthenticator`
interface in several ways:

* They return :class:`~synth.core.ensemble.DetectorVote` (ensemble-ready)
  instead of ``AuthResult``/``VisionAuthResult``.
* They expose :attr:`metadata` for registry integration.
* They provide explicit :meth:`cleanup` for resource management.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from synth.core.ensemble import DetectorVote
from synth.core.registry import DetectorMetadata

if TYPE_CHECKING:
    import numpy as np


class BaseTextDetector(ABC):
    """Interface for all text-domain detectors.

    Subclasses must implement:

    * :meth:`detect` — analyse text and return a :class:`DetectorVote`.
    * :attr:`metadata` — return the detector's :class:`DetectorMetadata`.
    * :meth:`cleanup` — release GPU memory, file handles, etc.
    """

    @abstractmethod
    def detect(self, text: str) -> DetectorVote:
        """Analyse *text* and return an ensemble-ready vote.

        Args:
            text: The content to analyse.

        Returns:
            :class:`DetectorVote` with a normalised score and verdict.
        """
        ...

    @property
    @abstractmethod
    def metadata(self) -> DetectorMetadata:
        """Return the detector's registry metadata."""
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources (models, GPU memory, temp files)."""
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} [{self.metadata.name}]>"


class BaseVisionDetector(ABC):
    """Interface for all image-domain detectors.

    Subclasses must implement:

    * :meth:`detect_file` — analyse an image file.
    * :meth:`detect_array` — analyse a numpy BGR array.
    * :attr:`metadata` — return the detector's :class:`DetectorMetadata`.
    * :meth:`cleanup` — release GPU memory, file handles, etc.
    """

    @abstractmethod
    def detect_file(self, path: Path) -> DetectorVote:
        """Classify an image file as real or AI-generated.

        Args:
            path: Path to the image file.

        Returns:
            :class:`DetectorVote` with a normalised score and verdict.
        """
        ...

    @abstractmethod
    def detect_array(self, array: Any, *, label: str = "<array>") -> DetectorVote:
        """Classify a numpy BGR array as real or AI-generated.

        Args:
            array: A ``uint8`` HxWxC BGR numpy array (OpenCV format).
            label: Human-readable label for logging.

        Returns:
            :class:`DetectorVote`.
        """
        ...

    @property
    @abstractmethod
    def metadata(self) -> DetectorMetadata:
        """Return the detector's registry metadata."""
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources (models, GPU memory, temp files)."""
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} [{self.metadata.name}]>"
