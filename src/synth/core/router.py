"""Automatic analysis mode detection.

Determines whether a file should be analysed with:

* **Text mode** — OCR → AI text detection (documents, text-heavy images, PDFs)
* **Image mode** — Vision Transformer forensics (photos, art, AI-generated images)

The resolver uses a lightweight OCR probe to check whether an image
contains readable text.  If enough text is found, the file is routed to
text mode; otherwise it goes to image forensics.

PDFs are always routed to text mode (they are documents by definition).
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# ── File extension sets ───────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXTENSION = ".pdf"
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | {PDF_EXTENSION}


class AnalysisMode(str, Enum):
    """Which pipeline to run on a file."""

    text = "text"
    image = "image"


class AnalysisModeResolver:
    """Decide whether each file should use text or image analysis.

    The resolver keeps a reference to an optional :class:`DocumentScanner`
    so it can reuse the already-loaded EasyOCR reader for probing.

    Parameters:
        text_char_threshold: Minimum number of OCR characters that must
            be extracted for the file to be classified as text-mode.
    """

    TEXT_CHAR_THRESHOLD: int = 30

    def __init__(self, text_char_threshold: int = TEXT_CHAR_THRESHOLD) -> None:
        self._threshold = text_char_threshold

    # ── Public API ────────────────────────────────────────────────────────

    def resolve_file(self, path: Path, scanner: object | None = None) -> AnalysisMode:
        """Determine the analysis mode for a single file.

        Args:
            path: Path to the file.
            scanner: An optional :class:`~synth.core.ocr.DocumentScanner`
                instance used for the text-density probe on images.

        Returns:
            :class:`AnalysisMode.text` or :class:`AnalysisMode.image`.
        """
        suffix = path.suffix.lower()

        # PDFs → always text
        if suffix == PDF_EXTENSION:
            logger.debug("Router: %s is PDF → text mode", path.name)
            return AnalysisMode.text

        # Not a supported image
        if suffix not in IMAGE_EXTENSIONS:
            logger.debug("Router: %s unsupported extension → default text", path.name)
            return AnalysisMode.text

        # Images → OCR probe
        if scanner is not None:
            return self._probe_image(path, scanner)

        # No scanner available — default to text
        logger.debug("Router: no scanner for probe → default text mode for %s", path.name)
        return AnalysisMode.text

    def resolve_batch(
        self,
        image_files: list[Path],
        pdf_files: list[Path],
        scanner: object | None = None,
    ) -> dict[Path, AnalysisMode]:
        """Classify every file in a mixed batch.

        Returns:
            A dict mapping each file path to its resolved mode.
        """
        modes: dict[Path, AnalysisMode] = {}

        for pdf in pdf_files:
            modes[pdf] = AnalysisMode.text

        for img in image_files:
            modes[img] = self.resolve_file(img, scanner=scanner)

        text_count = sum(1 for m in modes.values() if m == AnalysisMode.text)
        image_count = sum(1 for m in modes.values() if m == AnalysisMode.image)
        logger.info(
            "Router: %d file(s) → %d text, %d image",
            len(modes),
            text_count,
            image_count,
        )
        return modes

    # ── Internal ──────────────────────────────────────────────────────────

    def _probe_image(self, path: Path, scanner: object) -> AnalysisMode:
        """Quick OCR probe to determine text density.

        Calls the scanner's ``probe_text`` method (a fast, no-preprocessing
        OCR pass) and checks whether the result exceeds the threshold.
        """
        try:
            # Use the fast probe method (no preprocessing)
            probe_text: str = scanner.probe_text(path)  # type: ignore[union-attr]
            char_count = len(probe_text.strip())

            if char_count >= self._threshold:
                logger.debug(
                    "Router: %s has %d chars → text mode",
                    path.name,
                    char_count,
                )
                return AnalysisMode.text

            logger.debug(
                "Router: %s has %d chars (< %d) → image mode",
                path.name,
                char_count,
                self._threshold,
            )
            return AnalysisMode.image

        except Exception as exc:
            # If probe fails, default to text mode (safer)
            logger.debug(
                "Router: probe failed for %s (%s) → default text mode",
                path.name,
                exc,
            )
            return AnalysisMode.text
