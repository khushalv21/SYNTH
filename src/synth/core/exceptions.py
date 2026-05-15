"""Synth custom exceptions."""

from __future__ import annotations

from pathlib import Path


class SynthError(Exception):
    """Base exception for all Synth errors."""


class NoTextFoundError(SynthError):
    """Raised when OCR produces no readable text from an image."""

    def __init__(self, image_path: str | Path) -> None:
        self.image_path = Path(image_path)
        super().__init__(
            f"No readable text found in image: {self.image_path.name}"
        )


class ImageLoadError(SynthError):
    """Raised when an image cannot be loaded or decoded."""

    def __init__(self, image_path: str | Path, reason: str = "unknown") -> None:
        self.image_path = Path(image_path)
        super().__init__(
            f"Failed to load image '{self.image_path.name}': {reason}"
        )


class PDFLoadError(SynthError):
    """Raised when a PDF file cannot be loaded or rendered."""

    def __init__(self, pdf_path: str | Path, reason: str = "unknown") -> None:
        self.pdf_path = Path(pdf_path)
        super().__init__(
            f"Failed to load PDF '{self.pdf_path.name}': {reason}"
        )
