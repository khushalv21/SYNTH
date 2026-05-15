"""OCR pipeline — image preprocessing and text extraction.

This module provides :class:`DocumentScanner`, which chains an OpenCV
preprocessing pipeline with EasyOCR to extract clean text from document
images.  Hardware selection is automatic: CUDA → MPS → CPU.

PDF ingestion is supported via :func:`pdf_to_images`, which uses
``pypdfium2`` (a self-contained pdfium binding — no system-level
dependencies required) to convert each PDF page into a high-resolution
numpy array that feeds directly into the existing OCR pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from synth.core.device import detect_device
from synth.core.exceptions import ImageLoadError, NoTextFoundError, PDFLoadError

logger = logging.getLogger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────────────
ImageArray = np.ndarray  # uint8 HxW or HxWxC


# ── Preprocessing config ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class PreprocessConfig:
    """Tunable knobs for the OpenCV preprocessing pipeline.

    Attributes:
        adaptive_block_size: Neighbourhood size for adaptive thresholding
            (must be odd, ≥ 3).
        adaptive_c: Constant subtracted from the weighted mean.
        denoise_strength: Filter strength *h* for ``fastNlMeansDenoising``.
            Higher = more aggressive.
    """

    adaptive_block_size: int = 11
    adaptive_c: int = 2
    denoise_strength: int = 10


# ── Main class ────────────────────────────────────────────────────────────────

@dataclass
class DocumentScanner:
    """End-to-end document OCR: preprocess → extract → return text.

    Parameters:
        languages: BCP-47 language codes for EasyOCR (default: ``["en"]``).
        config: Preprocessing tunables.  Uses sensible defaults when omitted.

    Example::

        scanner = DocumentScanner()
        print(scanner.device)          # "mps" on Apple Silicon
        text = scanner.extract_text("receipt.jpg")
    """

    languages: list[str] = field(default_factory=lambda: ["en"])
    config: PreprocessConfig = field(default_factory=PreprocessConfig)

    # ── Resolved at post-init ─────────────────────────────────────────────
    device: str = field(init=False)
    _reader: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.device = detect_device()
        self._reader = self._init_reader()
        logger.info(
            "DocumentScanner ready · device=%s · languages=%s",
            self.device,
            self.languages,
        )

    # ── Reader bootstrap ──────────────────────────────────────────────────

    def _init_reader(self) -> object:
        """Lazily import and initialise the EasyOCR ``Reader``.

        EasyOCR natively supports CUDA but **not** MPS.  When MPS is the
        detected device the reader falls back to CPU for OCR while the
        device hint is preserved for downstream consumers (e.g. the
        authenticator's transformer models, which *do* support MPS).
        """
        import easyocr  # heavy import — deferred on purpose

        use_gpu = self.device == "cuda"

        if self.device == "mps":
            logger.info(
                "EasyOCR does not support MPS — OCR will run on CPU; "
                "MPS device hint preserved for transformer models"
            )

        reader = easyocr.Reader(
            self.languages,
            gpu=use_gpu,
            verbose=False,
        )
        return reader

    # ── Preprocessing pipeline ────────────────────────────────────────────

    def _apply_pipeline(self, img: ImageArray, *, label: str = "<array>") -> ImageArray:
        """Run the OpenCV cleanup pipeline on a loaded BGR/grayscale array.

        This is the shared core used by both :meth:`preprocess` (file-based)
        and :meth:`preprocess_array` (in-memory array).  Callers are
        responsible for loading / providing the raw image data.

        Pipeline stages:
            1. **Grayscale** — convert BGR → single-channel intensity
               (skipped if already single-channel).
            2. **Adaptive threshold** — Gaussian-weighted local threshold
               that removes shadows and compensates for uneven lighting.
            3. **Denoise** — non-local means denoising to suppress scanner
               noise and JPEG artifacts.
        """
        # 1 · Grayscale
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 2 · Adaptive threshold (shadow removal)
        thresh = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=self.config.adaptive_block_size,
            C=self.config.adaptive_c,
        )

        # 3 · Denoise
        denoised: ImageArray = cv2.fastNlMeansDenoising(
            thresh,
            h=self.config.denoise_strength,
        )

        logger.debug(
            "Preprocessed %s → shape=%s dtype=%s",
            label,
            denoised.shape,
            denoised.dtype,
        )
        return denoised

    def preprocess(self, image_path: str | Path) -> ImageArray:
        """Load an image from disk and run the full OpenCV cleanup pipeline.

        Args:
            image_path: Path to the source image file.

        Returns:
            Cleaned ``uint8`` grayscale numpy array ready for OCR.

        Raises:
            FileNotFoundError: If *image_path* does not exist.
            ImageLoadError: If OpenCV cannot decode the file.
        """
        path = Path(image_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = cv2.imread(str(path))
        if img is None:
            raise ImageLoadError(path, reason="OpenCV could not decode the file")

        return self._apply_pipeline(img, label=path.name)

    def preprocess_array(self, image: ImageArray) -> ImageArray:
        """Run the OpenCV cleanup pipeline on an in-memory image array.

        This is the primary entry point for the PDF pipeline, where pages
        are already loaded into memory as numpy arrays.

        Args:
            image: A ``uint8`` numpy array (HxW grayscale *or* HxWxC BGR).

        Returns:
            Cleaned ``uint8`` grayscale array ready for OCR.
        """
        return self._apply_pipeline(image, label="<in-memory>")

    # ── Text extraction ───────────────────────────────────────────────────

    def _ocr_to_text(
        self, processed: ImageArray, *, source_label: str = "<unknown>"
    ) -> str:
        """Run EasyOCR on a preprocessed array and return joined text."""
        results = self._reader.readtext(processed)  # type: ignore[union-attr]

        lines: list[str] = [
            text.strip()
            for _bbox, text, _conf in results
            if text.strip()
        ]

        if not lines:
            raise NoTextFoundError(source_label)

        extracted = "\n".join(lines)
        logger.info(
            "Extracted %d line(s) (%d chars) from %s",
            len(lines),
            len(extracted),
            source_label,
        )
        return extracted

    def extract_text(self, image_path: str | Path) -> str:
        """Run the full pipeline: preprocess → OCR → joined text.

        Args:
            image_path: Path to the source image.

        Returns:
            Extracted text with one OCR line per ``\\n``-delimited line.

        Raises:
            NoTextFoundError: If the image contains no readable characters.
            FileNotFoundError: If *image_path* does not exist.
            ImageLoadError: If the image cannot be decoded.
        """
        processed = self.preprocess(image_path)
        return self._ocr_to_text(processed, source_label=Path(image_path).name)

    def extract_text_from_array(self, image: ImageArray, *, label: str = "<page>") -> str:
        """Preprocess an in-memory image array and extract text.

        This is the entry point used by the PDF pipeline: each page is
        already a numpy array, so we skip the file-load step.

        Args:
            image: A ``uint8`` numpy array (HxW or HxWxC BGR).
            label: Human-readable label for log messages.

        Returns:
            Extracted text.

        Raises:
            NoTextFoundError: If the page contains no readable characters.
        """
        processed = self.preprocess_array(image)
        return self._ocr_to_text(processed, source_label=label)

    # ── Raw results (advanced) ────────────────────────────────────────────

    def extract_raw(
        self, image_path: str | Path
    ) -> list[tuple[list[list[int]], str, float]]:
        """Return raw EasyOCR results with bounding boxes and confidence.

        Each element is ``(bbox, text, confidence)`` where *bbox* is a list
        of four ``[x, y]`` corner points.

        Raises:
            NoTextFoundError: If the image contains no readable characters.
        """
        processed = self.preprocess(image_path)
        results = self._reader.readtext(processed)  # type: ignore[union-attr]

        if not results:
            raise NoTextFoundError(image_path)

        return results  # type: ignore[return-value]

    # ── Fast probe (for auto-detection router) ────────────────────────────

    def probe_text(self, image_path: str | Path) -> str:
        """Quick OCR pass *without* preprocessing — for auto-detection.

        Skips the grayscale → threshold → denoise pipeline and runs
        EasyOCR directly on a downscaled copy of the raw image.  This is
        ~2–5× faster than :meth:`extract_text` and is used by
        :class:`~synth.core.router.AnalysisModeResolver` to check
        whether an image contains enough text to warrant OCR mode.

        Returns:
            Concatenated text (may be empty if no text found).
            Never raises :class:`NoTextFoundError`.
        """
        path = Path(image_path).resolve()
        if not path.exists():
            return ""

        img = cv2.imread(str(path))
        if img is None:
            return ""

        # Resize to max 640px on longest side for probe speed
        h, w = img.shape[:2]
        max_dim = 640
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        try:
            results = self._reader.readtext(img)  # type: ignore[union-attr]
        except Exception:
            return ""

        text = " ".join(
            t.strip() for _, t, _ in results if t.strip()
        )
        logger.debug(
            "Probe OCR on %s → %d chars",
            path.name,
            len(text),
        )
        return text


# ── PDF ingestion ─────────────────────────────────────────────────────────────


def pdf_to_images(
    pdf_path: str | Path,
    *,
    scale: float = 3.0,
) -> list[ImageArray]:
    """Convert every page of a PDF into a high-resolution BGR numpy array.

    Uses ``pypdfium2``, a self-contained pdfium binding that requires
    **no system-level dependencies** (no poppler, no pdf2image).

    Args:
        pdf_path: Path to the ``.pdf`` file.
        scale: Render scale factor.  ``3.0`` ≈ 216 DPI — a good balance
            between OCR accuracy and memory usage.  Increase for very
            small text; decrease for large documents.

    Returns:
        A list of ``uint8`` HxWx3 BGR numpy arrays, one per page.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        PDFLoadError: If the file cannot be opened or rendered.
    """
    import pypdfium2 as pdfium  # lightweight — deferred import

    path = Path(pdf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        pdf = pdfium.PdfDocument(str(path))
    except Exception as exc:
        raise PDFLoadError(path, reason=str(exc)) from exc

    images: list[ImageArray] = []

    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            bitmap = page.render(scale=scale)
            # pypdfium2 renders to RGBA by default → convert to BGR for OpenCV
            pil_image = bitmap.to_pil()
            rgba_array = np.array(pil_image)

            # RGBA → BGR (drop alpha channel, swap R/B for OpenCV)
            bgr: ImageArray = cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2BGR)
            images.append(bgr)

            logger.debug(
                "Rendered page %d/%d from %s → shape=%s",
                page_index + 1,
                len(pdf),
                path.name,
                bgr.shape,
            )
    except PDFLoadError:
        raise
    except Exception as exc:
        raise PDFLoadError(path, reason=f"Page render failed: {exc}") from exc
    finally:
        pdf.close()

    logger.info(
        "Converted %s → %d page image(s)",
        path.name,
        len(images),
    )
    return images
