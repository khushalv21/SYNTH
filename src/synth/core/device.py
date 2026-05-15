"""Shared hardware detection utility.

Centralised device resolution used by both the OCR pipeline
(EasyOCR) and the authenticator pipeline (HuggingFace / PyTorch).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """Return the best available compute device string.

    Priority order:
        1. **cuda** — NVIDIA GPU via CUDA
        2. **mps**  — Apple Silicon GPU via Metal Performance Shaders
        3. **cpu**  — Fallback

    Returns:
        One of ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    if torch.cuda.is_available():
        device = "cuda"
        detail = torch.cuda.get_device_name(0)
        logger.info("Hardware probe: CUDA available → %s", detail)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Hardware probe: Apple MPS available")
    else:
        device = "cpu"
        logger.info("Hardware probe: No GPU detected → CPU fallback")
    return device


def get_torch_device() -> torch.device:
    """Return a ``torch.device`` for the best available hardware.

    Convenience wrapper around :func:`detect_device` for modules
    that need an actual ``torch.device`` object.
    """
    return torch.device(detect_device())


def estimate_available_vram() -> int:
    """Estimate available GPU memory in megabytes.

    Returns:
        Available VRAM in MB.  Returns ``0`` for CPU-only systems.
    """
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        mb = free // (1024 * 1024)
        logger.info("VRAM available: %d MB / %d MB total", mb, total // (1024 * 1024))
        return mb

    # MPS does not expose a direct VRAM query — return a heuristic
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon shares system RAM; assume ~50% available
        try:
            import os

            total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            mb = (total_ram // (1024 * 1024)) // 2
            logger.info("MPS estimated available: %d MB (heuristic)", mb)
            return mb
        except (ValueError, OSError):
            return 4096  # Conservative 4GB estimate

    return 0


def supports_mixed_precision() -> bool:
    """Check whether the current device supports float16 mixed precision.

    Returns:
        ``True`` if mixed precision is available and recommended.
    """
    if torch.cuda.is_available():
        # Ampere (sm_80+) and later have native FP16 tensor cores
        capability = torch.cuda.get_device_capability(0)
        supported = capability[0] >= 7  # Volta+
        logger.info(
            "Mixed precision: %s (compute capability %d.%d)",
            "supported" if supported else "not supported",
            *capability,
        )
        return supported

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS supports float16 on Apple Silicon
        return True

    return False
