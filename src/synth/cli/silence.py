"""Suppress noisy external library output for a clean CLI experience.

Centralises all logging/warning suppression so the rest of the codebase
stays clean.  Call :func:`silence_libraries` once at CLI startup (before
any model loading) to mute transformers, torch, easyocr, huggingface_hub,
PIL, and other chatty libraries.

When ``--verbose`` is active, call :func:`restore_logging` instead to
re-enable full debug output.
"""

from __future__ import annotations

import logging
import os
import warnings


# Libraries whose loggers we silence at startup
_NOISY_LOGGERS = [
    "transformers",
    "transformers.modeling_utils",
    "transformers.configuration_utils",
    "transformers.tokenization_utils_base",
    "torch",
    "torch.nn.modules",
    "easyocr",
    "PIL",
    "PIL.PngImagePlugin",
    "PIL.Image",
    "httpx",
    "httpcore",
    "huggingface_hub",
    "huggingface_hub.file_download",
    "filelock",
    "fsspec",
    "urllib3",
]


def silence_libraries() -> None:
    """Mute all external library logging, warnings, and progress bars.

    Safe to call multiple times.  Sets environment variables *before*
    the libraries are imported so they pick up the config at init time.
    """
    # ── Environment variables (must be set before import) ─────────────
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # ── Suppress Python warnings ──────────────────────────────────────
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # ── Silence named loggers ─────────────────────────────────────────
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    # ── Transformers has its own verbosity system ─────────────────────
    try:
        import transformers  # noqa: F811

        transformers.logging.set_verbosity_error()
    except ImportError:
        pass


def restore_logging() -> None:
    """Restore normal logging for ``--verbose`` mode.

    Resets all suppressed loggers to DEBUG and re-enables warnings.
    """
    warnings.resetwarnings()

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG)

    try:
        import transformers

        transformers.logging.set_verbosity_debug()
    except ImportError:
        pass
