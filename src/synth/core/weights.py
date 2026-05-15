"""Runtime weight download manager.

Downloads model weights on first use, caches them in
``~/.cache/synth/``, and validates integrity with SHA-256 checksums.
Shows a Rich progress bar during download.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Default cache directory ───────────────────────────────────────────────────

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "synth"


class WeightManager:
    """Download, cache, and validate model weight files.

    All weights are stored under ``~/.cache/synth/<name>/``.
    If a cached file exists and its checksum matches, no download occurs.

    Usage::

        path = WeightManager.ensure_weights(
            name="bnn-tiny",
            url="https://github.com/.../tiny_checkpoint.pth.tar",
            filename="tiny_checkpoint.pth.tar",
            checksum="sha256:abc123...",
        )
    """

    CACHE_DIR: Path = DEFAULT_CACHE_DIR

    @classmethod
    def ensure_weights(
        cls,
        *,
        name: str,
        url: str,
        filename: str,
        checksum: str = "",
    ) -> Path:
        """Return the local path to a weight file, downloading if needed.

        Args:
            name: Detector/model name (used as subdirectory).
            url: Remote URL to download the weights from.
            filename: Local filename for the cached weights.
            checksum: Optional ``sha256:<hex>`` string for validation.

        Returns:
            Absolute path to the cached weight file.

        Raises:
            RuntimeError: If the download fails or checksum mismatches.
        """
        cache_subdir = cls.CACHE_DIR / name
        cache_subdir.mkdir(parents=True, exist_ok=True)
        dest = cache_subdir / filename

        # Already cached — verify checksum if provided
        if dest.exists():
            if checksum and not cls.verify_checksum(dest, checksum):
                logger.warning(
                    "Checksum mismatch for %s — re-downloading", dest
                )
                dest.unlink()
            else:
                logger.debug("Weights cached: %s", dest)
                return dest

        # Download
        logger.info("Downloading weights: %s → %s", url, dest)
        cls._download_with_progress(url, dest)

        # Post-download checksum validation
        if checksum and not cls.verify_checksum(dest, checksum):
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded weights for '{name}' failed checksum validation. "
                f"Expected: {checksum}"
            )

        return dest

    @classmethod
    def _download_with_progress(cls, url: str, dest: Path) -> None:
        """Download a file with a Rich progress bar."""
        try:
            from rich.progress import (
                BarColumn,
                DownloadColumn,
                Progress,
                TextColumn,
                TimeRemainingColumn,
                TransferSpeedColumn,
            )

            with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))

                with Progress(
                    TextColumn("[cyan]Downloading[/cyan]"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                ) as progress:
                    task = progress.add_task("download", total=total or None)

                    # Write to a temp file, then atomic-move
                    tmp = dest.with_suffix(".tmp")
                    try:
                        with open(tmp, "wb") as f:
                            for chunk in response.iter_bytes(chunk_size=65536):
                                f.write(chunk)
                                progress.advance(task, len(chunk))
                        shutil.move(str(tmp), str(dest))
                    except Exception:
                        tmp.unlink(missing_ok=True)
                        raise

        except ImportError:
            # Rich not available — fall back to plain download
            cls._download_plain(url, dest)

    @classmethod
    def _download_plain(cls, url: str, dest: Path) -> None:
        """Simple download without progress bar (fallback)."""
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            tmp = dest.with_suffix(".tmp")
            try:
                with open(tmp, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        f.write(chunk)
                shutil.move(str(tmp), str(dest))
            except Exception:
                tmp.unlink(missing_ok=True)
                raise

    @classmethod
    def verify_checksum(cls, path: Path, expected: str) -> bool:
        """Verify a file's SHA-256 checksum.

        Args:
            path: Path to the file.
            expected: Checksum string in ``sha256:<hex>`` format,
                or just a raw hex digest.

        Returns:
            ``True`` if the checksum matches.
        """
        if not path.exists():
            return False

        # Strip prefix
        digest_expected = expected
        if expected.startswith("sha256:"):
            digest_expected = expected[7:]

        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)

        actual = sha.hexdigest()
        match = actual == digest_expected
        if not match:
            logger.debug(
                "Checksum mismatch: expected %s, got %s",
                digest_expected[:16] + "...",
                actual[:16] + "...",
            )
        return match

    @classmethod
    def cache_dir_for(cls, name: str) -> Path:
        """Return the cache subdirectory for a detector name."""
        d = cls.CACHE_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def is_cached(cls, name: str, filename: str) -> bool:
        """Check whether a weight file is already cached."""
        return (cls.CACHE_DIR / name / filename).exists()

    @classmethod
    def cached_path(cls, name: str, filename: str) -> Path | None:
        """Return the cached path if it exists, else ``None``."""
        p = cls.CACHE_DIR / name / filename
        return p if p.exists() else None
