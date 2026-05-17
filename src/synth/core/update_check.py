"""Automatic update checker — warns users when a newer version is available.

Queries the GitHub API for the latest commit on ``main`` and compares it
against a locally stored SHA recorded at install time.  Results are cached
for one hour to avoid excessive API calls.

The check runs in a background thread so it never blocks CLI startup.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

GITHUB_REPO = "khushalv21/SYNTH"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/commits/main"
CACHE_DIR = Path.home() / ".cache" / "synth"
STATE_FILE = CACHE_DIR / ".update_state.json"
CHECK_INTERVAL_SECONDS = 3600  # Re-check at most once per hour


# ── State management ─────────────────────────────────────────────────────────


def _read_state() -> dict:
    """Read the persisted update state (installed SHA, last check time, etc.)."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _write_state(state: dict) -> None:
    """Persist the update state to disk."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except OSError:
        pass


def _record_install_sha(sha: str) -> None:
    """Record the SHA that is currently installed."""
    state = _read_state()
    state["installed_sha"] = sha
    state["installed_at"] = time.time()
    _write_state(state)


# ── GitHub API ────────────────────────────────────────────────────────────────


def _fetch_latest_sha() -> Optional[str]:
    """Fetch the latest commit SHA from the GitHub API.

    Returns None on any network or API error (update check is best-effort).
    """
    try:
        import urllib.request

        req = urllib.request.Request(
            GITHUB_API_URL,
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "synth-update-checker",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("sha")
    except Exception:
        logger.debug("Update check failed (network error — skipping)")
        return None


# ── Core check logic ─────────────────────────────────────────────────────────


class UpdateCheckResult:
    """Result of an update check."""

    __slots__ = ("update_available", "current_sha", "latest_sha")

    def __init__(
        self,
        update_available: bool = False,
        current_sha: str = "",
        latest_sha: str = "",
    ) -> None:
        self.update_available = update_available
        self.current_sha = current_sha
        self.latest_sha = latest_sha


# Module-level container for the background check result
_bg_result: Optional[UpdateCheckResult] = None


def _do_check() -> None:
    """Run the update check (called in a background thread)."""
    global _bg_result

    state = _read_state()

    # Throttle: skip if we checked recently
    last_check = state.get("last_check_time", 0)
    if time.time() - last_check < CHECK_INTERVAL_SECONDS:
        # Use cached result
        if state.get("update_available"):
            _bg_result = UpdateCheckResult(
                update_available=True,
                current_sha=state.get("installed_sha", "")[:8],
                latest_sha=state.get("latest_sha", "")[:8],
            )
        return

    latest_sha = _fetch_latest_sha()
    if latest_sha is None:
        return  # Network error — silently skip

    installed_sha = state.get("installed_sha")

    # First time ever — record current remote SHA as "installed"
    if not installed_sha:
        _record_install_sha(latest_sha)
        state = _read_state()
        state["last_check_time"] = time.time()
        state["update_available"] = False
        state["latest_sha"] = latest_sha
        _write_state(state)
        return

    # Compare
    is_outdated = installed_sha != latest_sha

    # Persist
    state["last_check_time"] = time.time()
    state["update_available"] = is_outdated
    state["latest_sha"] = latest_sha
    _write_state(state)

    if is_outdated:
        _bg_result = UpdateCheckResult(
            update_available=True,
            current_sha=installed_sha[:8],
            latest_sha=latest_sha[:8],
        )


def start_update_check() -> None:
    """Kick off the update check in a background daemon thread.

    This is non-blocking and will never delay CLI startup.
    """
    thread = threading.Thread(target=_do_check, daemon=True)
    thread.start()


def get_update_result() -> Optional[UpdateCheckResult]:
    """Return the background check result, if available.

    Call this after the main work is done to display a warning if needed.
    Returns None if no update is available or the check hasn't completed.
    """
    return _bg_result


def print_update_warning() -> None:
    """Print a styled update warning to the console if an update is available."""
    result = get_update_result()
    if result is None or not result.update_available:
        return

    from rich.panel import Panel
    from rich.text import Text

    from synth.cli.display import console

    warning = Text()
    warning.append("  ⬆ ", style="bold yellow")
    warning.append("Update available", style="bold white")
    warning.append(f"  {result.current_sha}", style="dim red")
    warning.append(" → ", style="dim")
    warning.append(f"{result.latest_sha}", style="bold green")
    warning.append("\n  Run: ", style="dim")
    warning.append(
        "pip install --upgrade git+https://github.com/khushalv21/SYNTH.git",
        style="cyan",
    )

    console.print()
    console.print(
        Panel(
            warning,
            border_style="yellow",
            padding=(0, 1),
        )
    )


def mark_installed() -> None:
    """Record the current remote SHA as the installed version.

    Call this after a fresh install or upgrade to reset the update state.
    """
    latest_sha = _fetch_latest_sha()
    if latest_sha:
        _record_install_sha(latest_sha)
        state = _read_state()
        state["update_available"] = False
        state["latest_sha"] = latest_sha
        state["last_check_time"] = time.time()
        _write_state(state)
