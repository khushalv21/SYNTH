"""Synth CLI — AI Content Authenticator & OCR Engine.

Entry point for the ``synth`` command.  All subcommands are defined here
and delegate to :mod:`synth.core` for business logic.

Command structure::

    synth              → System dashboard (banner + info)
    synth <file>       → Auto-detect and analyse
    synth <folder>/    → Batch auto-analyse
    synth help         → Custom command menu
    synth -V           → Version
"""

from __future__ import annotations

# ── Silence libraries BEFORE any heavy imports ────────────────────────────────
# This must happen first so environment variables are set before
# transformers/torch/easyocr are imported transitively via auth.py.
import os as _os

_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import contextlib
import io
import logging
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.status import Status

from synth import __version__
from synth.cli.display import (
    build_dashboard,
    build_help_menu,
    build_results_table,
    build_summary_panel,
    build_text_panel,
    build_vision_results_table,
    build_vision_summary_panel,
    console,
    print_banner,
)
from synth.cli.silence import restore_logging, silence_libraries
from synth.core.auth import (
    AuthResult,
    DetectorFactory,
    VisionAuthResult,
    VisionAuthenticator,
)
from synth.core.exceptions import NoTextFoundError, PDFLoadError, SynthError
from synth.core.ocr import DocumentScanner, pdf_to_images
from synth.core.router import (
    IMAGE_EXTENSIONS,
    PDF_EXTENSION,
    SUPPORTED_EXTENSIONS,
    AnalysisMode,
    AnalysisModeResolver,
)

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


# ── CLI enum for --engine ─────────────────────────────────────────────────────


class EngineChoice(str, Enum):
    """Detection engine strategy."""

    local = "local"
    api = "api"


class ProfileChoice(str, Enum):
    """Operating profile — determines which detectors are loaded."""

    fast = "fast"
    balanced = "balanced"
    forensic = "forensic"


# ── App setup ─────────────────────────────────────────────────────────────────
#
# We use a FLAT @app.command() (not a Group with @app.callback) because Typer
# Groups treat extra positional args as subcommand names, which breaks both
# `synth help` and backward-compat `synth verify <file>`.
#
# allow_extra_args lets us capture old-style `synth verify <path>` gracefully.

app = typer.Typer(
    name="synth",
    help="[bold cyan]Synth[/bold cyan] — AI Content Authenticator & OCR Engine.",
    add_completion=False,
    no_args_is_help=False,
    rich_markup_mode="rich",
    add_help_option=False,
    context_settings={
        "allow_extra_args": True,
        "allow_interspersed_args": False,
    },
)


# ── Version callback ─────────────────────────────────────────────────────────


def _version_callback(value: bool) -> None:
    """Print the version and exit."""
    if value:
        console.print(
            f"[bold cyan]synth[/bold cyan] [dim]v{__version__}[/dim]"
        )
        raise typer.Exit()


# ══════════════════════════════════════════════════════════════════════════════
#  Single flat command — handles everything via keyword matching.
#
#  Keywords:  help, verify (compat), info (compat)
#  Otherwise: path → auto-detect and analyse
# ══════════════════════════════════════════════════════════════════════════════


@app.command()
def main(
    ctx: typer.Context,
    path: Optional[str] = typer.Argument(  # noqa: UP007
        None,
        help="Path to a file or directory to analyse, or 'help'.",
    ),
    version: Optional[bool] = typer.Option(  # noqa: UP007
        None,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable debug logging.",
    ),
    engine: EngineChoice = typer.Option(
        EngineChoice.local,
        "--engine",
        "-e",
        help="Detection strategy: local (HuggingFace) or api (remote).",
    ),
    profile: Optional[ProfileChoice] = typer.Option(  # noqa: UP007
        None,
        "--profile",
        "-p",
        help="Operating profile: fast, balanced (default), or forensic.",
    ),
    agent: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--agent",
        "-a",
        help="Model name or API provider override.",
    ),
    show_text: bool = typer.Option(
        True,
        "--show-text/--no-text",
        help="Show extracted text panel with highlighted AI patterns.",
    ),
    languages: str = typer.Option(
        "en",
        "--lang",
        "-l",
        help="Comma-separated OCR language codes.",
    ),
) -> None:
    """Synth — AI Content Authenticator & OCR Engine."""
    # ── Logging setup ─────────────────────────────────────────────────────
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
        )
        restore_logging()
    else:
        silence_libraries()

    # ── Background update check (non-blocking) ───────────────────────────
    from synth.core.update_check import start_update_check
    start_update_check()

    # ── No argument → show dashboard ──────────────────────────────────────
    if path is None:
        _show_dashboard()
        return

    # ── Keyword: help ─────────────────────────────────────────────────────
    if path.lower() == "help":
        panel = build_help_menu()
        console.print(panel)
        return

    # ── Keyword: models ───────────────────────────────────────────────────
    if path.lower() == "models":
        from synth.cli.display import build_models_table
        from synth.detectors import register_all
        from synth.core.auth import _register_legacy_detectors

        _register_legacy_detectors()
        register_all()
        console.print()
        console.print(build_models_table())
        console.print()
        return

    # ── Keyword: info (backward compat) ───────────────────────────────────
    if path.lower() == "info":
        console.print(
            "[dim]Note:[/dim] [yellow]'synth info'[/yellow] has been replaced. "
            "Use bare [cyan]synth[/cyan] for the dashboard.\n"
        )
        _show_dashboard()
        return

    # ── Keyword: verify (backward compat) ─────────────────────────────────
    if path.lower() == "verify":
        # Old-style: `synth verify <path>` — redirect gracefully
        if ctx.args:
            console.print(
                "[dim]Note:[/dim] [yellow]'synth verify'[/yellow] is deprecated. "
                "Use [cyan]synth <file>[/cyan] directly.\n"
            )
            path = ctx.args[0]
        else:
            console.print(
                "[bold red]✗[/bold red] Missing file path.\n"
                "  Usage: [cyan]synth <file>[/cyan]\n"
                "  Run [cyan]synth help[/cyan] for more info."
            )
            raise typer.Exit(code=1)

    # ── Resolve and validate as a filesystem path ─────────────────────────
    target = Path(path).resolve()

    if not target.exists():
        console.print(
            f"[bold red]✗[/bold red] Path not found: [yellow]{target}[/yellow]"
        )
        raise typer.Exit(code=1)

    # ── Auto-detect and analyse ───────────────────────────────────────────
    _run_auto(target, engine, agent, show_text, languages, verbose, profile)

    # ── Show update warning if available ──────────────────────────────────
    from synth.core.update_check import print_update_warning
    print_update_warning()


# ── Dashboard ─────────────────────────────────────────────────────────────────


def _show_dashboard() -> None:
    """Display the banner + system info panel."""
    print_banner()
    panel = build_dashboard()
    console.print(panel)



# ══════════════════════════════════════════════════════════════════════════════
#  Auto-detection and analysis
# ══════════════════════════════════════════════════════════════════════════════


def _collect_input_files(target: Path) -> tuple[list[Path], list[Path]]:
    """Collect image files and PDF files from *target*.

    Returns:
        A tuple of ``(image_files, pdf_files)``.
    """
    if target.is_file():
        suffix = target.suffix.lower()
        if suffix == PDF_EXTENSION:
            return [], [target]
        if suffix in IMAGE_EXTENSIONS:
            return [target], []
        console.print(
            f"[bold red]✗[/bold red] Unsupported file type: "
            f"[yellow]{target.suffix}[/yellow]. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        raise typer.Exit(code=1)

    if target.is_dir():
        images = sorted(
            f
            for f in target.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        pdfs = sorted(
            f
            for f in target.rglob("*")
            if f.is_file() and f.suffix.lower() == PDF_EXTENSION
        )
        if not images and not pdfs:
            console.print(
                f"[bold red]✗[/bold red] No supported files found in "
                f"[yellow]{target}[/yellow]"
            )
            raise typer.Exit(code=1)
        return images, pdfs

    # Defensive fallback — should be unreachable since path.exists()
    # is checked in main(), but kept for robustness.
    console.print(
        f"[bold red]✗[/bold red] Path not found: [yellow]{target}[/yellow]"
    )
    raise typer.Exit(code=1)


def _extract_pdf_pages(
    pdf_path: Path,
) -> list[tuple[str, "np.ndarray"]]:
    """Convert a PDF into labeled page arrays with a rich progress bar.

    Returns:
        A list of ``(label, bgr_array)`` tuples, one per page.
    """
    try:
        page_images = pdf_to_images(pdf_path)
    except PDFLoadError as exc:
        console.print(
            f"  [bold red]✗[/bold red] Failed to load PDF: {exc}"
        )
        return []

    return [
        (f"{pdf_path.stem}_page{i + 1}", img)
        for i, img in enumerate(page_images)
    ]


def _process_pdf_page(
    label: str,
    page_array: "np.ndarray",
    scanner: DocumentScanner,
    detector: object,
) -> tuple[str, str, AuthResult] | None:
    """OCR + detect on a single PDF page array. Returns (label, text, result) or None."""
    try:
        text = scanner.extract_text_from_array(page_array, label=label)
    except NoTextFoundError:
        console.print(
            f"  [dim yellow]⚠ {label}:[/dim yellow] No readable text found — skipped"
        )
        return None
    except Exception as exc:
        console.print(
            f"  [dim red]✗ {label}:[/dim red] OCR failed — {exc}"
        )
        return None

    try:
        result = detector.detect(text)  # type: ignore[union-attr]
    except SynthError as exc:
        console.print(
            f"  [dim red]✗ {label}:[/dim red] Detection failed — {exc}"
        )
        return None

    return label, text, result


def _process_single_file(
    image_path: Path,
    scanner: DocumentScanner,
    detector: object,
    show_text: bool,
) -> tuple[str, str, AuthResult] | None:
    """OCR + detect on a single file. Returns (filename, text, result) or None on error."""
    filename = image_path.name

    try:
        text = scanner.extract_text(image_path)
    except NoTextFoundError:
        console.print(
            f"  [dim yellow]⚠ {filename}:[/dim yellow] No readable text found — skipped"
        )
        return None
    except Exception as exc:
        console.print(
            f"  [dim red]✗ {filename}:[/dim red] OCR failed — {exc}"
        )
        return None

    try:
        result = detector.detect(text)  # type: ignore[union-attr]
    except SynthError as exc:
        console.print(
            f"  [dim red]✗ {filename}:[/dim red] Detection failed — {exc}"
        )
        return None

    return filename, text, result


# ── Unified auto-detect pipeline ──────────────────────────────────────────────


def _is_first_run(lang_list: list[str]) -> bool:
    """Detect whether this is the first run by checking model caches.

    Returns True if EasyOCR language data or the HuggingFace model
    has not been downloaded yet.
    """
    # Check EasyOCR model cache
    easyocr_cache = Path.home() / ".EasyOCR" / "model"
    has_easyocr = easyocr_cache.exists() and any(easyocr_cache.iterdir()) if easyocr_cache.exists() else False

    # Check HuggingFace model cache (roberta-base-openai-detector)
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    has_hf = hf_cache.exists() and any(hf_cache.iterdir()) if hf_cache.exists() else False

    return not (has_easyocr and has_hf)


def _premium_first_run_init(
    lang_list: list[str],
    verbose: bool,
) -> "DocumentScanner":
    """Premium first-run installation experience.

    Shows a single, unified progress bar with phased steps instead of
    dumping raw download output from EasyOCR and HuggingFace.
    """
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    # ── Premium welcome header ────────────────────────────────────────
    console.print()
    welcome_text = Text()
    welcome_text.append("  ◆ ", style="bold cyan")
    welcome_text.append("First-time setup", style="bold white")
    welcome_text.append(" — ", style="dim")
    welcome_text.append("downloading AI models", style="dim italic")

    console.print(
        Panel(
            Group(
                welcome_text,
                Text("  This only happens once. Models are cached locally for future runs.", style="dim"),
            ),
            border_style="cyan",
            padding=(0, 1),
        )
    )
    console.print()

    # ── Phased progress bar ───────────────────────────────────────────
    phases = [
        ("Preparing environment",        5),
        ("Downloading OCR engine",      35),
        ("Loading language models",     25),
        ("Initialising AI pipeline",    20),
        ("Optimising for your device",  15),
    ]
    total_weight = sum(w for _, w in phases)

    progress = Progress(
        SpinnerColumn(style="cyan", spinner_name="dots12"),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(
            bar_width=50,
            style="dim white",
            complete_style="bold cyan",
            finished_style="bold green",
            pulse_style="cyan",
        ),
        TaskProgressColumn(
            text_format="[bold]{task.percentage:>3.0f}%[/bold]",
        ),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    scanner = None
    error = None

    with progress:
        task = progress.add_task(phases[0][0], total=total_weight)

        # Phase 1: Preparing environment
        progress.update(task, description=f"[cyan]›[/cyan] {phases[0][0]}")
        time.sleep(0.3)
        progress.advance(task, phases[0][1])

        # Phase 2–4: The actual heavy work — load OCR engine
        # (EasyOCR downloads ~100MB of models on first run)
        progress.update(task, description=f"[cyan]›[/cyan] {phases[1][0]}")

        def _load_scanner() -> None:
            nonlocal scanner, error
            try:
                if verbose:
                    scanner = DocumentScanner(languages=lang_list)
                else:
                    with contextlib.redirect_stdout(io.StringIO()), \
                         contextlib.redirect_stderr(io.StringIO()):
                        scanner = DocumentScanner(languages=lang_list)
            except Exception as exc:
                error = exc

        loader_thread = threading.Thread(target=_load_scanner, daemon=True)
        loader_thread.start()

        # Animate progress while the actual download/load happens
        phase_weights = [phases[1][1], phases[2][1], phases[3][1]]
        phase_labels = [phases[1][0], phases[2][0], phases[3][0]]
        combined_weight = sum(phase_weights)
        elapsed_weight = 0.0

        while loader_thread.is_alive():
            time.sleep(0.15)
            # Advance smoothly but cap at 95% of combined weight
            if elapsed_weight < combined_weight * 0.95:
                increment = min(0.8, combined_weight * 0.95 - elapsed_weight)
                progress.advance(task, increment)
                elapsed_weight += increment

                # Update phase label based on progress
                ratio = elapsed_weight / combined_weight
                if ratio < 0.40:
                    progress.update(task, description=f"[cyan]›[/cyan] {phase_labels[0]}")
                elif ratio < 0.75:
                    progress.update(task, description=f"[cyan]›[/cyan] {phase_labels[1]}")
                else:
                    progress.update(task, description=f"[cyan]›[/cyan] {phase_labels[2]}")

        loader_thread.join()

        # Fill remaining weight for phases 2–4
        remaining = combined_weight - elapsed_weight
        if remaining > 0:
            progress.advance(task, remaining)

        # Phase 5: Optimising
        progress.update(task, description=f"[cyan]›[/cyan] {phases[4][0]}")
        time.sleep(0.2)
        progress.advance(task, phases[4][1])

        # Final state
        progress.update(task, description="[bold green]✓[/bold green] [bold]Setup complete[/bold]")

    if error is not None:
        console.print(
            f"\n[bold red]✗[/bold red] Failed to load OCR engine: {error}"
        )
        raise typer.Exit(code=1)
    # Record current SHA so the update checker has a baseline
    from synth.core.update_check import mark_installed
    threading.Thread(target=mark_installed, daemon=True).start()

    console.print()
    return scanner  # type: ignore[return-value]


def _quick_init(
    lang_list: list[str],
    verbose: bool,
) -> "DocumentScanner":
    """Fast model initialisation for subsequent runs (models already cached)."""
    with Status(
        "[bold cyan]Initialising models…[/bold cyan]",
        spinner="dots",
        console=console,
    ):
        try:
            if verbose:
                scanner = DocumentScanner(languages=lang_list)
            else:
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    scanner = DocumentScanner(languages=lang_list)
        except Exception as exc:
            console.print(
                f"[bold red]✗[/bold red] Failed to load OCR engine: {exc}"
            )
            raise typer.Exit(code=1)
    return scanner


def _run_auto(
    path: Path,
    engine: EngineChoice,
    agent: str | None,
    show_text: bool,
    languages: str,
    verbose: bool = False,
    profile: ProfileChoice | None = None,
) -> None:
    """Unified entry point: collect files → auto-detect mode → run pipeline."""
    # ── Collect files ─────────────────────────────────────────────────────
    image_files, pdf_files = _collect_input_files(path)

    # ── Initialise OCR engine (needed for both probe and text mode) ───────
    lang_list = [lang.strip() for lang in languages.split(",")]

    first_run = _is_first_run(lang_list)

    if first_run:
        scanner = _premium_first_run_init(lang_list, verbose)
    else:
        scanner = _quick_init(lang_list, verbose)

    # ── Auto-detect mode for image files ──────────────────────────────────
    #
    # NOTE: Images routed to text mode will be OCR'd twice — once here
    # (lightweight probe, no preprocessing) and again in the full text
    # pipeline (with preprocessing).  The probe is ~200-500ms and reuses
    # the already-loaded EasyOCR reader.  This is an acceptable tradeoff
    # for fully automatic mode detection.
    resolver = AnalysisModeResolver()

    text_images: list[Path] = []
    vision_images: list[Path] = []

    if image_files:
        with Status(
            "[bold cyan]Detecting analysis mode…[/bold cyan]",
            spinner="dots",
            console=console,
        ):
            for img_path in image_files:
                mode = resolver.resolve_file(img_path, scanner=scanner)
                if mode == AnalysisMode.text:
                    text_images.append(img_path)
                else:
                    vision_images.append(img_path)

    # PDFs are always text mode
    # text_images + pdf_files → text pipeline
    # vision_images → image pipeline

    has_text_work = len(text_images) > 0 or len(pdf_files) > 0
    has_vision_work = len(vision_images) > 0

    # ══════════════════════════════════════════════════════════════════════
    #  Ensemble mode (--profile)
    # ══════════════════════════════════════════════════════════════════════
    if profile is not None:
        _run_ensemble_mode(
            text_images, pdf_files, vision_images, scanner,
            profile.value, show_text, verbose,
        )
        return

    # ══════════════════════════════════════════════════════════════════════
    #  Legacy mode (--engine)
    # ══════════════════════════════════════════════════════════════════════

    # ── Run text pipeline ─────────────────────────────────────────────────
    text_results: list[tuple[str, AuthResult]] = []
    texts: dict[str, str] = {}

    if has_text_work:
        text_results, texts = _run_text_pipeline(
            text_images, pdf_files, scanner, engine, agent, show_text
        )

    # ── Run vision pipeline ───────────────────────────────────────────────
    vision_results: list[tuple[str, VisionAuthResult]] = []

    if has_vision_work:
        vision_results = _run_vision_pipeline(vision_images, agent)

    # ── Display results ───────────────────────────────────────────────────
    total_results = len(text_results) + len(vision_results)
    if total_results == 0:
        console.print(
            "[bold red]✗[/bold red] No files could be processed successfully."
        )
        raise typer.Exit(code=1)

    # Section headers when both pipelines produced results
    has_both = bool(text_results) and bool(vision_results)

    if text_results:
        if has_both:
            console.print("[bold]\n📝 Text Analysis Results[/bold]\n")
        table = build_results_table(text_results)
        console.print(table)
        console.print()

        if show_text:
            for filename, result in text_results:
                if filename in texts:
                    panel = build_text_panel(
                        texts[filename], result, filename=filename
                    )
                    console.print(panel)
                    console.print()

        if len(text_results) > 1:
            summary = build_summary_panel(text_results)
            console.print(summary)

    if vision_results:
        if has_both:
            console.print("[bold]\n🔍 Image Forensics Results[/bold]\n")
        table = build_vision_results_table(vision_results)
        console.print(table)
        console.print()

        if len(vision_results) > 1:
            summary = build_vision_summary_panel(vision_results)
            console.print(summary)

    # ── Exit code ─────────────────────────────────────────────────────────
    ai_text = any(r.verdict == "ai" for _, r in text_results)
    ai_image = any(r.verdict == "fake" for _, r in vision_results)
    if ai_text or ai_image:
        sys.exit(1)


# ── Ensemble mode pipeline ────────────────────────────────────────────────────


def _run_ensemble_mode(
    text_images: list[Path],
    pdf_files: list[Path],
    vision_images: list[Path],
    scanner: DocumentScanner,
    profile: str,
    show_text: bool,
    verbose: bool,
) -> None:
    """Run the multi-detector ensemble pipeline.

    Uses :class:`~synth.core.manager.MultiDetectorManager` to load
    the appropriate detectors for the given profile and aggregate
    results via ensemble voting.
    """
    from synth.cli.display import build_ensemble_results_table, build_ensemble_summary_panel
    from synth.core.auth import _register_legacy_detectors
    from synth.core.manager import MultiDetectorManager

    _register_legacy_detectors()

    console.print(
        f"\n[bold cyan]⚡ Ensemble mode[/bold cyan] · "
        f"profile=[bold]{profile}[/bold]\n"
    )

    mgr = MultiDetectorManager(profile=profile)
    any_ai = False

    # ── Text analysis (images with text + PDFs) ───────────────────────────
    has_text_work = len(text_images) > 0 or len(pdf_files) > 0

    if has_text_work:
        import numpy as np

        # Extract text from images
        for img_path in text_images:
            try:
                text = scanner.scan_to_text(str(img_path))
                if not text.strip():
                    continue

                with Status(
                    f"[bold cyan]Analysing text: {img_path.name}…[/bold cyan]",
                    spinner="dots",
                    console=console,
                ):
                    result = mgr.detect_text(text)

                console.print(build_ensemble_results_table(result, filename=img_path.name))
                console.print()
                console.print(build_ensemble_summary_panel(result, filename=img_path.name))
                console.print()

                if result.consensus_verdict in ("ai", "fake"):
                    any_ai = True

            except Exception as exc:
                console.print(
                    f"[bold red]✗[/bold red] Failed: {img_path.name}: {exc}"
                )

        # Extract text from PDFs
        from synth.core.ocr import pdf_to_images

        for pdf_path in pdf_files:
            try:
                pages = pdf_to_images(str(pdf_path))
                full_text = ""
                for idx, page_arr in enumerate(pages):
                    page_text = scanner.scan_to_text(page_arr)
                    full_text += page_text + "\n"

                if not full_text.strip():
                    continue

                with Status(
                    f"[bold cyan]Analysing text: {pdf_path.name}…[/bold cyan]",
                    spinner="dots",
                    console=console,
                ):
                    result = mgr.detect_text(full_text)

                console.print(build_ensemble_results_table(result, filename=pdf_path.name))
                console.print()
                console.print(build_ensemble_summary_panel(result, filename=pdf_path.name))
                console.print()

                if result.consensus_verdict in ("ai", "fake"):
                    any_ai = True

            except Exception as exc:
                console.print(
                    f"[bold red]✗[/bold red] Failed: {pdf_path.name}: {exc}"
                )

    # ── Image forensics ───────────────────────────────────────────────────
    if vision_images:
        for img_path in vision_images:
            try:
                with Status(
                    f"[bold cyan]Analysing image: {img_path.name}…[/bold cyan]",
                    spinner="dots",
                    console=console,
                ):
                    result = mgr.detect_image(img_path)

                console.print(build_ensemble_results_table(result, filename=img_path.name))
                console.print()
                console.print(build_ensemble_summary_panel(result, filename=img_path.name))
                console.print()

                if result.consensus_verdict in ("ai", "fake"):
                    any_ai = True

            except Exception as exc:
                console.print(
                    f"[bold red]✗[/bold red] Failed: {img_path.name}: {exc}"
                )

    # ── Cleanup ───────────────────────────────────────────────────────────
    mgr.unload_all()

    if any_ai:
        sys.exit(1)


# ── Text pipeline ─────────────────────────────────────────────────────────────


def _run_text_pipeline(
    image_files: list[Path],
    pdf_files: list[Path],
    scanner: DocumentScanner,
    engine: EngineChoice,
    agent: str | None,
    show_text: bool,
) -> tuple[list[tuple[str, AuthResult]], dict[str, str]]:
    """Run OCR + text detection on images and PDFs.

    Returns:
        ``(results, texts)`` — list of (name, AuthResult) and dict of extracted texts.
    """
    import numpy as np  # Deferred — only needed in text pipeline

    # ── Extract PDF pages ─────────────────────────────────────────────────
    pdf_pages: list[tuple[str, np.ndarray]] = []

    if pdf_files:
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(bar_width=40, style="cyan", complete_style="bold cyan"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                "Extracting PDF pages…", total=len(pdf_files)
            )

            for pdf_path_item in pdf_files:
                progress.update(
                    task,
                    description=(
                        f"Extracting PDF pages… "
                        f"[cyan]{pdf_path_item.name}[/cyan]"
                    ),
                )
                pages = _extract_pdf_pages(pdf_path_item)
                pdf_pages.extend(pages)
                progress.advance(task)

        if pdf_pages:
            console.print(
                f"[bold green]✓[/bold green] Extracted "
                f"[cyan]{len(pdf_pages)}[/cyan] page(s) from "
                f"{len(pdf_files)} PDF(s)\n"
            )

    # ── Load detection model ──────────────────────────────────────────────
    detector_kwargs: dict[str, object] = {}
    if agent:
        if engine == EngineChoice.local:
            detector_kwargs["model_name"] = agent

    with Status(
        "[bold cyan]Loading detection model…[/bold cyan]",
        spinner="dots",
        console=console,
    ):
        detector = DetectorFactory.create(engine.value, **detector_kwargs)

    console.print("[bold green]✓[/bold green] Models loaded\n")

    # ── Process files ─────────────────────────────────────────────────────
    results: list[tuple[str, AuthResult]] = []
    texts: dict[str, str] = {}

    total_work_items = len(image_files) + len(pdf_pages)
    is_batch = total_work_items > 1

    if is_batch:
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(bar_width=40, style="cyan", complete_style="bold cyan"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Scanning files…", total=total_work_items)

            for image_path in image_files:
                progress.update(
                    task,
                    description=f"Processing [cyan]{image_path.name}[/cyan]",
                )

                outcome = _process_single_file(
                    image_path, scanner, detector, show_text
                )

                if outcome is not None:
                    filename, text, result = outcome
                    results.append((filename, result))
                    texts[filename] = text

                progress.advance(task)

            for label, page_array in pdf_pages:
                progress.update(
                    task,
                    description=f"Processing [cyan]{label}[/cyan]",
                )

                outcome = _process_pdf_page(
                    label, page_array, scanner, detector
                )

                if outcome is not None:
                    page_label, text, result = outcome
                    results.append((page_label, result))
                    texts[page_label] = text

                progress.advance(task)

        console.print()

    else:
        # Single file
        if image_files:
            image_path = image_files[0]
            with Status(
                f"[bold cyan]Analysing [white]{image_path.name}[/white]…[/bold cyan]",
                spinner="dots",
                console=console,
            ):
                outcome = _process_single_file(
                    image_path, scanner, detector, show_text
                )

            if outcome is not None:
                filename, text, result = outcome
                results.append((filename, result))
                texts[filename] = text

        elif pdf_pages:
            label, page_array = pdf_pages[0]
            with Status(
                f"[bold cyan]Analysing [white]{label}[/white]…[/bold cyan]",
                spinner="dots",
                console=console,
            ):
                outcome = _process_pdf_page(
                    label, page_array, scanner, detector
                )

            if outcome is not None:
                page_label, text, result = outcome
                results.append((page_label, result))
                texts[page_label] = text

    return results, texts


# ── Vision pipeline ───────────────────────────────────────────────────────────


def _run_vision_pipeline(
    image_files: list[Path],
    agent: str | None,
) -> list[tuple[str, VisionAuthResult]]:
    """Run ViT image forensics on a list of images.

    Returns:
        List of ``(filename, VisionAuthResult)`` tuples.
    """
    vision_kwargs: dict[str, str] = {}
    if agent:
        vision_kwargs["model_name"] = agent

    with Status(
        "[bold cyan]Loading vision model…[/bold cyan]",
        spinner="dots",
        console=console,
    ):
        vision = VisionAuthenticator(**vision_kwargs)

    console.print(
        f"[bold green]✓[/bold green] Vision model loaded "
        f"([cyan]{vision.name}[/cyan])\n"
    )

    results: list[tuple[str, VisionAuthResult]] = []
    is_batch = len(image_files) > 1

    if is_batch:
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(bar_width=40, style="cyan", complete_style="bold cyan"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                "Analysing images…", total=len(image_files)
            )

            for img_path in image_files:
                progress.update(
                    task,
                    description=f"Analysing [cyan]{img_path.name}[/cyan]",
                )

                try:
                    result = vision.detect_file(img_path)
                    results.append((img_path.name, result))
                except Exception as exc:
                    console.print(
                        f"  [dim red]✗ {img_path.name}:[/dim red] {exc}"
                    )

                progress.advance(task)

        console.print()

    else:
        img_path = image_files[0]

        with Status(
            f"[bold cyan]Analysing [white]{img_path.name}[/white]…[/bold cyan]",
            spinner="dots",
            console=console,
        ):
            try:
                result = vision.detect_file(img_path)
                results.append((img_path.name, result))
            except Exception as exc:
                console.print(
                    f"[bold red]✗[/bold red] Vision analysis failed: {exc}"
                )
                raise typer.Exit(code=1)

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
