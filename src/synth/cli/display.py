"""Rich terminal UI components for Synth.

Reusable display helpers: banner, result tables, text panels,
and progress bars. Keeps the main CLI module focused on command logic.

All user-facing strings are written in plain, friendly English so that
anyone — not just developers — can understand the output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from synth.core.auth import AuthResult, VisionAuthResult

console = Console()

# ── ASCII Banner ──────────────────────────────────────────────────────────────

BANNER = r"""
[bold cyan]  ███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗[/bold cyan]
[bold cyan]  ██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║[/bold cyan]
[bold cyan]  ███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║[/bold cyan]
[bold cyan]  ╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║[/bold cyan]
[bold cyan]  ███████║   ██║   ██║ ╚████║   ██║   ██║  ██║[/bold cyan]
[bold cyan]  ╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝[/bold cyan]
[dim]  Check if text or images were made by AI[/dim]
"""


def print_banner() -> None:
    """Print the stylised SYNTH ASCII banner."""
    console.print(BANNER)


# ── Verdict helpers (plain English + emoji) ───────────────────────────────────

_VERDICT_LABELS: dict[str, str] = {
    "human": "✅ Written by a human",
    "ai":    "🤖 Likely AI-generated",
    "mixed": "⚠️  Possibly AI-assisted",
    "real":  "✅ Real photo",
    "fake":  "🤖 Likely AI-generated",
}

_VERDICT_STYLES: dict[str, str] = {
    "human": "bold green",
    "ai": "bold red",
    "mixed": "bold yellow",
    "real": "bold green",
    "fake": "bold red",
}


def _verdict_styled(verdict: str) -> str:
    """Return a Rich-markup-wrapped verdict in plain English."""
    style = _VERDICT_STYLES.get(verdict, "bold white")
    label = _VERDICT_LABELS.get(verdict, verdict.upper())
    return f"[{style}]{label}[/{style}]"


def _verdict_short(verdict: str) -> str:
    """Short emoji-only verdict for table rows."""
    short_map = {
        "human": "[bold green]✅ Human[/bold green]",
        "ai": "[bold red]🤖 AI[/bold red]",
        "mixed": "[bold yellow]⚠️  Mixed[/bold yellow]",
        "real": "[bold green]✅ Real[/bold green]",
        "fake": "[bold red]🤖 AI[/bold red]",
    }
    return short_map.get(verdict, verdict)


def _confidence_label(score: float) -> str:
    """Convert a raw 0.0–1.0 score into a human-readable confidence level with percentage."""
    pct = f"{score * 100:.0f}%"
    if score >= 0.90:
        return f"[bold red]{pct} · Very High[/bold red]"
    if score >= 0.75:
        return f"[red]{pct} · High[/red]"
    if score >= 0.50:
        return f"[yellow]{pct} · Moderate[/yellow]"
    if score >= 0.30:
        return f"[yellow]{pct} · Low[/yellow]"
    return f"[green]{pct} · Very Low[/green]"


def _score_styled(score: float) -> str:
    """Return colour-coded percentage: green ≤0.3, yellow 0.3–0.7, red ≥0.7."""
    pct = f"{score * 100:.0f}%"
    if score <= 0.3:
        return f"[green]{pct}[/green]"
    if score <= 0.7:
        return f"[yellow]{pct}[/yellow]"
    return f"[red]{pct}[/red]"


# ── Results table (text mode) ────────────────────────────────────────────────


def build_results_table(
    results: list[tuple[str, AuthResult]],
) -> Table:
    """Build a Rich table summarising scan results.

    Args:
        results: List of ``(filename, AuthResult)`` tuples.

    Returns:
        A styled :class:`rich.table.Table`.
    """
    table = Table(
        title="[bold]📋 Scan Results[/bold]",
        title_style="cyan",
        border_style="dim cyan",
        header_style="bold white",
        row_styles=["", "dim"],
        expand=True,
        padding=(0, 1),
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("File", style="white", ratio=3)
    table.add_column("Result", justify="center", width=12)
    table.add_column("Confidence", justify="center", width=14)

    for idx, (filename, result) in enumerate(results, 1):
        table.add_row(
            str(idx),
            filename,
            _verdict_short(result.verdict),
            _confidence_label(result.score),
        )

    return table


# ── Text panel ────────────────────────────────────────────────────────────────

_HIGHLIGHT_PATTERNS = [
    "however",
    "furthermore",
    "moreover",
    "in conclusion",
    "it is important to note",
    "significantly",
    "consequently",
    "as a result",
    "in summary",
    "overall",
    "notably",
    "essentially",
]


def build_text_panel(
    text: str,
    result: AuthResult,
    filename: str = "",
    max_chars: int = 800,
) -> Panel:
    """Build a Rich panel showing extracted text with AI patterns highlighted.

    Common AI-associated connector words are highlighted in red to give
    the user a visual sense of synthetic patterning.

    Args:
        text: The extracted text content.
        result: The detection result.
        filename: Optional filename for the panel title.
        max_chars: Truncate text beyond this limit.
    """
    # Truncate long texts
    display_text = text[:max_chars]
    if len(text) > max_chars:
        display_text += f"\n[dim]… ({len(text) - max_chars} more characters)[/dim]"

    # Highlight synthetic patterns
    rich_text = Text(display_text)
    for pattern in _HIGHLIGHT_PATTERNS:
        rich_text.highlight_words(
            [pattern, pattern.capitalize(), pattern.upper()],
            style="bold red underline",
        )

    # Build subtitle in plain English
    verdict_display = _verdict_styled(result.verdict)
    confidence = _confidence_label(result.score)
    subtitle = f"{verdict_display}  │  Confidence: {confidence}"

    title = f"[bold cyan]📄 {filename}[/bold cyan]" if filename else "[bold cyan]Extracted Text[/bold cyan]"

    return Panel(
        Align.left(rich_text),
        title=title,
        subtitle=subtitle,
        border_style="cyan",
        padding=(1, 2),
        expand=True,
    )


# ── Summary panel (text mode) ────────────────────────────────────────────────


def build_summary_panel(
    results: list[tuple[str, AuthResult]],
) -> Panel:
    """Build a summary panel with aggregate statistics in plain language."""
    total = len(results)
    ai_count = sum(1 for _, r in results if r.verdict == "ai")
    human_count = sum(1 for _, r in results if r.verdict == "human")
    mixed_count = sum(1 for _, r in results if r.verdict == "mixed")

    lines = [
        f"  Checked [bold]{total}[/bold] file{'s' if total != 1 else ''}",
        "",
    ]
    if human_count:
        lines.append(f"  [bold green]✅ {human_count}[/bold green] look{'s' if human_count == 1 else ''} human-written")
    if ai_count:
        lines.append(f"  [bold red]🤖 {ai_count}[/bold red] look{'s' if ai_count == 1 else ''} AI-generated")
    if mixed_count:
        lines.append(f"  [bold yellow]⚠️  {mixed_count}[/bold yellow] could be a mix of human and AI")

    return Panel(
        "\n".join(lines),
        title="[bold cyan]📊 Summary[/bold cyan]",
        border_style="cyan",
        padding=(1, 1),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Vision mode display (AI image forensics)
# ══════════════════════════════════════════════════════════════════════════════


def build_vision_results_table(
    results: list[tuple[str, VisionAuthResult]],
) -> Table:
    """Build a Rich table for AI image detection results."""
    table = Table(
        title="[bold]🖼️  Image Scan Results[/bold]",
        title_style="cyan",
        border_style="dim cyan",
        header_style="bold white",
        row_styles=["", "dim"],
        expand=True,
        padding=(0, 1),
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("File", style="white", ratio=3)
    table.add_column("Result", justify="center", width=12)
    table.add_column("Confidence", justify="center", width=14)

    for idx, (filename, result) in enumerate(results, 1):
        table.add_row(
            str(idx),
            filename,
            _verdict_short(result.verdict),
            _confidence_label(result.ai_probability),
        )

    return table


def build_vision_summary_panel(
    results: list[tuple[str, VisionAuthResult]],
) -> Panel:
    """Build a summary panel for vision-mode batch results."""
    total = len(results)
    real_count = sum(1 for _, r in results if r.verdict == "real")
    fake_count = sum(1 for _, r in results if r.verdict == "fake")

    lines = [
        f"  Checked [bold]{total}[/bold] image{'s' if total != 1 else ''}",
        "",
    ]
    if real_count:
        lines.append(f"  [bold green]✅ {real_count}[/bold green] look{'s' if real_count == 1 else ''} like real photo{'s' if real_count != 1 else ''}")
    if fake_count:
        lines.append(f"  [bold red]🤖 {fake_count}[/bold red] look{'s' if fake_count == 1 else ''} AI-generated")

    return Panel(
        "\n".join(lines),
        title="[bold cyan]📊 Image Summary[/bold cyan]",
        border_style="cyan",
        padding=(1, 1),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Ensemble mode display (multi-detector results)
# ══════════════════════════════════════════════════════════════════════════════


def build_ensemble_results_table(
    result: object,
    *,
    filename: str = "",
) -> Table:
    """Build a Rich table showing per-detector votes from an ensemble result.

    Args:
        result: An :class:`~synth.core.ensemble.EnsembleResult` instance.
        filename: Optional filename for the table title.
    """
    from synth.core.ensemble import EnsembleResult

    assert isinstance(result, EnsembleResult)

    title = f"[bold]🔬 Deep Scan — {filename}[/bold]" if filename else "[bold]🔬 Deep Scan[/bold]"
    table = Table(
        title=title,
        title_style="cyan",
        border_style="dim cyan",
        header_style="bold white",
        row_styles=["", "dim"],
        expand=True,
        padding=(0, 1),
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Scanner", style="white", ratio=2)
    table.add_column("Result", justify="center", width=12)
    table.add_column("Confidence", justify="center", width=14)
    table.add_column("Speed", justify="right", width=10, style="dim")

    for idx, vote in enumerate(result.individual_votes, 1):
        table.add_row(
            str(idx),
            vote.detector_name,
            _verdict_short(vote.verdict),
            _confidence_label(vote.score),
            f"{vote.latency_ms:.0f}ms",
        )

    # Consensus row (separator + bold)
    table.add_section()
    table.add_row(
        "",
        "[bold cyan]OVERALL[/bold cyan]",
        _verdict_short(result.consensus_verdict),
        _confidence_label(result.consensus_score),
        "",
    )

    return table


def build_ensemble_summary_panel(
    result: object,
    *,
    filename: str = "",
) -> Panel:
    """Build a summary panel for an ensemble result.

    Shows consensus, agreement level, and a warning if scanners disagree.
    """
    from synth.core.ensemble import EnsembleResult

    assert isinstance(result, EnsembleResult)

    agreement_pct = result.agreement_ratio * 100
    if agreement_pct >= 80:
        agreement_style = "green"
        agreement_word = "Strong"
    elif agreement_pct >= 60:
        agreement_style = "yellow"
        agreement_word = "Moderate"
    else:
        agreement_style = "red"
        agreement_word = "Weak"

    lines = [
        f"  [bold]Result:[/bold]      {_verdict_styled(result.consensus_verdict)}",
        f"  [bold]Confidence:[/bold]  {_confidence_label(result.consensus_score)}",
        f"  [bold]Agreement:[/bold]   [{agreement_style}]{agreement_word} ({agreement_pct:.0f}%)[/{agreement_style}]",
        f"  [bold]Scanners:[/bold]    {len(result.individual_votes)} used",
    ]

    if result.disagreement_warning:
        lines.append("")
        lines.append("  [bold yellow]⚠️  The scanners disagree on this one.[/bold yellow]")
        lines.append("  [dim]You may want to check this yourself to be sure.[/dim]")

    title = (
        f"[bold cyan]📊 Summary — {filename}[/bold cyan]"
        if filename
        else "[bold cyan]📊 Summary[/bold cyan]"
    )

    return Panel(
        "\n".join(lines),
        title=title,
        border_style="cyan",
        padding=(1, 1),
    )


def build_models_table() -> Table:
    """Build a Rich table listing all registered detectors.

    Used by the ``synth models`` command.
    """
    from synth.core.registry import DetectorRegistry

    table = Table(
        title="[bold]🧠 Available Scanners[/bold]",
        title_style="cyan",
        border_style="dim cyan",
        header_style="bold white",
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Name", style="cyan", ratio=2)
    table.add_column("Type", justify="center", width=10)
    table.add_column("Speed", justify="center", width=12)
    table.add_column("GPU?", justify="center", width=6)
    table.add_column("Size", justify="right", width=8)
    table.add_column("Description", style="dim", ratio=3)
    table.add_column("Status", justify="center", width=14)

    for meta in DetectorRegistry.all_metadata():
        gpu_str = "[green]✓[/green]" if meta.requires_gpu else "[dim]—[/dim]"
        size_str = f"{meta.model_size_mb}MB" if meta.model_size_mb else "—"
        status = "[yellow]experimental[/yellow]" if meta.experimental else "[green]stable[/green]"

        tier_colors = {"fast": "green", "balanced": "yellow", "forensic": "red"}
        tier_color = tier_colors.get(meta.speed_tier, "white")
        tier_str = f"[{tier_color}]{meta.speed_tier}[/{tier_color}]"

        table.add_row(
            meta.name,
            meta.capability.value,
            tier_str,
            gpu_str,
            size_str,
            meta.description,
            status,
        )

    return table



# ══════════════════════════════════════════════════════════════════════════════
#  Dashboard (bare `synth` command)
# ══════════════════════════════════════════════════════════════════════════════


def build_dashboard() -> Panel:
    """Build a welcoming dashboard when the user runs bare ``synth``.

    Focused on what the tool does and how to use it — no developer
    internals like PyTorch versions or CUDA/MPS status.
    """
    from synth import __version__
    from synth.core.device import detect_device

    device = detect_device()
    device_label = {
        "cuda": "🚀 GPU (NVIDIA CUDA)",
        "mps": "⚡ Apple Silicon (Metal)",
        "cpu": "💻 CPU",
    }.get(device, f"💻 {device}")

    from rich.console import Group

    info = (
        f"  [bold]Version:[/bold]  {__version__}\n"
        f"  [bold]Speed:[/bold]    {device_label}\n"
    )

    how_to = (
        "[bold]How to use:[/bold]\n"
        "\n"
        "  [cyan]synth[/cyan] [dim]photo.png[/dim]       Check if an image is AI-generated\n"
        "  [cyan]synth[/cyan] [dim]essay.pdf[/dim]       Check if text in a PDF was written by AI\n"
        "  [cyan]synth[/cyan] [dim]./folder/[/dim]       Check all files in a folder\n"
        "  [cyan]synth[/cyan] [dim]help[/dim]            Show all options\n"
    )

    group = Group(
        info,
        "",
        how_to,
    )

    return Panel(
        group,
        title="[bold cyan]Welcome to Synth[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Help menu (synth help)
# ══════════════════════════════════════════════════════════════════════════════


def build_help_menu() -> Panel:
    """Build a friendly help menu for ``synth help``.

    Focused on the essentials — what to do, not developer internals.
    """
    from rich.console import Group

    # ── What is Synth? ────────────────────────────────────────────────────
    intro = (
        "[bold]What is Synth?[/bold]\n"
        "\n"
        "  Synth checks whether text or images were created by AI.\n"
        "  Just point it at a file and it'll tell you what it finds.\n"
    )

    # ── Commands table ────────────────────────────────────────────────────
    cmd_table = Table(
        show_header=True,
        header_style="bold white",
        border_style="dim cyan",
        padding=(0, 2),
        expand=True,
    )
    cmd_table.add_column("What to type", style="cyan", ratio=2)
    cmd_table.add_column("What it does", style="white", ratio=3)
    cmd_table.add_row("synth [dim]photo.png[/dim]", "Check a single file")
    cmd_table.add_row("synth [dim]./folder/[/dim]", "Check all files in a folder")
    cmd_table.add_row("synth", "Show the welcome screen")
    cmd_table.add_row("synth help", "Show this help page")
    cmd_table.add_row("synth -V", "Show the version number")

    # ── Extra options ─────────────────────────────────────────────────────
    opt_table = Table(
        show_header=True,
        header_style="bold white",
        border_style="dim cyan",
        padding=(0, 2),
        expand=True,
    )
    opt_table.add_column("Option", style="cyan", ratio=2)
    opt_table.add_column("What it does", style="white", ratio=3)
    opt_table.add_row("--no-text", "Don't show the extracted text")
    opt_table.add_row("--lang [dim]en,fr,...[/dim]", "Set languages for text reading")
    opt_table.add_row("--verbose", "Show detailed technical output")
    opt_table.add_row("--profile [dim]fast|balanced|forensic[/dim]", "How thorough to scan")

    # ── Examples ──────────────────────────────────────────────────────────
    examples = (
        "[dim]$[/dim] [cyan]synth[/cyan] screenshot.png\n"
        "[dim]$[/dim] [cyan]synth[/cyan] homework.pdf\n"
        "[dim]$[/dim] [cyan]synth[/cyan] ./suspicious-folder/"
    )

    # ── Supported formats ─────────────────────────────────────────────────
    formats = (
        "[cyan]Images:[/cyan]  PNG  JPG  JPEG  WEBP  TIFF  BMP\n"
        "[cyan]Docs:[/cyan]    PDF"
    )

    group = Group(
        intro,
        "[bold]Commands[/bold]\n",
        cmd_table,
        "",
        "[bold]Extra Options[/bold] [dim](for advanced users)[/dim]\n",
        opt_table,
        "",
        "[bold]Examples[/bold]\n",
        examples,
        "",
        "[bold]Supported File Types[/bold]\n",
        formats,
    )

    return Panel(
        group,
        title="[bold cyan]SYNTH HELP[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
        expand=True,
    )
