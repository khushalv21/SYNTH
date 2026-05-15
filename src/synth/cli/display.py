"""Rich terminal UI components for Synth.

Reusable display helpers: banner, result tables, text panels,
and progress bars. Keeps the main CLI module focused on command logic.
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
[dim]  AI Content Authenticator & OCR Engine[/dim]
"""


def print_banner() -> None:
    """Print the stylised SYNTH ASCII banner."""
    console.print(BANNER)


# ── Verdict colours ───────────────────────────────────────────────────────────

_VERDICT_STYLES: dict[str, str] = {
    "human": "bold green",
    "ai": "bold red",
    "mixed": "bold yellow",
    "real": "bold green",
    "fake": "bold red",
}


def _verdict_styled(verdict: str) -> str:
    """Return a Rich-markup-wrapped verdict string."""
    style = _VERDICT_STYLES.get(verdict, "bold white")
    tag = verdict.upper()
    return f"[{style}]{tag}[/{style}]"


def _score_styled(score: float) -> str:
    """Return colour-coded score: green ≤0.3, yellow 0.3–0.7, red ≥0.7."""
    pct = f"{score * 100:.1f}%"
    if score <= 0.3:
        return f"[green]{pct}[/green]"
    if score <= 0.7:
        return f"[yellow]{pct}[/yellow]"
    return f"[red]{pct}[/red]"


# ── Results table (text mode) ────────────────────────────────────────────────


def build_results_table(
    results: list[tuple[str, AuthResult]],
) -> Table:
    """Build a Rich table summarising verification results.

    Args:
        results: List of ``(filename, AuthResult)`` tuples.

    Returns:
        A styled :class:`rich.table.Table`.
    """
    table = Table(
        title="[bold]Verification Results[/bold]",
        title_style="cyan",
        border_style="dim cyan",
        header_style="bold white",
        row_styles=["", "dim"],
        expand=True,
        padding=(0, 1),
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("File", style="white", ratio=3)
    table.add_column("Verdict", justify="center", width=10)
    table.add_column("AI Score", justify="center", width=10)
    table.add_column("Model", style="dim", ratio=2)

    for idx, (filename, result) in enumerate(results, 1):
        table.add_row(
            str(idx),
            filename,
            _verdict_styled(result.verdict),
            _score_styled(result.score),
            result.model,
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

    # Build subtitle
    verdict_display = _verdict_styled(result.verdict)
    score_display = _score_styled(result.score)
    subtitle = f"Verdict: {verdict_display}  │  AI Score: {score_display}"

    title = f"[bold cyan]📄 {filename}[/bold cyan]" if filename else "[bold cyan]Text Analysis[/bold cyan]"

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
    """Build a summary panel with aggregate statistics."""
    total = len(results)
    ai_count = sum(1 for _, r in results if r.verdict == "ai")
    human_count = sum(1 for _, r in results if r.verdict == "human")
    mixed_count = sum(1 for _, r in results if r.verdict == "mixed")
    avg_score = sum(r.score for _, r in results) / total if total else 0.0

    lines = [
        f"[bold]Files scanned:[/bold]  {total}",
        f"[bold green]Human:[/bold green]          {human_count}",
        f"[bold red]AI-generated:[/bold red]   {ai_count}",
        f"[bold yellow]Mixed:[/bold yellow]          {mixed_count}",
        f"[bold]Avg AI score:[/bold]   {_score_styled(avg_score)}",
    ]

    return Panel(
        "\n".join(lines),
        title="[bold cyan]📊 Batch Summary[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Vision mode display (AI image forensics)
# ══════════════════════════════════════════════════════════════════════════════


def build_vision_results_table(
    results: list[tuple[str, VisionAuthResult]],
) -> Table:
    """Build a Rich table for AI image detection results.

    Displays 'AI Image Probability' and 'Real / Fake' verdict instead
    of text-centric metrics like burstiness.
    """
    table = Table(
        title="[bold]🔍 Image Forensics Results[/bold]",
        title_style="cyan",
        border_style="dim cyan",
        header_style="bold white",
        row_styles=["", "dim"],
        expand=True,
        padding=(0, 1),
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("File", style="white", ratio=3)
    table.add_column("Verdict", justify="center", width=10)
    table.add_column("AI Image Prob.", justify="center", width=16)
    table.add_column("Model", style="dim", ratio=2)

    for idx, (filename, result) in enumerate(results, 1):
        table.add_row(
            str(idx),
            filename,
            _verdict_styled(result.verdict),
            _score_styled(result.ai_probability),
            result.model,
        )

    return table


def build_vision_summary_panel(
    results: list[tuple[str, VisionAuthResult]],
) -> Panel:
    """Build a summary panel for vision-mode batch results."""
    total = len(results)
    real_count = sum(1 for _, r in results if r.verdict == "real")
    fake_count = sum(1 for _, r in results if r.verdict == "fake")
    avg_prob = (
        sum(r.ai_probability for _, r in results) / total if total else 0.0
    )

    lines = [
        f"[bold]Images scanned:[/bold]     {total}",
        f"[bold green]Real (authentic):[/bold green]  {real_count}",
        f"[bold red]Fake (AI-generated):[/bold red] {fake_count}",
        f"[bold]Avg AI probability:[/bold] {_score_styled(avg_prob)}",
    ]

    return Panel(
        "\n".join(lines),
        title="[bold cyan]📊 Image Forensics Summary[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
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

    title = f"[bold]🔬 Ensemble Analysis — {filename}[/bold]" if filename else "[bold]🔬 Ensemble Analysis[/bold]"
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
    table.add_column("Detector", style="white", ratio=2)
    table.add_column("Verdict", justify="center", width=10)
    table.add_column("AI Score", justify="center", width=10)
    table.add_column("Weight", justify="center", width=8)
    table.add_column("Latency", justify="right", width=10, style="dim")

    for idx, vote in enumerate(result.individual_votes, 1):
        table.add_row(
            str(idx),
            vote.detector_name,
            _verdict_styled(vote.verdict),
            _score_styled(vote.score),
            f"{vote.weight:.1f}",
            f"{vote.latency_ms:.0f}ms",
        )

    # Consensus row (separator + bold)
    table.add_section()
    table.add_row(
        "",
        "[bold cyan]CONSENSUS[/bold cyan]",
        _verdict_styled(result.consensus_verdict),
        _score_styled(result.consensus_score),
        "",
        "",
    )

    return table


def build_ensemble_summary_panel(
    result: object,
    *,
    filename: str = "",
) -> Panel:
    """Build a summary panel for an ensemble result.

    Shows consensus score, agreement ratio, and disagreement warnings.
    """
    from synth.core.ensemble import EnsembleResult

    assert isinstance(result, EnsembleResult)

    agreement_pct = result.agreement_ratio * 100
    if agreement_pct >= 80:
        agreement_style = "green"
    elif agreement_pct >= 60:
        agreement_style = "yellow"
    else:
        agreement_style = "red"

    lines = [
        f"[bold]Consensus:[/bold]         {_verdict_styled(result.consensus_verdict)}",
        f"[bold]AI Score:[/bold]          {_score_styled(result.consensus_score)}",
        f"[bold]Agreement:[/bold]         [{agreement_style}]{agreement_pct:.0f}%[/{agreement_style}]",
        f"[bold]Detectors used:[/bold]    {len(result.individual_votes)}",
        f"[bold]Domain:[/bold]            {result.domain}",
    ]

    if result.disagreement_warning:
        lines.append("")
        lines.append("[bold red]⚠ Significant detector disagreement detected[/bold red]")
        lines.append("[dim]Results may be less reliable — consider manual review[/dim]")

    title = (
        f"[bold cyan]📊 Ensemble Summary — {filename}[/bold cyan]"
        if filename
        else "[bold cyan]📊 Ensemble Summary[/bold cyan]"
    )

    return Panel(
        "\n".join(lines),
        title=title,
        border_style="cyan",
        padding=(1, 2),
    )


def build_models_table() -> Table:
    """Build a Rich table listing all registered detectors.

    Used by the ``synth models`` command.
    """
    from synth.core.registry import DetectorRegistry

    table = Table(
        title="[bold]🧠 Registered Detectors[/bold]",
        title_style="cyan",
        border_style="dim cyan",
        header_style="bold white",
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Name", style="cyan", ratio=2)
    table.add_column("Domain", justify="center", width=10)
    table.add_column("Speed Tier", justify="center", width=12)
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
    """Build the system dashboard shown when the user runs bare ``synth``.

    Displays the ASCII banner, version, device info, strategies, and
    PyTorch/CUDA/MPS availability in a single styled panel.
    """
    import sys

    from synth import __version__
    from synth.core.auth import DetectorFactory
    from synth.core.device import detect_device

    device = detect_device()

    info_lines = [
        f"[bold]Version:[/bold]       {__version__}",
        f"[bold]Python:[/bold]        {sys.version.split()[0]}",
        f"[bold]Compute:[/bold]       [cyan]{device}[/cyan]",
        f"[bold]Strategies:[/bold]    {', '.join(DetectorFactory.available())}",
    ]

    try:
        import torch

        info_lines.append(f"[bold]PyTorch:[/bold]       {torch.__version__}")
        info_lines.append(
            f"[bold]CUDA:[/bold]          {'[green]✓ ' + torch.cuda.get_device_name(0) + '[/green]' if torch.cuda.is_available() else '[red]✗ not available[/red]'}"
        )
        info_lines.append(
            f"[bold]MPS:[/bold]           {'[green]✓ available[/green]' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else '[red]✗ not available[/red]'}"
        )
    except ImportError:
        pass

    return Panel(
        "\n".join(info_lines),
        title="[bold cyan]System Info[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Help menu (synth help)
# ══════════════════════════════════════════════════════════════════════════════


def build_help_menu() -> Panel:
    """Build a custom Rich help menu for ``synth help``.

    Returns a styled panel with commands, options, examples, and
    supported file formats.
    """
    # ── Commands table ────────────────────────────────────────────────────
    cmd_table = Table(
        show_header=True,
        header_style="bold white",
        border_style="dim cyan",
        padding=(0, 2),
        expand=True,
    )
    cmd_table.add_column("Command", style="cyan", ratio=2)
    cmd_table.add_column("Description", style="white", ratio=3)
    cmd_table.add_row("synth", "System dashboard & info")
    cmd_table.add_row("synth [dim]<file>[/dim]", "Auto-analyse a file")
    cmd_table.add_row("synth [dim]<folder>/[/dim]", "Batch-analyse a directory")
    cmd_table.add_row("synth help", "Show this menu")
    cmd_table.add_row("synth -V", "Show version")

    # ── Options table ─────────────────────────────────────────────────────
    opt_table = Table(
        show_header=True,
        header_style="bold white",
        border_style="dim cyan",
        padding=(0, 2),
        expand=True,
    )
    opt_table.add_column("Option", style="cyan", ratio=2)
    opt_table.add_column("Description", style="white", ratio=3)
    opt_table.add_row("--engine [dim]local|api[/dim]", "Detection backend")
    opt_table.add_row("--agent [dim]<model>[/dim]", "Model name override")
    opt_table.add_row("--lang [dim]en,fr,...[/dim]", "OCR language codes")
    opt_table.add_row("--no-text", "Hide extracted text panel")
    opt_table.add_row("--verbose", "Enable debug output")

    # ── Examples ──────────────────────────────────────────────────────────
    examples = (
        "[dim]$[/dim] [cyan]synth[/cyan] photo.png\n"
        "[dim]$[/dim] [cyan]synth[/cyan] report.pdf\n"
        "[dim]$[/dim] [cyan]synth[/cyan] ./documents/\n"
        "[dim]$[/dim] [cyan]synth[/cyan] image.jpg --engine api\n"
        "[dim]$[/dim] [cyan]synth[/cyan] scan.jpg --lang en,fr"
    )

    # ── Supported formats ─────────────────────────────────────────────────
    formats = (
        "[cyan]Images:[/cyan]  png  jpg  jpeg  webp  tiff  tif  bmp\n"
        "[cyan]Docs:[/cyan]    pdf"
    )

    # ── Assemble ──────────────────────────────────────────────────────────
    from rich.console import Group

    group = Group(
        "[bold]Commands[/bold]\n",
        cmd_table,
        "",
        "[bold]Options[/bold]\n",
        opt_table,
        "",
        "[bold]Examples[/bold]\n",
        examples,
        "",
        "[bold]Supported Formats[/bold]\n",
        formats,
    )

    return Panel(
        group,
        title="[bold cyan]SYNTH COMMAND MENU[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
        expand=True,
    )

