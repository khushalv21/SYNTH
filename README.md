```
  ███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗
  ██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║
  ███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║
  ╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║
  ███████║   ██║   ██║ ╚████║   ██║   ██║  ██║
  ╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝
```

<p align="center">
  <a href="https://github.com/khushalv21/SYNTH"><img src="https://img.shields.io/github/stars/khushalv21/SYNTH?style=social" alt="Stars"></a>
  <a href="https://github.com/khushalv21/SYNTH/blob/main/LICENSE"><img src="https://img.shields.io/github/license/khushalv21/SYNTH.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#cli-reference">CLI Reference</a> •
  <a href="#detection-profiles">Profiles</a> •
  <a href="#configuration">Configuration</a> •
  <a href="docs/ARCHITECTURE.md">Architecture</a> •
  <a href="docs/UNIVERSAL_API_GUIDE.md">API Guide</a>
</p>

<p align="center">
  <strong>AI Content Authenticator &amp; Forensic Engine</strong><br>
  <em>Detect AI-generated text and images with an ensemble of state-of-the-art models — in a single command.</em>
</p>

---

## What is Synth?

**Synth** is a production-grade CLI forensic engine that authenticates content origin — whether text extracted from images/PDFs, or the images themselves. It runs an intelligent ensemble of multiple AI detection models, routes them by operating profile, and aggregates a consensus verdict. Everything runs in your terminal — no web server, no GUI, no cloud.

### Features

- 🔍 **OCR Pipeline** — OpenCV preprocessing + EasyOCR extraction with multi-language support
- 🤖 **AI Text Detection** — Ensemble of RoBERTa (legacy) + DivEye (IBM surprisal + XGBoost) + Luminol-AI (perplexity-under-shuffling)
- 🖼️ **AI Image Forensics** — Ensemble of BNN (ultra-fast, ~50ms), ViT (legacy), and CO-SPY (semantic + pixel fusion)
- ⚡ **Operating Profiles** — `fast`, `balanced`, `forensic` — trade speed for accuracy
- 🧩 **Plugin Architecture** — `DetectorRegistry` + lazy-loaded `MultiDetectorManager` for drop-in custom detectors
- 📄 **PDF Ingestion** — Multi-page PDFs with zero system dependencies
- 🍎 **Hardware Agnostic** — Auto-detects CUDA, MPS (Apple Silicon), or CPU; VRAM-aware loading
- 📁 **Batch Processing** — Scan entire directories with per-file ensemble reports
- 🎨 **Rich TUI** — Per-detector vote tables, agreement ratios, colour-coded consensus verdicts
- 📦 **Simple Setup** — Clone, install, detect — three commands

---

## Installation

> **Requirements:** Python 3.10+ and `git`. Synth auto-detects your hardware — CUDA, Apple MPS, or CPU — no manual config needed.

### Quick install (all platforms)

```bash
git clone https://github.com/khushalv21/SYNTH.git
cd SYNTH
pip install .
```

### Optional extras

```bash
pip install ".[fast]"      # + XGBoost, SciPy (DivEye)
pip install ".[vision]"    # + torchvision (CO-SPY)
pip install ".[full]"      # Everything
```

### Verify installation

```bash
synth --version
synth models     # List all registered detectors
```

> **Note**: On first run, HuggingFace models download and cache automatically (~500 MB text, ~350 MB image, ~400 MB CO-SPY).

---

### 🍎 macOS

**Prerequisites:** Xcode Command Line Tools (provides `git` and system compilers).

```bash
xcode-select --install          # If not already installed
```

**Install Synth:**

```bash
git clone https://github.com/khushalv21/SYNTH.git
cd SYNTH
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

**GPU acceleration:** Apple Silicon (M1/M2/M3/M4) is detected automatically via MPS — no additional setup required. Intel Macs run on CPU.

**Development install:**

```bash
git clone https://github.com/khushalv21/SYNTH.git
cd SYNTH
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

### 🐧 Linux

<details>
<summary><strong>Ubuntu / Debian</strong></summary>

**Install system dependencies:**

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
```

> The `libgl1` and `libglib2.0-0` packages are required by OpenCV (used in the OCR pipeline).

**Install Synth:**

```bash
git clone https://github.com/khushalv21/SYNTH.git
cd SYNTH
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

**Development install:**

```bash
git clone https://github.com/khushalv21/SYNTH.git
cd SYNTH
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

</details>

<details>
<summary><strong>Fedora / RHEL / CentOS</strong></summary>

**Install system dependencies:**

```bash
sudo dnf install -y python3 python3-pip git \
    mesa-libGL glib2 libSM libXext libXrender
```

**Install Synth:**

```bash
git clone https://github.com/khushalv21/SYNTH.git
cd SYNTH
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

</details>

<details>
<summary><strong>Arch Linux</strong></summary>

**Install system dependencies:**

```bash
sudo pacman -S python python-pip git mesa glib2
```

**Install Synth:**

```bash
git clone https://github.com/khushalv21/SYNTH.git
cd SYNTH
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

</details>

**GPU acceleration (NVIDIA CUDA):**

If you have an NVIDIA GPU and want accelerated inference:

```bash
# Install PyTorch with CUDA support (replace cu124 with your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Then install Synth
git clone https://github.com/khushalv21/SYNTH.git && cd SYNTH && pip install .
```

Verify GPU detection:

```bash
synth          # Dashboard shows "Device: cuda" if detected
```

> **Tip:** Run `nvidia-smi` to check your CUDA version and ensure drivers are installed.

---

### 🪟 Windows

**Prerequisites:**

1. **Python 3.10+** — Download from [python.org](https://www.python.org/downloads/). During installation, check **"Add python.exe to PATH"**.
2. **Git** — Download from [git-scm.com](https://git-scm.com/download/win) or install via `winget install Git.Git`.

**Install Synth (Command Prompt or PowerShell):**

```powershell
git clone https://github.com/khushalv21/SYNTH.git
cd SYNTH
python -m venv .venv
.venv\Scripts\activate
pip install .
```

> **Note:** On Windows, use `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

**Install with extras:**

```powershell
pip install ".[full]"
```

**GPU acceleration (NVIDIA CUDA):**

```powershell
# Install PyTorch with CUDA support first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Then install Synth
git clone https://github.com/khushalv21/SYNTH.git; cd SYNTH; pip install .
```

**Development install:**

```powershell
git clone https://github.com/khushalv21/SYNTH.git
cd SYNTH
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

> **Troubleshooting (Windows):**
> - If `synth` is not recognized after install, ensure your venv is activated or add Python's `Scripts` directory to your `PATH`.
> - For SSL errors behind a corporate proxy, configure your proxy settings or use `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org .` when installing dependencies.
> - If you encounter C++ build errors installing dependencies, install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

---

## Quick Start

### Analyse any file (auto-detection)

Synth automatically detects whether to run text analysis or image forensics — no flags needed.

```bash
synth photo.png
synth report.pdf
```

### Ensemble mode (new)

Use `--profile` to activate the multi-detector ensemble pipeline:

```bash
synth photo.png --profile fast       # BNN only — ultra-fast (~50ms)
synth photo.png --profile balanced   # BNN + ViT + DivEye ensemble
synth photo.png --profile forensic   # BNN + ViT + CO-SPY + DivEye + Luminol ensemble
```

The ensemble output shows a per-detector vote table, individual confidence scores, and a weighted consensus verdict with an agreement ratio.

### List all registered detectors

```bash
synth models
```

### Batch scan a directory

```bash
synth ./documents/ --profile balanced
```

### Use a remote API (legacy)

```bash
synth document.jpg --engine api --agent gpt-4o
```

### Specify OCR languages

```bash
synth scan.jpg --lang en,fr,de
```

---

## CLI Reference

### Commands

| Command | Description |
|---|---|
| `synth` | System dashboard — version, device, registered models |
| `synth models` | List all registered detectors with metadata |
| `synth <file>` | Auto-detect and analyse a file |
| `synth <folder>/` | Batch-analyse a directory |
| `synth help` | Show the command reference menu |
| `synth -V` | Print version and exit |

### Options

| Option | Default | Description |
|---|---|---|
| `--profile`, `-p` | *(none)* | Ensemble profile: `fast`, `balanced`, or `forensic` |
| `--engine`, `-e` | `local` | Legacy strategy: `local` (HuggingFace) or `api` (remote HTTP) |
| `--agent`, `-a` | *(auto)* | Model name (local) or model ID (API) |
| `--show-text` / `--no-text` | `--show-text` | Toggle extracted text panel with AI-pattern highlighting |
| `--lang`, `-l` | `en` | Comma-separated OCR language codes |
| `--verbose` | `false` | Enable debug logging |

> `--profile` and `--engine` are independent. `--profile` activates the ensemble pipeline; `--engine` uses the legacy single-model pipeline. When `--profile` is set, it takes precedence.

**Exit codes:**
- `0` — All content verified as human-created
- `1` — AI-generated content detected (useful in CI/CD pipelines)

**Supported formats:**
- **Images:** `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.webp`
- **Documents:** `.pdf` (multi-page)

---

## Detection Profiles

| Profile | Text Detectors | Image Detectors | Target Use Case |
|---|---|---|---|
| `fast` | RoBERTa (legacy) | BNN (~50ms) | CI gates, high-volume triage |
| `balanced` | RoBERTa + DivEye | BNN + ViT | General-purpose verification |
| `forensic` | RoBERTa + DivEye + Luminol* | BNN + ViT + CO-SPY | Deep investigation, legal/academic |

*Luminol is experimental — enabled only in `forensic` profile.

### Registered Detectors

| Name | Domain | Speed | GPU | Size | Status |
|---|---|---|---|---|---|
| `legacy-text` | text | fast | — | 500 MB | stable |
| `legacy-vision` | image | balanced | — | 350 MB | stable |
| `diveye` | text | balanced | — | 550 MB | stable |
| `bnn` | image | fast | — | 25 MB | stable |
| `cospy` | image | forensic | ✓ | 400 MB | stable |
| `luminol` | text (statistical) | forensic | — | 550 MB | experimental |

---

## Configuration

### Local engine (default)

No configuration required. Synth downloads and caches models on first use.

```bash
synth image.png --engine local
```

### API engine

Create a `.env` file (see `.env.example`):

```env
SYNTH_API_BASE_URL=https://api.openai.com/v1/chat/completions
SYNTH_API_KEY=sk-your-key-here
SYNTH_API_MODEL=gpt-4o
```

For detailed API configuration (Ollama, Anthropic, custom endpoints), see the **[Universal API Guide](docs/UNIVERSAL_API_GUIDE.md)**.

---

## Project Structure

```
synth/
├── pyproject.toml                   # Package metadata & optional extras
├── .env.example                     # API config template
├── config/
│   ├── payload_openai.json          # OpenAI payload mapping
│   └── payload_anthropic.json       # Anthropic payload mapping
├── docs/
│   ├── ARCHITECTURE.md              # Technical deep-dive
│   └── UNIVERSAL_API_GUIDE.md       # API configuration guide
└── src/synth/
    ├── __init__.py                  # Version string
    ├── cli/
    │   ├── main.py                  # Typer commands, profile routing
    │   ├── display.py               # Rich TUI — ensemble tables, models table
    │   └── silence.py               # Library log suppressor
    ├── core/
    │   ├── auth.py                  # Authenticators + legacy adapters
    │   ├── device.py                # Hardware auto-detection + VRAM estimation
    │   ├── ensemble.py              # EnsembleAggregator + DetectorVote
    │   ├── exceptions.py            # Custom exceptions
    │   ├── manager.py               # MultiDetectorManager (lazy load + cache)
    │   ├── normalizer.py            # ConfidenceNormalizer (0→1 unified scale)
    │   ├── ocr.py                   # OpenCV + EasyOCR + PDF pipeline
    │   ├── registry.py              # DetectorRegistry + DetectorCapability
    │   ├── router.py                # AnalysisModeResolver (text vs. image routing)
    │   └── weights.py               # WeightManager (download, cache, checksum)
    ├── data/
    │   └── luminol_distributions.json  # Pre-fitted Gamma distribution params
    └── detectors/
        ├── base.py                  # BaseTextDetector, BaseVisionDetector
        ├── bnn/                     # BNN — ultra-fast image forensics
        ├── cospy/                   # CO-SPY — semantic + pixel fusion
        ├── diveye/                  # DivEye — surprisal-based text detection
        └── luminol/                 # Luminol-AI — perplexity-under-shuffling
```

---

## Development

```bash
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
