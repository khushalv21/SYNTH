```text
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
  <a href="#configuration">Configuration</a>
</p>

<p align="center">
  <strong>Catch AI fakes in seconds.</strong><br>
  <em>Synth is a tool that tells you if an image or document was created by an AI or a human.</em>
</p>

---

## What is Synth?

**Synth** acts like a digital detective. Whenever you encounter a picture, a PDF, or text and wonder, *"Did a computer make this?"*, Synth gives you the answer.

Instead of relying on just one check, Synth uses a team of specialized AI models. Each model looks at the file from a different angle, and they all vote to give you a highly accurate, final verdict. Everything runs safely and privately on your own computer.

### Features

- 🔍 **Reads Anything** — Just point it at an image or a PDF. It extracts text and analyzes images automatically.
- 🤖 **Team of Experts** — Uses an ensemble of state-of-the-art models that vote together to catch fakes.
- ⚡ **Adjustable Depth** — Choose between a quick glance or a deep, forensic investigation.
- 🍎 **Works Anywhere** — Runs smartly on Mac, Windows, and Linux with automatic hardware acceleration.
- 🔒 **100% Private** — No cloud uploads needed. Your files stay perfectly private.

---

## Installation

Getting started is simple. You just need Python (3.10 or newer) and `git` installed on your computer. 

Open your terminal and run:

```bash
git clone https://github.com/khushalv21/SYNTH.git
cd SYNTH
pip install .
```

*Note: Synth will automatically detect your system's hardware to run as fast as possible. The first time you run it, it will download the necessary AI models.*

---

## Quick Start

Synth is designed to be effortless. Just type `synth` followed by the file you want to check:

```bash
synth my_picture.png
synth my_document.pdf
```

### Choose your investigation depth

You can tell Synth how closely you want it to look using "Profiles":

```bash
synth my_picture.png --profile fast       # A quick check
synth my_picture.png --profile balanced   # The normal check (recommended)
synth my_picture.png --profile forensic   # A deep, thorough investigation
```

### Scan a whole folder

Have a lot of files? You can scan a whole directory at once:

```bash
synth ./my_documents_folder/
```

---

## CLI Reference

| Command | Description |
|---|---|
| `synth <file>` | Analyze a file (Image or PDF) |
| `synth <folder>/` | Analyze an entire folder |
| `synth models` | See all the internal AI models doing the work |
| `synth help` | Show the help menu |

**Tip:** If you only want to extract the text without running an AI check, you can use the `--no-text` flag to toggle the text panel.

---

## Configuration & Advanced Use

For developers or those needing advanced setup (like plugging in external APIs such as OpenAI or Anthropic), simply create a `.env` file in the directory. You can use the provided `.env.example` as a template.

For full technical documentation, see the [Architecture Guide](docs/ARCHITECTURE.md) and [API Guide](docs/UNIVERSAL_API_GUIDE.md).

---

## License

MIT — see [LICENSE](LICENSE) for details.
