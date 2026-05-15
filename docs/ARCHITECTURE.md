# Synth — Architecture Deep-Dive

## System Overview

```
              ┌──────────────────────────────────────────────────────┐
              │                    CLI Layer                          │
              │   cli/main.py — Typer commands, profile routing      │
              │   cli/display.py — Ensemble tables, models table     │
              │   cli/silence.py — Library log suppressor            │
              └───────────────────────┬──────────────────────────────┘
                                      │
              ┌───────────────────────▼──────────────────────────────┐
              │                 Orchestration Layer                    │
              │   core/manager.py — MultiDetectorManager             │
              │   core/registry.py — DetectorRegistry                │
              │   core/ensemble.py — EnsembleAggregator              │
              │   core/router.py  — AnalysisModeResolver             │
              └──────┬──────────────────────────────┬────────────────┘
                     │                              │
       ┌─────────────▼──────────┐     ┌────────────▼───────────────┐
       │      Text Pipeline      │     │      Vision Pipeline        │
       │  core/ocr.py (extract) │     │  detectors/bnn/            │
       │  core/auth.py (legacy) │     │  detectors/cospy/          │
       │  detectors/diveye/     │     │  core/auth.py (legacy ViT) │
       │  detectors/luminol/    │     └────────────────────────────┘
       └────────────────────────┘
                     │                              │
              ┌──────▼──────────────────────────────▼────────────────┐
              │               Foundation Layer                         │
              │   core/normalizer.py — ConfidenceNormalizer           │
              │   core/weights.py   — WeightManager                   │
              │   core/device.py    — Hardware auto-detection          │
              │   core/exceptions.py                                   │
              └────────────────────────────────────────────────────────┘
```

## Operating Profiles

The `--profile` flag activates the `MultiDetectorManager` ensemble pipeline. Each profile selects a different subset of detectors from the `DetectorRegistry`.

| Profile | Text Detectors | Image Detectors | Experimental |
|---|---|---|:---:|
| `fast` | legacy-text | bnn | — |
| `balanced` | legacy-text, diveye | bnn, legacy-vision | — |
| `forensic` | legacy-text, diveye, luminol | bnn, legacy-vision, cospy | ✓ |

When no `--profile` is given, the legacy single-model path (controlled by `--engine local|api`) is used — fully backward compatible.

## Data Flow — Ensemble Mode

```
file → AnalysisModeResolver → route: "text" or "image"
                                         │
                           ┌─────────────▼──────────────┐
                           │    MultiDetectorManager      │
                           │  (profile-selected detectors)│
                           └──┬──┬──┬────────────────────┘
                              │  │  │  (parallel lazy-load)
                  ┌───────────┘  │  └──────────────┐
                  ▼              ▼                  ▼
            Detector A      Detector B         Detector C
            score=0.82      score=0.91         score=0.78
            weight=1.0      weight=1.2         weight=1.5
                  └──────────────┬──────────────────┘
                                 ▼
                       EnsembleAggregator
                       weighted mean → consensus_score
                       majority vote → consensus_verdict
                       agreement ratio, disagreement warning
                                 │
                                 ▼
                    Rich TUI: vote table + summary panel
```

## Data Flow — Legacy Mode (`--engine`)

```
image.png → OpenCV Preprocess → EasyOCR Extract → AI Detect → Rich Display
              (grayscale,         (readtext)       (Strategy)   (table,
               threshold,                                        panel)
               denoise)
```

## Hardware Auto-Detection

**File:** `src/synth/core/device.py`

```python
def detect_device() -> str:
    if torch.cuda.is_available():           return "cuda"   # NVIDIA
    elif torch.backends.mps.is_available(): return "mps"    # Apple Silicon
    else:                                   return "cpu"    # Fallback

def estimate_available_vram() -> int:  # MB
    # CUDA: torch.cuda.mem_get_info()
    # MPS:  ~50% of total system RAM (heuristic)
    # CPU:  0

def supports_mixed_precision() -> bool:
    # CUDA Volta+ (sm_70): True
    # MPS: True
    # CPU: False
```

| Component | CUDA | MPS | CPU |
|---|:---:|:---:|:---:|
| EasyOCR | ✅ gpu=True | ❌ CPU fallback | ✅ |
| HuggingFace Transformers | ✅ | ✅ Native | ✅ |
| OpenCV | N/A | N/A | ✅ |
| BNN Backbone | ✅ | ✅ | ✅ |
| CO-SPY Fusion | ✅ | ✅ | ✅ (slow) |

## DetectorRegistry

**File:** `src/synth/core/registry.py`

Central singleton that maps detector names to metadata + factory functions. Detectors auto-register when their package is imported.

```python
class DetectorCapability(str, Enum):
    TEXT_DETECTION    = "text"
    IMAGE_FORENSICS   = "image"
    STATISTICAL_TEXT  = "stat_text"

@dataclass
class DetectorMetadata:
    name: str
    capability: DetectorCapability
    speed_tier: str          # "fast" | "balanced" | "forensic"
    requires_gpu: bool
    model_size_mb: int
    weight: float            # ensemble contribution weight
    experimental: bool       # excluded from non-forensic profiles
```

Profile-based lookup:

```python
DetectorRegistry.get_by_profile("forensic", capability=DetectorCapability.IMAGE_FORENSICS)
# → ["bnn", "legacy-vision", "cospy"]
```

## MultiDetectorManager

**File:** `src/synth/core/manager.py`

Lazy-loads and caches detector instances. Accepts text or image input and returns an `EnsembleResult`.

```python
mgr = MultiDetectorManager(profile="balanced")
result = mgr.detect_text("Some article text...")
result = mgr.detect_image(Path("photo.png"))
# result.consensus_score, result.consensus_verdict, result.individual_votes
mgr.unload_all()  # Frees GPU memory
```

**Convenience factory:**

```python
mgr = DetectorFactory.create_multi("forensic")
```

## EnsembleAggregator

**File:** `src/synth/core/ensemble.py`

Weighted mean aggregation with binary majority-vote fallback:

```python
consensus_score = Σ(vote.score × vote.weight) / Σ(vote.weight)
consensus_verdict = "ai" if majority of weighted votes predict AI else "human"
agreement_ratio  = agreement_count / total_votes  (0.0 → 1.0)
disagreement_warning = True if agreement_ratio < 0.6
```

## Registered Detectors

### DivEye — IBM Surprisal-Based Text Detection

**Files:** `src/synth/detectors/diveye/`

Pipeline:
1. Tokenise with GPT-2 tokenizer
2. Compute per-token log-probability (surprisal) via GPT-2 LM
3. Extract 10-dimensional feature vector (mean, std, min, max, median, skew, kurtosis, high/low/mid surprisal ratios)
4. XGBoost binary classifier (`fast` path); threshold fallback if weights unavailable

### BNN — Faster Than Lies

**Files:** `src/synth/detectors/bnn/`

Ultra-fast deepfake detection for images:
1. Compute 3 forensic channels: Sobel gradient magnitude, FFT magnitude spectrum, LBP texture
2. Stack into 6-channel tensor alongside RGB
3. ResNet-18 backbone with custom 6→3 input adapter
4. Sigmoid output → binary verdict
- **Target latency:** ~50ms CPU

### CO-SPY — Semantic + Pixel Fusion

**Files:** `src/synth/detectors/cospy/`

Forensic-tier dual-branch architecture:
- **Semantic branch:** ResNet-50 → 2048-dim features
- **Artifact branch:** ResNet-18 → 512-dim features
- **Fusion MLP:** concat (2560-dim) → FC(512) → ReLU → FC(128) → ReLU → FC(1) → sigmoid
- Requires GPU for practical throughput

### Luminol-AI — Perplexity-Under-Shuffling *(experimental)*

**Files:** `src/synth/detectors/luminol/`

Zero-shot text detection from the Luminol-AI paper (Section 3.1–3.4):
1. Shuffle input text (word-level for single sentences, sentence-level for paragraphs)
2. Compute GPT-2 perplexity of both original and shuffled text
3. Extract 5 perplexity features: `sum`, `diff`, `ratio`, `log_ratio`, `pct_change`
4. Classify via Gamma PDF density estimation (pre-fitted on RAID benchmark parameters)

## OCR Pipeline — OpenCV Filters

**File:** `src/synth/core/ocr.py`

### Stage 1: Grayscale
`cv2.cvtColor(img, COLOR_BGR2GRAY)` — Reduces 3 channels to 1.

### Stage 2: Adaptive Thresholding
```python
cv2.adaptiveThreshold(gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, blockSize=11, C=2)
```
Calculates a local threshold per 11×11 pixel neighbourhood — handles uneven lighting and coloured backgrounds.

### Stage 3: Denoising
`cv2.fastNlMeansDenoising(thresh, h=10)` — Non-local means. Suppresses scanner noise and JPEG artifacts without destroying text edges.

## Authentication — Legacy Strategy Pattern

**File:** `src/synth/core/auth.py`

```
BaseAuthenticator (ABC)
├── detect(text) → AuthResult
└── name → str
      │
 ┌────┴────┐
 │          │
LocalHF  UniversalAPI         _LegacyTextAdapter   (bridges → registry)
            ├── OpenAI        _LegacyVisionAdapter (bridges → registry)
            ├── Anthropic
            ├── Ollama
            └── Custom
```

Legacy authenticators are wrapped into `_LegacyTextAdapter` / `_LegacyVisionAdapter` and registered as `"legacy-text"` / `"legacy-vision"` in the `DetectorRegistry`, making them available to the ensemble pipeline alongside the new detectors.

## ConfidenceNormalizer

**File:** `src/synth/core/normalizer.py`

Unified 0→1 normalisation regardless of source format:

| Input Format | Method |
|---|---|
| Raw probability (0→1) | Pass-through with clamp |
| Logit (unbounded) | Sigmoid |
| Binary label (0 or 1) | Cast to float |
| String label ("ai"/"human") | 1.0 / 0.0 |

## WeightManager

**File:** `src/synth/core/weights.py`

Downloads model weights at runtime with:
- SHA-256 checksum verification
- Rich progress bar
- Local cache at `~/.cache/synth/weights/`
- Skip re-download if cached + hash matches

## Extension Points

1. **New detector** — Subclass `BaseTextDetector` or `BaseVisionDetector`, call `DetectorRegistry.register()` in your package `__init__.py`
2. **New profile** — Extend profile→tier mapping in `registry.py` `get_by_profile()`
3. **New languages** — `synth scan.jpg --lang en,ja,ko` (80+ EasyOCR languages)
4. **Tune preprocessing** — Pass a custom `PreprocessConfig` to `DocumentScanner`
5. **Legacy strategy plugin** — `DetectorFactory.register("custom", MyClass)` for the legacy path
