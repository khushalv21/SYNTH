"""AI content authentication — Strategy Pattern implementation.

This module provides a pluggable detection system where **any** model
(local HuggingFace checkpoint, OpenAI, Anthropic, or custom HTTP API)
can be slotted in behind a common :class:`BaseAuthenticator` interface.

For AI *image* forensics (detecting Midjourney, Stable Diffusion, DALL·E),
use :class:`VisionAuthenticator` which runs a ViT classification model.

Use :class:`DetectorFactory` to instantiate text-detection strategies by name.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from synth.core.device import detect_device, get_torch_device
from synth.core.exceptions import SynthError

logger = logging.getLogger(__name__)

# Load .env once at import time
load_dotenv()


# ── Verdict constants ─────────────────────────────────────────────────────────

VERDICT_HUMAN = "human"
VERDICT_AI = "ai"
VERDICT_MIXED = "mixed"


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AuthResult:
    """Structured result from an AI detection run.

    Attributes:
        score: Probability that the text is AI-generated
            (``0.0`` = definitely human, ``1.0`` = definitely AI).
        verdict: One of ``"human"``, ``"ai"``, or ``"mixed"``.
        reasoning: Human-readable explanation of the verdict.
        model: Identifier of the model that produced this result.
    """

    score: float
    verdict: str
    reasoning: str
    model: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "score": self.score,
            "verdict": self.verdict,
            "reasoning": self.reasoning,
            "model": self.model,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy interface
# ══════════════════════════════════════════════════════════════════════════════


class BaseAuthenticator(ABC):
    """Abstract strategy interface for AI content detection.

    All authenticators **must** implement :meth:`detect`, returning a
    standardised :class:`AuthResult` regardless of the underlying model.
    """

    @abstractmethod
    def detect(self, text: str) -> AuthResult:
        """Analyse *text* and return an AI-authorship verdict.

        Args:
            text: The content to analyse.

        Returns:
            :class:`AuthResult` with score, verdict, reasoning, and model.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this strategy."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 1 — Local HuggingFace model
# ══════════════════════════════════════════════════════════════════════════════


class LocalHFAuthenticator(BaseAuthenticator):
    """Detect AI-generated text using a local HuggingFace transformer.

    Default model: ``roberta-base-openai-detector`` (OpenAI's GPT-2 output
    detector fine-tuned on RoBERTa).

    **Device selection** is fully automatic::

        cuda  →  NVIDIA GPU (fastest)
        mps   →  Apple Silicon Metal  (M-series Macs)
        cpu   →  Fallback

    Example::

        auth = LocalHFAuthenticator()
        result = auth.detect("Some text to check...")
        print(result.verdict)  # "ai" | "human" | "mixed"
    """

    DEFAULT_MODEL = "roberta-base-openai-detector"

    # ── Verdict thresholds ────────────────────────────────────────────────
    AI_THRESHOLD: float = 0.75
    HUMAN_THRESHOLD: float = 0.25

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._device_str = detect_device()
        self._device = get_torch_device()
        self._pipeline = self._load_pipeline()
        logger.info(
            "LocalHFAuthenticator ready · model=%s · device=%s",
            self._model_name,
            self._device,
        )

    # ── Device & model bootstrap ──────────────────────────────────────────

    def _load_pipeline(self) -> Any:
        """Load the HuggingFace ``text-classification`` pipeline.

        The pipeline is placed on the best available device automatically.
        ``top_k=None`` ensures all label scores are returned.
        """
        from transformers import pipeline as hf_pipeline  # heavy — deferred

        return hf_pipeline(
            "text-classification",
            model=self._model_name,
            device=self._device,
            top_k=None,  # return scores for every label
        )

    # ── Core detection ────────────────────────────────────────────────────

    def detect(self, text: str) -> AuthResult:
        """Run the local model on *text* and return a verdict.

        The input is truncated to 512 tokens to respect the model's
        context window.

        Raises:
            SynthError: If *text* is empty or whitespace-only.
        """
        if not text.strip():
            raise SynthError("Cannot authenticate empty text")

        # Most classification models cap at 512 tokens
        results = self._pipeline(text[:512])

        # results is List[List[Dict]] when top_k=None
        label_scores: dict[str, float] = {
            r["label"]: r["score"] for r in results[0]
        }

        ai_score = self._extract_ai_score(label_scores)
        verdict = self._score_to_verdict(ai_score)

        return AuthResult(
            score=round(ai_score, 4),
            verdict=verdict,
            reasoning=self._build_reasoning(ai_score, verdict, label_scores),
            model=self._model_name,
        )

    # ── Label parsing ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_ai_score(label_scores: dict[str, float]) -> float:
        """Extract the AI probability from heterogeneous label formats.

        Supports common naming conventions:
            - ``LABEL_1`` / ``LABEL_0`` (roberta-base-openai-detector)
            - ``Fake`` / ``Real``
            - ``ai`` / ``human``
        """
        # Try known AI-positive labels first
        for ai_label in ("LABEL_1", "Fake", "fake", "ai", "AI", "machine"):
            if ai_label in label_scores:
                return label_scores[ai_label]

        # Binary fallback: assume second label is AI
        labels = list(label_scores.keys())
        if len(labels) == 2:
            return label_scores[labels[1]]

        return max(label_scores.values())

    @staticmethod
    def _score_to_verdict(score: float) -> str:
        """Map a continuous score to a discrete verdict."""
        if score >= LocalHFAuthenticator.AI_THRESHOLD:
            return VERDICT_AI
        if score <= LocalHFAuthenticator.HUMAN_THRESHOLD:
            return VERDICT_HUMAN
        return VERDICT_MIXED

    @staticmethod
    def _build_reasoning(
        score: float,
        verdict: str,
        raw: dict[str, float],
    ) -> str:
        pct = score * 100
        raw_str = ", ".join(f"{k}={v:.3f}" for k, v in raw.items())
        return (
            f"AI probability: {pct:.1f}%. "
            f"Verdict: {verdict}. "
            f"Raw model scores: [{raw_str}]"
        )

    @property
    def name(self) -> str:
        return f"local:{self._model_name}"


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 2 — Universal remote API
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class APIEndpointConfig:
    """Configuration for a remote detection API.

    Can be hydrated from:

    * **Environment variables / ``.env``** — via :meth:`from_env`
    * **JSON config file** — via :meth:`from_json`

    ``payload_template`` uses ``{text}`` as a placeholder that gets
    substituted with the actual input text at request time.
    """

    base_url: str
    api_key: str
    model: str = ""

    # Auth
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer"

    # Request shape — {text} is replaced with input
    payload_template: dict[str, Any] = field(default_factory=dict)

    # Response parsing — dot-notation paths into the JSON response
    #   e.g. "choices.0.message.content" for OpenAI
    score_path: str = "score"
    label_path: str = "label"
    reasoning_path: str = "reasoning"

    # Network
    timeout_seconds: float = 30.0

    # ── Constructors ──────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> APIEndpointConfig:
        """Build config from environment / ``.env`` file.

        Required variables:
            ``SYNTH_API_BASE_URL`` — Full endpoint URL.
            ``SYNTH_API_KEY``      — Authentication key.

        Optional variables:
            ``SYNTH_API_MODEL``    — Model identifier.
            ``SYNTH_PAYLOAD_MAP``  — Path to a JSON payload mapping file.
        """
        load_dotenv()

        base_url = os.getenv("SYNTH_API_BASE_URL", "")
        api_key = os.getenv("SYNTH_API_KEY", "")
        model = os.getenv("SYNTH_API_MODEL", "")

        if not base_url:
            raise SynthError(
                "SYNTH_API_BASE_URL is not set. "
                "Add it to your .env file or export it as an env var."
            )
        if not api_key:
            raise SynthError(
                "SYNTH_API_KEY is not set. "
                "Add it to your .env file or export it as an env var."
            )

        # Optional payload mapping file
        payload_map_path = os.getenv("SYNTH_PAYLOAD_MAP", "")
        payload_template: dict[str, Any] = {}
        overrides: dict[str, Any] = {}

        if payload_map_path:
            mapping = cls._load_json(Path(payload_map_path))
            payload_template = mapping.pop("payload_template", {})
            # Pull any override keys that match our fields
            for key in (
                "score_path",
                "label_path",
                "reasoning_path",
                "auth_header",
                "auth_prefix",
                "timeout_seconds",
            ):
                if key in mapping:
                    overrides[key] = mapping[key]

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            payload_template=payload_template,
            **overrides,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> APIEndpointConfig:
        """Load a complete config from a JSON file.

        The JSON must contain at least ``base_url`` and ``api_key``.
        All other fields are optional and will use defaults.
        """
        data = cls._load_json(Path(path))
        return cls(**data)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise SynthError(f"Config file not found: {path}")
        return json.loads(path.read_text())  # type: ignore[no-any-return]


class UniversalAPIAuthenticator(BaseAuthenticator):
    """Generic API-based AI content detector.

    Connects to **any** HTTP endpoint — OpenAI, Anthropic, or a custom
    detection service — through a configurable payload template and
    dot-notation response path mapping.

    Lifecycle::

        # From .env
        auth = UniversalAPIAuthenticator()

        # From explicit config
        cfg = APIEndpointConfig(base_url="https://...", api_key="sk-...")
        auth = UniversalAPIAuthenticator(config=cfg)

        # From JSON file
        cfg = APIEndpointConfig.from_json("config/openai.json")
        auth = UniversalAPIAuthenticator(config=cfg)

        result = auth.detect("Text to check...")

    Supports context-manager protocol for clean shutdown::

        with UniversalAPIAuthenticator() as auth:
            result = auth.detect(text)
    """

    def __init__(self, config: APIEndpointConfig | None = None) -> None:
        self._config = config or APIEndpointConfig.from_env()
        self._client = httpx.Client(
            base_url=self._config.base_url,
            headers=self._build_headers(),
            timeout=self._config.timeout_seconds,
        )
        logger.info(
            "UniversalAPIAuthenticator ready · endpoint=%s · model=%s",
            self._config.base_url,
            self._config.model or "(default)",
        )

    # ── Request construction ──────────────────────────────────────────────

    def _build_headers(self) -> dict[str, str]:
        cfg = self._config
        return {
            cfg.auth_header: f"{cfg.auth_prefix} {cfg.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, text: str) -> dict[str, Any]:
        """Render the payload, substituting ``{text}`` placeholders.

        If no ``payload_template`` is configured, falls back to an
        OpenAI-compatible chat-completion request that asks the model
        to act as an AI content detector.
        """
        template = self._config.payload_template

        if not template:
            # Sensible default — OpenAI / Anthropic chat format
            return {
                "model": self._config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an AI content detector. Analyse the "
                            "following text and respond with valid JSON "
                            "containing exactly three keys: "
                            '"score" (float 0.0=human to 1.0=AI), '
                            '"verdict" ("human", "ai", or "mixed"), '
                            'and "reasoning" (one sentence explanation).'
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                "response_format": {"type": "json_object"},
            }

        return self._deep_substitute(template, text)

    def _deep_substitute(self, obj: Any, text: str) -> Any:
        """Recursively replace ``{text}`` placeholders in nested structures."""
        if isinstance(obj, str):
            return obj.replace("{text}", text)
        if isinstance(obj, dict):
            return {k: self._deep_substitute(v, text) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._deep_substitute(item, text) for item in obj]
        return obj

    # ── Response parsing ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_path(data: Any, path: str) -> Any:
        """Walk a dot-notation path through nested dicts/lists.

        Example::

            _resolve_path(data, "choices.0.message.content")
            # → data["choices"][0]["message"]["content"]
        """
        current = data
        for key in path.split("."):
            try:
                if isinstance(current, list):
                    current = current[int(key)]
                elif isinstance(current, dict):
                    current = current[key]
                else:
                    raise SynthError(
                        f"Cannot traverse '{path}': "
                        f"unexpected type {type(current).__name__} at '{key}'"
                    )
            except (KeyError, IndexError, ValueError) as exc:
                raise SynthError(
                    f"Response path '{path}' failed at '{key}': {exc}"
                ) from exc
        return current

    def _parse_response(self, data: dict[str, Any]) -> AuthResult:
        """Extract score / verdict / reasoning from the API response.

        Handles two response shapes:
            1. **Structured API** — score, label, reasoning at direct paths.
            2. **LLM chat API** — a JSON string embedded in the message
               content (e.g. OpenAI ``choices.0.message.content``).
        """
        cfg = self._config

        raw_content = self._resolve_path(data, cfg.score_path)

        # Case 1: LLM response — content is a JSON string
        if isinstance(raw_content, str):
            try:
                parsed = json.loads(raw_content)
                return AuthResult(
                    score=round(float(parsed.get("score", 0.5)), 4),
                    verdict=str(parsed.get("verdict", VERDICT_MIXED)),
                    reasoning=str(
                        parsed.get("reasoning", "No reasoning provided")
                    ),
                    model=cfg.model or cfg.base_url,
                )
            except (json.JSONDecodeError, AttributeError):
                # Not JSON — treat the whole string as reasoning
                return AuthResult(
                    score=0.5,
                    verdict=VERDICT_MIXED,
                    reasoning=raw_content[:500],
                    model=cfg.model or cfg.base_url,
                )

        # Case 2: Structured response — score is numeric
        score = float(raw_content)
        verdict = str(self._resolve_path(data, cfg.label_path))
        reasoning = str(self._resolve_path(data, cfg.reasoning_path))

        return AuthResult(
            score=round(score, 4),
            verdict=verdict,
            reasoning=reasoning,
            model=cfg.model or cfg.base_url,
        )

    # ── Core detection ────────────────────────────────────────────────────

    def detect(self, text: str) -> AuthResult:
        """Send *text* to the remote API and return a verdict.

        Raises:
            SynthError: On empty input, HTTP errors, or unparseable responses.
        """
        if not text.strip():
            raise SynthError("Cannot authenticate empty text")

        payload = self._build_payload(text)

        try:
            response = self._client.post("", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise SynthError(
                f"API request failed ({exc.response.status_code}): "
                f"{exc.response.text[:500]}"
            ) from exc
        except httpx.RequestError as exc:
            raise SynthError(f"API connection error: {exc}") from exc

        return self._parse_response(response.json())

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"api:{self._config.model or self._config.base_url}"

    def close(self) -> None:
        """Shut down the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> UniversalAPIAuthenticator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Vision authenticator — AI image forensics
# ══════════════════════════════════════════════════════════════════════════════


VERDICT_REAL = "real"
VERDICT_FAKE = "fake"


@dataclass(frozen=True)
class VisionAuthResult:
    """Structured result from an AI *image* detection run.

    Unlike :class:`AuthResult` (which analyses text), this captures the
    probability that an image was generated by an AI tool such as
    Midjourney, Stable Diffusion, or DALL·E.

    Attributes:
        ai_probability: ``0.0`` = definitely real, ``1.0`` = definitely
            AI-generated.
        verdict: ``"real"`` or ``"fake"``.
        reasoning: Human-readable explanation.
        model: Identifier of the model that produced this result.
    """

    ai_probability: float
    verdict: str
    reasoning: str
    model: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "ai_probability": self.ai_probability,
            "verdict": self.verdict,
            "reasoning": self.reasoning,
            "model": self.model,
        }


class VisionAuthenticator:
    """Detect AI-generated images using a Vision Transformer (ViT).

    Default model: ``dima806/ai_vs_human_generated_image_detection``
    — a ViT-base model fine-tuned on 80k real/AI-generated images with
    ~98% accuracy.  Detects output from Midjourney, Stable Diffusion,
    DALL·E, and similar generators.

    **Device selection** is fully automatic via :func:`detect_device`::

        cuda  →  NVIDIA GPU (fastest)
        mps   →  Apple Silicon Metal  (M-series Macs)
        cpu   →  Fallback

    Accepts both file paths and raw numpy arrays (e.g. from the PDF
    pipeline)::

        vis = VisionAuthenticator()
        result = vis.detect_file("photo.png")
        result = vis.detect_array(numpy_bgr_array, label="page_1")
    """

    DEFAULT_MODEL = "dima806/ai_vs_human_generated_image_detection"

    # ── Verdict threshold ─────────────────────────────────────────────────
    FAKE_THRESHOLD: float = 0.50

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._device_str = detect_device()
        self._device = get_torch_device()
        self._pipeline = self._load_pipeline()
        logger.info(
            "VisionAuthenticator ready · model=%s · device=%s",
            self._model_name,
            self._device,
        )

    def _load_pipeline(self) -> Any:
        """Load the HuggingFace ``image-classification`` pipeline.

        The ViT model runs natively on MPS, CUDA, and CPU.
        """
        from transformers import pipeline as hf_pipeline  # heavy — deferred

        return hf_pipeline(
            "image-classification",
            model=self._model_name,
            device=self._device,
            top_k=None,  # return scores for all labels
        )

    # ── Label parsing ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_ai_probability(label_scores: dict[str, float]) -> float:
        """Extract the AI-generated probability from the label map.

        Supports label conventions from popular detection models:
            - ``AI-generated`` / ``human``  (dima806)
            - ``Fake`` / ``Real``
            - ``ai`` / ``human``
            - ``LABEL_1`` / ``LABEL_0``
        """
        for ai_label in (
            "AI-generated",
            "Fake",
            "fake",
            "ai",
            "AI",
            "LABEL_1",
            "artificial",
        ):
            if ai_label in label_scores:
                return label_scores[ai_label]

        # Binary fallback: assume second label is AI
        labels = list(label_scores.keys())
        if len(labels) == 2:
            return label_scores[labels[1]]

        return max(label_scores.values())

    def _to_verdict(self, ai_prob: float) -> str:
        """Map a probability to a discrete verdict."""
        return VERDICT_FAKE if ai_prob >= self.FAKE_THRESHOLD else VERDICT_REAL

    @staticmethod
    def _build_reasoning(
        ai_prob: float, verdict: str, raw: dict[str, float]
    ) -> str:
        pct = ai_prob * 100
        raw_str = ", ".join(f"{k}={v:.3f}" for k, v in raw.items())
        return (
            f"AI image probability: {pct:.1f}%. "
            f"Verdict: {verdict}. "
            f"Raw model scores: [{raw_str}]"
        )

    # ── Core detection — from PIL image ───────────────────────────────────

    def _classify(self, image: Any) -> VisionAuthResult:
        """Run classification on a PIL Image and return a result."""
        results = self._pipeline(image)

        label_scores: dict[str, float] = {
            r["label"]: r["score"] for r in results
        }

        ai_prob = self._extract_ai_probability(label_scores)
        verdict = self._to_verdict(ai_prob)

        return VisionAuthResult(
            ai_probability=round(ai_prob, 4),
            verdict=verdict,
            reasoning=self._build_reasoning(ai_prob, verdict, label_scores),
            model=self._model_name,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def detect_file(self, image_path: str | Path) -> VisionAuthResult:
        """Classify an image file as real or AI-generated.

        Args:
            image_path: Path to an image file.

        Returns:
            :class:`VisionAuthResult` with probability and verdict.

        Raises:
            FileNotFoundError: If *image_path* does not exist.
            SynthError: If the image cannot be processed.
        """
        from PIL import Image  # shipped with transformers/torch

        path = Path(image_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            raise SynthError(
                f"Failed to open image '{path.name}': {exc}"
            ) from exc

        return self._classify(img)

    def detect_array(
        self, image_array: Any, *, label: str = "<array>"
    ) -> VisionAuthResult:
        """Classify a numpy BGR array as real or AI-generated.

        This is the entry point for PDF page arrays and in-memory images.

        Args:
            image_array: A ``uint8`` HxWxC BGR numpy array.
            label: Human-readable label for logging.

        Returns:
            :class:`VisionAuthResult`.
        """
        import cv2
        from PIL import Image

        # BGR (OpenCV) → RGB (PIL)
        rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        logger.debug("VisionAuthenticator: classifying %s", label)
        return self._classify(img)

    @property
    def name(self) -> str:
        return f"vision:{self._model_name}"


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════


class DetectorFactory:
    """Create authenticator instances by strategy name.

    Built-in strategies:
        - ``"local"`` → :class:`LocalHFAuthenticator`
        - ``"api"``   → :class:`UniversalAPIAuthenticator`

    Custom strategies can be registered at runtime::

        DetectorFactory.register("my_custom", MyCustomAuth)
        detector = DetectorFactory.create("my_custom", **opts)
    """

    _REGISTRY: dict[str, type[BaseAuthenticator]] = {
        "local": LocalHFAuthenticator,
        "api": UniversalAPIAuthenticator,
    }

    @classmethod
    def create(cls, strategy: str, **kwargs: Any) -> BaseAuthenticator:
        """Instantiate an authenticator by strategy name.

        Args:
            strategy: One of the registered strategy names.
            **kwargs: Forwarded to the authenticator constructor.

        Raises:
            SynthError: If *strategy* is not recognised.
        """
        key = strategy.lower().strip()

        if key not in cls._REGISTRY:
            available = ", ".join(sorted(cls._REGISTRY))
            raise SynthError(
                f"Unknown strategy '{strategy}'. Available: {available}"
            )

        logger.info("DetectorFactory: creating '%s' authenticator", key)
        return cls._REGISTRY[key](**kwargs)

    @classmethod
    def register(
        cls, name: str, authenticator_cls: type[BaseAuthenticator]
    ) -> None:
        """Register a custom authenticator at runtime.

        This allows third-party plugins to extend Synth without
        modifying the source::

            from synth.core.auth import DetectorFactory, BaseAuthenticator

            class MyDetector(BaseAuthenticator):
                ...

            DetectorFactory.register("my_detector", MyDetector)
        """
        cls._REGISTRY[name.lower().strip()] = authenticator_cls
        logger.info("Registered custom strategy: '%s'", name)

    @classmethod
    def available(cls) -> list[str]:
        """Return a sorted list of all registered strategy names."""
        return sorted(cls._REGISTRY)

    @classmethod
    def create_multi(cls, profile: str = "balanced", **kwargs: Any) -> Any:
        """Create a :class:`~synth.core.manager.MultiDetectorManager`.

        This is the recommended entry point for new code.  The manager
        lazy-loads detectors based on the selected profile and returns
        ensemble results.

        Args:
            profile: One of ``"fast"``, ``"balanced"``, or ``"forensic"``.

        Returns:
            A :class:`MultiDetectorManager` instance.
        """
        from synth.core.manager import MultiDetectorManager

        return MultiDetectorManager(profile=profile)


# ══════════════════════════════════════════════════════════════════════════════
#  Legacy adapters — bridge old authenticators into the new detector system
# ══════════════════════════════════════════════════════════════════════════════


def _register_legacy_detectors() -> None:
    """Register the existing authenticators with the new DetectorRegistry.

    Called lazily to avoid circular imports.  Wraps LocalHFAuthenticator
    and VisionAuthenticator as registry-compatible factories.
    """
    from synth.core.registry import (
        DetectorCapability,
        DetectorMetadata,
        DetectorRegistry,
    )

    # Legacy text detector (roberta-base-openai-detector)
    if "legacy-text" not in DetectorRegistry.available():
        DetectorRegistry.register(
            DetectorMetadata(
                name="legacy-text",
                capability=DetectorCapability.TEXT_DETECTION,
                speed_tier="fast",
                requires_gpu=False,
                model_size_mb=500,
                description="RoBERTa OpenAI detector (legacy)",
                weight=1.0,
            ),
            factory=lambda: _LegacyTextAdapter(),
        )

    # Legacy vision detector (ViT)
    if "legacy-vision" not in DetectorRegistry.available():
        DetectorRegistry.register(
            DetectorMetadata(
                name="legacy-vision",
                capability=DetectorCapability.IMAGE_FORENSICS,
                speed_tier="balanced",
                requires_gpu=False,
                model_size_mb=350,
                description="ViT AI image detector (legacy)",
                weight=1.1,
            ),
            factory=lambda: _LegacyVisionAdapter(),
        )


class _LegacyTextAdapter:
    """Adapter: wraps :class:`LocalHFAuthenticator` for the new system."""

    def __init__(self) -> None:
        self._auth: LocalHFAuthenticator | None = None

    def _ensure_loaded(self) -> None:
        if self._auth is None:
            self._auth = LocalHFAuthenticator()

    def detect(self, text: str) -> Any:
        from synth.core.ensemble import DetectorVote

        self._ensure_loaded()
        assert self._auth is not None
        result = self._auth.detect(text)
        return DetectorVote(
            detector_name="legacy-text",
            score=result.score,
            verdict=result.verdict,
            weight=1.0,
        )

    def cleanup(self) -> None:
        self._auth = None

    @property
    def metadata(self) -> Any:
        from synth.core.registry import (
            DetectorCapability,
            DetectorMetadata,
        )

        return DetectorMetadata(
            name="legacy-text",
            capability=DetectorCapability.TEXT_DETECTION,
            speed_tier="fast",
            weight=1.0,
        )


class _LegacyVisionAdapter:
    """Adapter: wraps :class:`VisionAuthenticator` for the new system."""

    def __init__(self) -> None:
        self._auth: VisionAuthenticator | None = None

    def _ensure_loaded(self) -> None:
        if self._auth is None:
            self._auth = VisionAuthenticator()

    def detect_file(self, path: Any) -> Any:
        from synth.core.ensemble import DetectorVote

        self._ensure_loaded()
        assert self._auth is not None
        result = self._auth.detect_file(path)
        return DetectorVote(
            detector_name="legacy-vision",
            score=result.ai_probability,
            verdict=result.verdict,
            weight=1.1,
        )

    def detect_array(self, array: Any, *, label: str = "<array>") -> Any:
        from synth.core.ensemble import DetectorVote

        self._ensure_loaded()
        assert self._auth is not None
        result = self._auth.detect_array(array, label=label)
        return DetectorVote(
            detector_name="legacy-vision",
            score=result.ai_probability,
            verdict=result.verdict,
            weight=1.1,
        )

    def cleanup(self) -> None:
        self._auth = None

    @property
    def metadata(self) -> Any:
        from synth.core.registry import (
            DetectorCapability,
            DetectorMetadata,
        )

        return DetectorMetadata(
            name="legacy-vision",
            capability=DetectorCapability.IMAGE_FORENSICS,
            speed_tier="balanced",
            weight=1.1,
        )
