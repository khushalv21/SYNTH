"""DivEye XGBoost classifier wrapper.

Loads a pre-trained XGBoost binary classifier that maps the 10-dimensional
DivEye feature vector to a human/AI classification.

The classifier is shipped as a ``joblib``-serialised model.  If the
pre-trained model is not available, falls back to a simple threshold-based
classification using the raw feature vector statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Feature-based fallback thresholds (derived from DivEye paper analysis)
# mean_surprisal index 0: AI-generated text tends to have lower mean surprisal
FALLBACK_MEAN_SURPRISAL_THRESHOLD = 3.5
# var_surprisal index 2: AI text has lower variance
FALLBACK_VAR_SURPRISAL_THRESHOLD = 8.0


class DivEyeClassifier:
    """Binary classifier over DivEye feature vectors.

    Tries to load an XGBoost model first; falls back to threshold-based
    heuristic if XGBoost is not available or the model file is missing.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self._xgb_model: Any = None
        self._using_fallback = False

        if model_path is not None and model_path.exists():
            self._load_xgb(model_path)
        else:
            self._try_load_xgb_from_package()

        if self._xgb_model is None:
            logger.info(
                "DivEye: XGBoost model not available, using threshold fallback"
            )
            self._using_fallback = True

    def predict_proba(self, features: list[float]) -> float:
        """Return the AI probability for a feature vector.

        Args:
            features: 10-element DivEye feature vector.

        Returns:
            AI probability in [0.0, 1.0].
        """
        if self._using_fallback:
            return self._threshold_classify(features)

        X = np.array([features])
        proba = self._xgb_model.predict_proba(X)
        # proba shape: (1, 2) → [human_prob, ai_prob]
        return float(proba[0, 1])

    # ── Loaders ───────────────────────────────────────────────────────────

    def _load_xgb(self, path: Path) -> None:
        """Load an XGBoost model from a joblib file."""
        try:
            import joblib

            self._xgb_model = joblib.load(path)
            logger.info("DivEye: loaded XGBoost model from %s", path)
        except ImportError:
            logger.warning("DivEye: joblib not available, cannot load model")
        except Exception as exc:
            logger.warning("DivEye: failed to load XGBoost model: %s", exc)

    def _try_load_xgb_from_package(self) -> None:
        """Try to load the bundled XGBoost model from package data."""
        try:
            import importlib.resources as pkg_resources

            data_dir = Path(__file__).parent.parent.parent / "data"
            model_path = data_dir / "diveye_classifier.joblib"
            if model_path.exists():
                self._load_xgb(model_path)
        except Exception:
            pass

    # ── Fallback ──────────────────────────────────────────────────────────

    @staticmethod
    def _threshold_classify(features: list[float]) -> float:
        """Simple threshold-based classification when XGBoost is unavailable.

        Uses mean surprisal and variance as primary discriminants:
        - AI text: lower mean surprisal, lower variance (more predictable)
        - Human text: higher mean surprisal, higher variance (more varied)

        Returns:
            AI probability in [0.0, 1.0].
        """
        if len(features) < 10:
            return 0.5

        mean_s = features[0]
        var_s = features[2]
        kurt_s = features[4]

        # Score components
        score = 0.5

        # Low mean surprisal → more likely AI
        if mean_s < FALLBACK_MEAN_SURPRISAL_THRESHOLD:
            score += 0.15
        elif mean_s > FALLBACK_MEAN_SURPRISAL_THRESHOLD * 1.5:
            score -= 0.15

        # Low variance → more likely AI
        if var_s < FALLBACK_VAR_SURPRISAL_THRESHOLD:
            score += 0.15
        elif var_s > FALLBACK_VAR_SURPRISAL_THRESHOLD * 2:
            score -= 0.15

        # High kurtosis (heavy tails) → more likely human
        if kurt_s > 5.0:
            score -= 0.1
        elif kurt_s < 1.0:
            score += 0.1

        return max(0.0, min(1.0, score))

    @property
    def is_using_fallback(self) -> bool:
        """Return ``True`` if using threshold fallback instead of XGBoost."""
        return self._using_fallback
