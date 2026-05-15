"""Perplexity features + density estimation for Luminol-AI.

Implements the five perplexity-feature types from the paper (Section 3.2):

1. **sum** — total magnitude: ``ppl + ppl_shuf``
2. **diff** — absolute difference: ``ppl_shuf - ppl``
3. **ratio** — relative change: ``ppl_shuf / ppl``
4. **log_ratio** — symmetric relative: ``log(ppl_shuf / ppl)``
5. **pct_change** — percentage: ``(ppl_shuf - ppl) / ppl * 100``

Also implements density-based classification using pre-fitted
Burr/Gamma distribution parameters (Section 3.3-3.4).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Feature extraction ────────────────────────────────────────────────────────

FEATURE_NAMES = ["sum", "diff", "ratio", "log_ratio", "pct_change"]


def extract_perplexity_features(
    ppl_original: float,
    ppl_shuffled: float,
) -> dict[str, float]:
    """Compute the 5 perplexity-feature types.

    Args:
        ppl_original: Perplexity of the original text.
        ppl_shuffled: Perplexity of the shuffled text.

    Returns:
        Dictionary mapping feature names to values.
    """
    # Guard against edge cases
    if ppl_original <= 0 or math.isinf(ppl_original):
        ppl_original = 1.0
    if ppl_shuffled <= 0 or math.isinf(ppl_shuffled):
        ppl_shuffled = 1.0

    return {
        "sum": ppl_original + ppl_shuffled,
        "diff": ppl_shuffled - ppl_original,
        "ratio": ppl_shuffled / ppl_original,
        "log_ratio": math.log(ppl_shuffled / ppl_original),
        "pct_change": (ppl_shuffled - ppl_original) / ppl_original * 100,
    }


# ── Density-based classification ─────────────────────────────────────────────

# Default distribution parameters (pre-fitted on RAID benchmark data)
# These are placeholder values — would be calibrated with actual RAID data
DEFAULT_DISTRIBUTIONS: dict[str, Any] = {
    "diff": {
        "mgt": {"family": "gamma", "shape": 2.5, "loc": 5.0, "scale": 15.0},
        "hgt": {"family": "gamma", "shape": 1.8, "loc": 2.0, "scale": 8.0},
    },
    "ratio": {
        "mgt": {"family": "gamma", "shape": 3.0, "loc": 1.2, "scale": 0.8},
        "hgt": {"family": "gamma", "shape": 2.0, "loc": 1.05, "scale": 0.3},
    },
    "log_ratio": {
        "mgt": {"family": "gamma", "shape": 2.5, "loc": 0.2, "scale": 0.5},
        "hgt": {"family": "gamma", "shape": 1.5, "loc": 0.05, "scale": 0.2},
    },
    "pct_change": {
        "mgt": {"family": "gamma", "shape": 2.0, "loc": 20.0, "scale": 50.0},
        "hgt": {"family": "gamma", "shape": 1.5, "loc": 5.0, "scale": 20.0},
    },
    "sum": {
        "mgt": {"family": "gamma", "shape": 3.0, "loc": 30.0, "scale": 20.0},
        "hgt": {"family": "gamma", "shape": 4.0, "loc": 50.0, "scale": 30.0},
    },
}


def _gamma_pdf(x: float, shape: float, loc: float, scale: float) -> float:
    """Compute Gamma PDF at x (without scipy dependency).

    Uses the standard parameterisation: f(x) = ((x-loc)/scale)^(a-1)
    * exp(-(x-loc)/scale) / (scale * Gamma(a))
    """
    if scale <= 0 or shape <= 0:
        return 0.0
    z = (x - loc) / scale
    if z <= 0:
        return 0.0

    try:
        log_pdf = (
            (shape - 1) * math.log(z)
            - z
            - math.lgamma(shape)
            - math.log(scale)
        )
        return math.exp(log_pdf)
    except (ValueError, OverflowError):
        return 0.0


def classify_with_density(
    features: dict[str, float],
    distributions: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """Classify text using density estimation (Luminol-AI Section 3.4).

    For each feature, evaluate the PDF under both the mgt and hgt
    distributions.  Each feature "votes" for the class with higher
    density.  The final prediction is the majority vote.

    Args:
        features: Dictionary of perplexity feature values.
        distributions: Pre-fitted distribution parameters.
            Defaults to built-in parameters.

    Returns:
        Tuple of (ai_probability, verdict).
    """
    if distributions is None:
        distributions = DEFAULT_DISTRIBUTIONS

    mgt_votes = 0
    hgt_votes = 0
    proba_sum = 0.0
    n_features = 0

    for feat_name in FEATURE_NAMES:
        if feat_name not in features or feat_name not in distributions:
            continue

        z = features[feat_name]
        dist = distributions[feat_name]

        # Evaluate PDF under both classes
        mgt_params = dist["mgt"]
        hgt_params = dist["hgt"]

        pdf_mgt = _gamma_pdf(z, mgt_params["shape"], mgt_params["loc"], mgt_params["scale"])
        pdf_hgt = _gamma_pdf(z, hgt_params["shape"], hgt_params["loc"], hgt_params["scale"])

        total = pdf_mgt + pdf_hgt
        if total > 0:
            p_mgt = pdf_mgt / total
            proba_sum += p_mgt
            n_features += 1

            if p_mgt > 0.5:
                mgt_votes += 1
            else:
                hgt_votes += 1

    if n_features == 0:
        return 0.5, "mixed"

    avg_proba = proba_sum / n_features
    verdict = "ai" if mgt_votes > hgt_votes else "human"

    return avg_proba, verdict


def load_distributions(path: Path | None = None) -> dict[str, Any]:
    """Load pre-fitted distribution parameters from JSON.

    Falls back to built-in defaults if the file is not found.
    """
    if path is None:
        # Try package data
        pkg_data = Path(__file__).parent.parent.parent / "data" / "luminol_distributions.json"
        if pkg_data.exists():
            path = pkg_data

    if path is not None and path.exists():
        with open(path) as f:
            return json.load(f)  # type: ignore[no-any-return]

    logger.debug("Using built-in Luminol distribution parameters")
    return DEFAULT_DISTRIBUTIONS
