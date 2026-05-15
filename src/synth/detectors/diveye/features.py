"""DivEye feature extraction — surprisal statistics from a proxy LM.

Port of IBM's ``diveye_utils.py``, adapted for single-text inference
(instead of CSV batch processing) with robust short-text handling.

The feature vector is a 10-dimensional array of surprisal statistics:

1. mean_surprisal
2. std_surprisal
3. var_surprisal
4. skew_surprisal
5. kurtosis_surprisal
6. mean_diff_surprisal
7. std_diff_surprisal
8. var_2nd_order_diff
9. entropy_2nd_order_diff
10. autocorrelation_2nd_order_diff
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Minimum tokens for a valid feature extraction
MIN_TOKENS = 10


class DivEyeFeatureExtractor:
    """Extract DivEye surprisal features from text using a causal LM.

    Args:
        model: A ``transformers.AutoModelForCausalLM`` instance.
        tokenizer: The matching tokenizer.
    """

    def __init__(self, model: torch.nn.Module, tokenizer: object) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def compute(self, text: str) -> list[float] | None:
        """Compute the 10-feature DivEye vector for *text*.

        Returns:
            A list of 10 floats, or ``None`` if the text is too short
            for meaningful analysis.
        """
        surprisals = self._surprisal(text)

        if surprisals is None or len(surprisals) < MIN_TOKENS:
            logger.warning(
                "DivEye: text too short for analysis (%d tokens < %d min)",
                0 if surprisals is None else len(surprisals),
                MIN_TOKENS,
            )
            return None

        return self._extract_features(surprisals)

    # ── Internal ──────────────────────────────────────────────────────────

    def _log_likelihoods(self, text: str) -> np.ndarray | None:
        """Compute per-token log-likelihoods using the proxy LM."""
        tokens = self._tokenizer.encode(  # type: ignore[union-attr]
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        if tokens.shape[1] < 3:
            return None

        device = next(self._model.parameters()).device
        tokens = tokens.to(device)

        with torch.no_grad():
            outputs = self._model(tokens, labels=tokens)

        logits = outputs.logits
        shift_logits = logits[:, :-1, :].squeeze(0)
        shift_labels = tokens[:, 1:].squeeze(0)

        log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
        token_log_likelihoods = (
            log_probs[range(shift_labels.shape[0]), shift_labels]
            .cpu()
            .numpy()
        )
        return token_log_likelihoods

    def _surprisal(self, text: str) -> np.ndarray | None:
        """Compute per-token surprisal (negative log-likelihood)."""
        ll = self._log_likelihoods(text)
        if ll is None:
            return None
        return -ll

    @staticmethod
    def _extract_features(surprisals: np.ndarray) -> list[float]:
        """Extract the 10-dimensional feature vector from surprisals."""
        from scipy.stats import entropy, kurtosis, skew

        s = np.array(surprisals)

        # Basic surprisal statistics
        mean_s = float(np.mean(s))
        std_s = float(np.std(s))
        var_s = float(np.var(s))
        skew_s = float(skew(s))
        kurt_s = float(kurtosis(s))

        # First-order differences
        diff_s = np.diff(s)
        mean_diff = float(np.mean(diff_s))
        std_diff = float(np.std(diff_s))

        # Second-order differences (of log-likelihoods)
        ll = -s  # back to log-likelihoods
        first_order = np.diff(ll)
        second_order = np.diff(first_order)

        if len(second_order) < 2:
            var_2nd = 0.0
            entropy_2nd = 0.0
            autocorr_2nd = 0.0
        else:
            var_2nd = float(np.var(second_order))
            hist, _ = np.histogram(second_order, bins=20, density=True)
            entropy_2nd = float(entropy(hist + 1e-10))
            autocorr_2nd = float(
                np.corrcoef(second_order[:-1], second_order[1:])[0, 1]
            ) if len(second_order) > 1 else 0.0

            # Handle NaN from constant arrays
            if np.isnan(autocorr_2nd):
                autocorr_2nd = 0.0

        return [
            mean_s, std_s, var_s, skew_s, kurt_s,
            mean_diff, std_diff,
            var_2nd, entropy_2nd, autocorr_2nd,
        ]
