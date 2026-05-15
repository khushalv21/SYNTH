"""Perplexity computation for Luminol-AI.

Computes text perplexity using a small causal LM (GPT-2) with
sliding-window context.  This is used to measure the perplexity
of both the original and shuffled text.
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


def compute_perplexity(
    text: str,
    model: torch.nn.Module,
    tokenizer: object,
    *,
    max_length: int = 1024,
    stride: int = 512,
) -> float:
    """Compute the perplexity of *text* under the given causal LM.

    Uses a sliding-window approach with stride to handle texts longer
    than the model's context window.

    Args:
        text: Input text.
        model: A ``transformers.AutoModelForCausalLM`` instance.
        tokenizer: The matching tokenizer.
        max_length: Model context window size.
        stride: Sliding window stride (typically max_length // 2).

    Returns:
        The perplexity (float).  Returns ``float('inf')`` if the text
        is too short to compute.
    """
    encodings = tokenizer.encode(  # type: ignore[union-attr]
        text,
        return_tensors="pt",
        truncation=False,
    )

    seq_len = encodings.shape[1]
    if seq_len < 2:
        return float("inf")

    device = next(model.parameters()).device

    nlls: list[float] = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end  # tokens to score in this window

        input_ids = encodings[:, begin:end].to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        # Shift logits/labels to align
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]

        # Only score the non-overlapping portion
        score_start = max(0, logits.shape[1] - target_len)
        log_probs = torch.log_softmax(logits[:, score_start:, :].float(), dim=-1)
        scored_labels = labels[:, score_start:]

        nll = -log_probs[0, range(scored_labels.shape[1]), scored_labels[0]]
        nlls.extend(nll.cpu().tolist())

        prev_end = end
        if end == seq_len:
            break

    if not nlls:
        return float("inf")

    avg_nll = sum(nlls) / len(nlls)
    return math.exp(avg_nll)
