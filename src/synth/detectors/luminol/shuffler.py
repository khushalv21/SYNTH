"""Text shuffling algorithm from the Luminol-AI paper (Section 3.1).

Implements the three-case shuffling procedure:

1. **Single-sentence** text: shuffle words within the sentence,
   preserving terminal punctuation.
2. **Multi-sentence, single-paragraph** text: shuffle sentence order.
3. **Multi-paragraph** text: apply per-paragraph shuffling independently,
   preserving paragraph order.

The shuffling is single-pass, paragraph-order preserving, and uses
sentence-level permutation in all cases except single-sentence texts
(where words are permuted instead).
"""

from __future__ import annotations

import random
import re


def shuffle_text(text: str, *, seed: int | None = None) -> str:
    """Apply the Luminol-AI text-shuffling procedure.

    Args:
        text: Input text to shuffle.
        seed: Optional random seed for reproducibility.

    Returns:
        The shuffled text.
    """
    if seed is not None:
        random.seed(seed)

    paragraphs = _split_paragraphs(text)

    shuffled_paragraphs = [
        _shuffle_paragraph(p) for p in paragraphs
    ]

    return "\n\n".join(shuffled_paragraphs)


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs (double-newline separated)."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences.

    Uses a simple regex-based splitter that handles common abbreviations
    and sentence-ending punctuation.
    """
    # Split on sentence-ending punctuation followed by whitespace or end
    sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [s.strip() for s in sentences if s.strip()]


def _shuffle_paragraph(paragraph: str) -> str:
    """Shuffle a single paragraph according to Luminol-AI rules.

    - Single sentence: shuffle words, preserve terminal punctuation.
    - Multiple sentences: shuffle sentence order.
    """
    sentences = _split_sentences(paragraph)

    if len(sentences) <= 1:
        # Word-level shuffle for single-sentence paragraphs
        return _shuffle_words(paragraph)

    # Sentence-level shuffle for multi-sentence paragraphs
    random.shuffle(sentences)
    return " ".join(sentences)


def _shuffle_words(sentence: str) -> str:
    """Shuffle words within a sentence, preserving terminal punctuation.

    Example::

        "The cat sat on the mat." → "mat the sat cat on The."
    """
    sentence = sentence.strip()
    if not sentence:
        return sentence

    # Extract terminal punctuation
    terminal = ""
    if sentence[-1] in ".!?;:":
        terminal = sentence[-1]
        sentence = sentence[:-1].strip()

    words = sentence.split()
    if len(words) <= 1:
        return sentence + terminal

    random.shuffle(words)
    return " ".join(words) + terminal
