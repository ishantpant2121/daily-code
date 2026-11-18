"""Tokenization and word n-gram extraction helpers."""

import re


def tokenize(
    document: str,
    pattern: re.Pattern[str],
    *,
    lowercase: bool,
    stop_words: frozenset[str],
) -> list[str]:
    """Tokenize one document, applying case normalization and stop-word removal."""
    source = document.lower() if lowercase else document
    matches = pattern.findall(source)
    tokens = [matches] if isinstance(matches, str) else matches
    return [token for token in tokens if token and token not in stop_words]


def extract_ngrams(tokens: list[str], ngram_range: tuple[int, int]) -> list[str]:
    """Return all contiguous word n-grams in ascending n-gram size."""
    minimum, maximum = ngram_range
    features: list[str] = []
    for size in range(minimum, maximum + 1):
        for start in range(len(tokens) - size + 1):
            features.append(" ".join(tokens[start : start + size]))
    return features
