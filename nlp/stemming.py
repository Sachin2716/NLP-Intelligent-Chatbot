"""Stemming module using NLTK PorterStemmer.
Converts words to their grammatical root or base stem using the Porter algorithm.
"""

import string
import re

try:
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()
    _PORTER_AVAILABLE = True
except Exception:
    _STEMMER = None
    _PORTER_AVAILABLE = False


def _fallback_stem(word: str) -> str:
    """Simple heuristic fallback if NLTK is unavailable."""
    w = word.lower()
    suffixes = [
        ('sses', 'ss'), ('ies', 'i'), ('ss', 'ss'), ('s', ''),
        ('eed', 'ee'), ('ing', ''), ('ed', ''), ('ly', ''),
        ('tion', 'te'), ('tional', 'tion'), ('ement', '')
    ]
    for suf, repl in suffixes:
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)] + repl
    return w


def stem_text(tokens: list) -> dict:
    """Apply PorterStemmer to a list of tokens.

    Args:
        tokens (list): List of token strings.

    Returns:
        dict: Stemming results containing pairs of (original, stemmed), stats, and table data.
    """
    if not tokens:
        return {
            "results": [],
            "total_processed": 0,
            "changed_count": 0,
            "unchanged_count": 0,
            "status": "empty"
        }

    punct_set = set(string.punctuation).union({'“', '”', '‘', '’', '—', '–', '…', '¿', '¡', '«', '»'})
    results = []
    changed_count = 0
    unchanged_count = 0

    for token in tokens:
        is_punct = token in punct_set or all(c in punct_set for c in token)
        is_num = token.replace('.', '', 1).replace(',', '', 1).isdigit()

        if is_punct or is_num:
            results.append({
                "original": token,
                "stemmed": token,
                "changed": False,
                "is_word": False
            })
            unchanged_count += 1
            continue

        if _PORTER_AVAILABLE and _STEMMER is not None:
            stemmed = _STEMMER.stem(token)
        else:
            stemmed = _fallback_stem(token)

        # Compare case-insensitively or as stemmed
        changed = (token.lower() != stemmed.lower())
        if changed:
            changed_count += 1
        else:
            unchanged_count += 1

        results.append({
            "original": token,
            "stemmed": stemmed,
            "changed": changed,
            "is_word": True
        })

    return {
        "results": results,
        "total_processed": len(tokens),
        "changed_count": changed_count,
        "unchanged_count": unchanged_count,
        "status": "success"
    }
