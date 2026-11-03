"""Tokenization module using NLTK word_tokenize with robust fallback.
Breaks user sentences into individual tokens, detects punctuation, and computes metrics.
"""

import string
import re

try:
    import nltk
    from nltk.tokenize import word_tokenize
    _NLTK_TOKENIZE_AVAILABLE = True
except Exception:
    _NLTK_TOKENIZE_AVAILABLE = False


def tokenize_text(text: str) -> dict:
    """Tokenize the input text into individual words and punctuation tokens.

    Args:
        text (str): Input text from user.

    Returns:
        dict: Token analysis including tokens list, counts, and categorization.
    """
    if not text or not text.strip():
        return {
            "tokens": [],
            "total_tokens": 0,
            "word_count": 0,
            "punctuation_count": 0,
            "details": [],
            "message": "Empty input provided."
        }

    raw_text = text.strip()
    tokens = []

    if _NLTK_TOKENIZE_AVAILABLE:
        try:
            tokens = word_tokenize(raw_text)
        except Exception:
            # Fallback regex tokenizer if punkt data is missing
            tokens = re.findall(r"\w+|[^\w\s]", raw_text, re.UNICODE)
    else:
        tokens = re.findall(r"\w+|[^\w\s]", raw_text, re.UNICODE)

    details = []
    word_count = 0
    punctuation_count = 0
    number_count = 0

    punct_set = set(string.punctuation).union({'“', '”', '‘', '’', '—', '–', '…', '¿', '¡', '«', '»'})

    for token in tokens:
        if token in punct_set or all(c in punct_set for c in token):
            token_type = "punctuation"
            punctuation_count += 1
        elif token.replace('.', '', 1).replace(',', '', 1).isdigit():
            token_type = "number"
            number_count += 1
        else:
            token_type = "word"
            word_count += 1

        details.append({
            "token": token,
            "type": token_type
        })

    return {
        "tokens": tokens,
        "total_tokens": len(tokens),
        "word_count": word_count,
        "punctuation_count": punctuation_count,
        "number_count": number_count,
        "details": details,
        "status": "success"
    }
