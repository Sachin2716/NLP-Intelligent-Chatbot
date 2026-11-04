"""Lemmatization module using NLTK WordNetLemmatizer with POS tagging.
Transforms words into meaningful dictionary base forms (lemmas) based on context.
"""

import string
import re

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    _LEMMATIZER = WordNetLemmatizer()
    _WORDNET_AVAILABLE = True
except Exception:
    _LEMMATIZER = None
    _WORDNET_AVAILABLE = False


def _get_wordnet_pos(treebank_tag: str) -> tuple:
    """Map Penn Treebank POS tag to WordNet POS constant and readable name.

    Returns:
        tuple: (wordnet_pos, readable_category)
    """
    if treebank_tag.startswith('J'):
        return ('a', "Adjective")
    elif treebank_tag.startswith('V'):
        return ('v', "Verb")
    elif treebank_tag.startswith('N'):
        return ('n', "Noun")
    elif treebank_tag.startswith('R'):
        return ('r', "Adverb")
    else:
        # Default fallback to Noun for content words
        return ('n', "Noun")


# Common irregular lookup fallback in case WordNet or POS tag needs a boost
_IRREGULAR_LEMMAS = {
    "better": ("good", "Adjective"),
    "best": ("good", "Adjective"),
    "worse": ("bad", "Adjective"),
    "worst": ("bad", "Adjective"),
    "children": ("child", "Noun"),
    "mice": ("mouse", "Noun"),
    "geese": ("goose", "Noun"),
    "feet": ("foot", "Noun"),
    "teeth": ("tooth", "Noun"),
    "people": ("person", "Noun"),
    "men": ("man", "Noun"),
    "women": ("woman", "Noun"),
}


def lemmatize_text(tokens: list, raw_text: str = "") -> dict:
    """Lemmatize tokens with POS-tag awareness.

    Args:
        tokens (list): List of token strings.
        raw_text (str): Original input text for POS tagging context.

    Returns:
        dict: Lemmatization results with original, POS tag, category, and lemma.
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

    # Obtain POS tags
    pos_tagged = []
    try:
        pos_tagged = nltk.pos_tag(tokens)
    except Exception:
        # Fallback if tagger data missing
        pos_tagged = [(tok, "NN") for tok in tokens]

    results = []
    changed_count = 0
    unchanged_count = 0

    for idx, (token, tag) in enumerate(pos_tagged):
        is_punct = token in punct_set or all(c in punct_set for c in token)
        is_num = token.replace('.', '', 1).replace(',', '', 1).isdigit()

        if is_punct:
            results.append({
                "original": token,
                "lemma": token,
                "pos_tag": tag,
                "pos_category": "Punctuation",
                "changed": False,
                "is_word": False
            })
            unchanged_count += 1
            continue

        if is_num:
            results.append({
                "original": token,
                "lemma": token,
                "pos_tag": tag,
                "pos_category": "Number",
                "changed": False,
                "is_word": False
            })
            unchanged_count += 1
            continue

        wn_pos, category = _get_wordnet_pos(tag)
        token_lower = token.lower()

        # Check special irregular table first for specific single-word queries
        lemma_found = None
        if token_lower in _IRREGULAR_LEMMAS:
            lemma_found, irregular_cat = _IRREGULAR_LEMMAS[token_lower]
            category = irregular_cat

        if lemma_found is None:
            global _LEMMATIZER, _WORDNET_AVAILABLE
            if _LEMMATIZER is None:
                try:
                    from nltk.stem import WordNetLemmatizer
                    _LEMMATIZER = WordNetLemmatizer()
                    _WORDNET_AVAILABLE = True
                except Exception:
                    pass
            if _WORDNET_AVAILABLE and _LEMMATIZER is not None:
                try:
                    lemma_found = _LEMMATIZER.lemmatize(token_lower, pos=wn_pos)
                except Exception:
                    lemma_found = token_lower

        if lemma_found is None:
            lemma_found = token_lower

        # Preserve capitalization style if original was Capitalized
        if token.istitle() and len(lemma_found) > 0:
            display_lemma = lemma_found.capitalize()
        elif token.isupper() and len(token) > 1:
            display_lemma = lemma_found.upper()
        else:
            display_lemma = lemma_found

        changed = (token.lower() != lemma_found.lower())
        if changed:
            changed_count += 1
        else:
            unchanged_count += 1

        results.append({
            "original": token,
            "lemma": display_lemma,
            "pos_tag": tag,
            "pos_category": category,
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
