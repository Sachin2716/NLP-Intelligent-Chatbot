"""Sentiment Analysis module using NLTK VADER with rule-based fallback.
Classifies text into POSITIVE, NEGATIVE, or NEUTRAL with detailed polarity scores.
"""

import re

_VADER_ANALYZER = None
_VADER_AVAILABLE = False

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _VADER_ANALYZER = SentimentIntensityAnalyzer()
    _VADER_AVAILABLE = True
except Exception:
    _VADER_ANALYZER = None
    _VADER_AVAILABLE = False


# Local fallback lexicon for completely offline resilience
FALLBACK_POSITIVE_WORDS = {
    'love', 'loved', 'loving', 'happy', 'great', 'good', 'excellent', 'fantastic',
    'wonderful', 'amazing', 'best', 'superb', 'delightful', 'joy', 'excited',
    'pleased', 'awesome', 'nice', 'cool', 'brilliant', 'perfect', 'fabulous',
    'stellar', 'beautiful', 'glad', 'satisfied', 'terrific', 'splendid'
}

FALLBACK_NEGATIVE_WORDS = {
    'terrible', 'bad', 'horrible', 'awful', 'disappointing', 'disappointed',
    'hate', 'hated', 'hating', 'worst', 'poor', 'sad', 'angry', 'frustrated',
    'annoyed', 'disaster', 'ugly', 'useless', 'broken', 'failure', 'fail',
    'dreadful', 'pathetic', 'worthless', 'rubbish', 'garbage', 'pain'
}

INTENSIFIERS = {'very', 'extremely', 'really', 'super', 'so', 'hugely', 'absolutely'}
NEGATIONS = {'not', "n't", 'never', 'no', 'hardly', 'barely', 'scarcely'}


def _fallback_sentiment(text: str) -> dict:
    """Rule-based sentiment analyzer if VADER is unavailable."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    pos_score = 0.0
    neg_score = 0.0
    is_negated = False

    for i, w in enumerate(words):
        multiplier = 1.0
        # Check previous word for intensifiers or negations
        if i > 0:
            prev = words[i - 1]
            if prev in INTENSIFIERS:
                multiplier = 1.5
            if prev in NEGATIONS:
                is_negated = not is_negated

        if w in FALLBACK_POSITIVE_WORDS:
            if is_negated:
                neg_score += 1.0 * multiplier
            else:
                pos_score += 1.0 * multiplier
            is_negated = False
        elif w in FALLBACK_NEGATIVE_WORDS:
            if is_negated:
                pos_score += 1.0 * multiplier
            else:
                neg_score += 1.0 * multiplier
            is_negated = False

    total_emotional = pos_score + neg_score
    total_words = len(words)

    if total_emotional == 0:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    compound = (pos_score - neg_score) / (pos_score + neg_score + 1.0)
    pos_ratio = round(pos_score / (total_words + 1.0), 3)
    neg_ratio = round(neg_score / (total_words + 1.0), 3)
    neu_ratio = round(max(0.0, 1.0 - (pos_ratio + neg_ratio)), 3)

    return {
        "neg": neg_ratio,
        "neu": neu_ratio,
        "pos": pos_ratio,
        "compound": round(compound, 4)
    }


def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment polarity of user text.

    Args:
        text (str): Input text from user.

    Returns:
        dict: Detailed sentiment results including category, compound, pos/neg/neu scores,
              color, and visual indicators.
    """
    if not text or not text.strip():
        return {
            "category": "NEUTRAL",
            "compound": 0.0,
            "pos": 0.0,
            "neg": 0.0,
            "neu": 1.0,
            "pos_percent": 0.0,
            "neg_percent": 0.0,
            "neu_percent": 100.0,
            "color": "#9CA3AF",
            "icon": "😐",
            "mood": "Neutral",
            "method": "none",
            "status": "empty"
        }

    raw_text = text.strip()
    scores = None
    method = "vader"

    global _VADER_ANALYZER, _VADER_AVAILABLE
    if not _VADER_AVAILABLE or _VADER_ANALYZER is None:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            _VADER_ANALYZER = SentimentIntensityAnalyzer()
            _VADER_AVAILABLE = True
        except Exception:
            pass

    if _VADER_AVAILABLE and _VADER_ANALYZER is not None:
        try:
            scores = _VADER_ANALYZER.polarity_scores(raw_text)
        except Exception:
            scores = _fallback_sentiment(raw_text)
            method = "fallback_lexicon"
    else:
        scores = _fallback_sentiment(raw_text)
        method = "fallback_lexicon"

    compound = round(float(scores.get("compound", 0.0)), 4)
    pos = round(float(scores.get("pos", 0.0)), 3)
    neg = round(float(scores.get("neg", 0.0)), 3)
    neu = round(float(scores.get("neu", 1.0)), 3)

    # Classification using standard VADER thresholds
    if compound >= 0.05:
        category = "POSITIVE"
        color = "#10B981"  # Emerald green
        bg = "rgba(16, 185, 129, 0.15)"
        icon = "😊"
        if compound >= 0.6:
            mood = "Strongly Positive"
        else:
            mood = "Moderately Positive"
    elif compound <= -0.05:
        category = "NEGATIVE"
        color = "#EF4444"  # Red
        bg = "rgba(239, 68, 68, 0.15)"
        icon = "😞"
        if compound <= -0.6:
            mood = "Strongly Negative"
        else:
            mood = "Moderately Negative"
    else:
        category = "NEUTRAL"
        color = "#60A5FA"  # Soft Blue
        bg = "rgba(96, 165, 250, 0.15)"
        icon = "😐"
        mood = "Balanced / Neutral"

    return {
        "category": category,
        "compound": compound,
        "compound_display": f"{compound:+.4f}",
        "pos": pos,
        "neg": neg,
        "neu": neu,
        "pos_percent": round(pos * 100, 1),
        "neg_percent": round(neg * 100, 1),
        "neu_percent": round(neu * 100, 1),
        "color": color,
        "bg": bg,
        "icon": icon,
        "mood": mood,
        "method": method,
        "status": "success"
    }
