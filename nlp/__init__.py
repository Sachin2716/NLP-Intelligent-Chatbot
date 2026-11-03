"""NLP package initialization and pipeline coordinator.
Runs all five NLP techniques (Tokenization, Stemming, Lemmatization, NER, Sentiment Analysis)
and generates conversational responses.
"""

from datetime import datetime
from .tokenizer import tokenize_text
from .stemming import stem_text
from .lemmatization import lemmatize_text
from .ner import extract_entities
from .sentiment import analyze_sentiment
from .chatbot import generate_response

__all__ = [
    "tokenize_text",
    "stem_text",
    "lemmatize_text",
    "extract_entities",
    "analyze_sentiment",
    "generate_response",
    "process_text_pipeline"
]


def process_text_pipeline(text: str) -> dict:
    """Execute all five NLP techniques and generate a conversational response.

    Args:
        text (str): User input text.

    Returns:
        dict: Complete analysis and conversational response payload.
    """
    raw_text = text if text is not None else ""

    # 1. Tokenization
    token_data = tokenize_text(raw_text)
    tokens = token_data.get("tokens", [])

    # 2. Stemming
    stem_data = stem_text(tokens)

    # 3. Lemmatization (POS-aware)
    lemma_data = lemmatize_text(tokens, raw_text)

    # 4. Named Entity Recognition
    ner_data = extract_entities(raw_text)

    # 5. Sentiment Analysis
    sentiment_data = analyze_sentiment(raw_text)

    # Consolidated NLP dictionary
    nlp_results = {
        "tokenization": token_data,
        "stemming": stem_data,
        "lemmatization": lemma_data,
        "ner": ner_data,
        "sentiment": sentiment_data
    }

    # Generate chatbot response
    bot_reply = generate_response(raw_text, nlp_results)

    return {
        "input_text": raw_text,
        "response": bot_reply,
        "nlp": nlp_results,
        "timestamp": datetime.now().strftime("%I:%M %p"),
        "status": "success"
    }
