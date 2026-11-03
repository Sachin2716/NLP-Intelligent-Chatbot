"""Flask web application for the Smart NLP Chatbot.
Serves the modern dark-themed interactive chat interface and exposes local NLP endpoints.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from nlp import process_text_pipeline
from nlp.tokenizer import tokenize_text
from nlp.stemming import stem_text
from nlp.lemmatization import lemmatize_text
from nlp.ner import extract_entities
from nlp.sentiment import analyze_sentiment

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    """Render the primary chatbot interface."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Process a user chat message through all 5 NLP techniques and return response."""
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({
            "error": "Invalid request. 'message' field is required.",
            "status": "error"
        }), 400

    user_message = data.get("message", "")
    try:
        result = process_text_pipeline(user_message)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "server_error"
        }), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_only():
    """Standalone endpoint to run all 5 NLP techniques on any raw input text."""
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({
            "error": "Invalid request. 'text' field is required.",
            "status": "error"
        }), 400

    raw_text = data.get("text", "")
    try:
        tokens_data = tokenize_text(raw_text)
        tokens = tokens_data.get("tokens", [])

        stem_data = stem_text(tokens)
        lemma_data = lemmatize_text(tokens, raw_text)
        ner_data = extract_entities(raw_text)
        sentiment_data = analyze_sentiment(raw_text)

        return jsonify({
            "text": raw_text,
            "nlp": {
                "tokenization": tokens_data,
                "stemming": stem_data,
                "lemmatization": lemma_data,
                "ner": ner_data,
                "sentiment": sentiment_data
            },
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "server_error"
        }), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Check the availability and status of local NLP engines."""
    from nlp.tokenizer import _NLTK_TOKENIZE_AVAILABLE
    from nlp.stemming import _PORTER_AVAILABLE
    from nlp.lemmatization import _WORDNET_AVAILABLE
    from nlp.ner import _SPACY_NLP, _NLTK_AVAILABLE
    from nlp.sentiment import _VADER_AVAILABLE

    return jsonify({
        "status": "healthy",
        "engines": {
            "tokenization": "NLTK punkt" if _NLTK_TOKENIZE_AVAILABLE else "Regex Fallback",
            "stemming": "NLTK PorterStemmer" if _PORTER_AVAILABLE else "Rule Fallback",
            "lemmatization": "NLTK WordNet + POS Tagger" if _WORDNET_AVAILABLE else "Lookup Fallback",
            "ner": "spaCy (en_core_web_sm) + NLTK + Contextual Rules" if _SPACY_NLP is not None else "NLTK + Contextual Rules",
            "sentiment": "NLTK VADER" if _VADER_AVAILABLE else "Lexicon Fallback"
        },
        "mode": "100% Local / Zero API Calls"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Smart NLP Chatbot on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)
