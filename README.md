# NLP-Intelligent-Chatbot

An intelligent, interactive Natural Language Processing (NLP) Chatbot and live 5-Technique Analysis Dashboard built with **Python, Flask, NLTK, spaCy, HTML5, CSS3, and JavaScript**.

> **100% Local Processing & Zero External AI APIs**: Operates completely offline without OpenAI, Gemini, Hugging Face, or paid cloud AI APIs. All NLP processing is executed locally on-device.

---

## Table of Contents

- [Project Title & Description](#project-title--description)
- [Project Objectives](#project-objectives)
- [NLP Concepts Implemented](#nlp-concepts-implemented)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Technologies Used](#technologies-used)
- [Installation Instructions](#installation-instructions)
- [How to Run Locally](#how-to-run-locally)
- [Live Deployment Options](#live-deployment-options)
- [Screenshots & UI Preview](#screenshots--ui-preview)
- [Test Verification Suite](#test-verification-suite)
- [Future Improvements](#future-improvements)
- [License & Acknowledgments](#license--acknowledgments)

---

## Project Title & Description

**NLP-Intelligent-Chatbot** is an educational, production-ready, open-source conversational agent that analyzes every user input across five core computational linguistic dimensions while simultaneously providing intelligent, intent-aware conversational responses.

Unlike conventional chatbots that either rely on expensive, opaque cloud LLM APIs or merely regurgitate static predefined strings, this application executes authentic tokenization, morphological stemming, part-of-speech-aware lemmatization, 7-class named entity extraction, and sentiment polarity scoring on every incoming message.

---

## Project Objectives

1. **Demonstrate 5 Core NLP Techniques**: Expose the computational mechanisms behind Tokenization, Stemming, Lemmatization, Named Entity Recognition, and Sentiment Analysis.
2. **Deterministic & Offline Architecture**: Eliminate reliance on external APIs, API keys, subscription models, and network latency.
3. **Real-Time Interactive Visualization**: Present a modern dark-mode dual-panel interface displaying conversational bubbles alongside live linguistic breakdown cards.
4. **Intelligent Rule-Based Conversational Engine**: Deliver contextual responses to greetings, conceptual questions, sentiment feedback, and benchmark inquiries.
5. **Production Readiness**: Provide complete unit tests, reproducible environment setup, clean Git repository structure, and free cloud deployment support.

---

## NLP Concepts Implemented

### 1. Tokenization
- **Library**: NLTK `word_tokenize` with a regex-based fallback.
- **Mechanism**: Splits unstructured text into words, punctuation marks, and numbers.
- **Metrics**: Computes total tokens, word counts, and punctuation tallies.

### 2. Stemming
- **Library**: NLTK `PorterStemmer` (Martin Porter's 1980 algorithm).
- **Mechanism**: Applies cascading morphological suffix stripping to reduce inflections to stems (e.g., *Playing* → *play*, *Studies* → *studi*, *Running* → *run*).
- **Presentation**: Comparative tabular view indicating original tokens, stems, and transformation status.

### 3. Lemmatization
- **Library**: NLTK `WordNetLemmatizer` enhanced by `nltk.pos_tag`.
- **Mechanism**: Maps Penn Treebank POS tags to WordNet constants (`NOUN`, `VERB`, `ADJ`, `ADV`) to resolve genuine dictionary lemmas (e.g., *Children* → *Child*, *Running* → *Run*, *Better* → *Good*).
- **Presentation**: POS-tagged breakdown with canonical dictionary lemmas.

### 4. Named Entity Recognition (NER)
- **Library**: Hybrid engine combining local **spaCy** (`en_core_web_sm`), **NLTK `ne_chunk`**, and specialized regex extractors.
- **Categories Detected**:
  - `PERSON`: Human names (e.g., *Sachin*, *Alice*, *Alan Turing*)
  - `LOCATION`: Cities, regions, countries (e.g., *Bengaluru*, *New York*, *India*)
  - `ORGANIZATION`: Companies, institutions (e.g., *Google*, *OpenAI*, *NASA*)
  - `DATE`: Absolute and relative dates (e.g., *January 15*, *today*, *2026/11/03*)
  - `TIME`: Clock times (e.g., *3:30 PM*, *14:30*)
  - `MONEY`: Currency and amounts (e.g., *$250*, *50 euros*, *₹500*)
  - `PERCENTAGE`: Numeric percentages (e.g., *95%*, *15%*)
- **Integrity**: Excludes false positives and outputs *"No entities detected"* when none exist.

### 5. Sentiment Analysis
- **Library**: NLTK **VADER** (`SentimentIntensityAnalyzer`) with rule-based lexicon fallback.
- **Classification**: Classifies text into `POSITIVE`, `NEGATIVE`, or `NEUTRAL`.
- **Metrics**: Normalized compound score (`-1.0000` to `+1.0000`), positive ratio (`pos`), negative ratio (`neg`), and neutral ratio (`neu`).
- **Visualization**: Animated color-coded progress meters and mood indicators.

---

## Key Features

- **Split-Screen UX**: Left chat interaction stream paired with right live NLP analysis cards.
- **Deep Dark Theme (`#09090B`)**: High contrast, elegant typography, purple (`#8B5CF6`) and blue (`#3B82F6`) accents.
- **Instant JSON Export**: Single-click "Copy JSON" button with toast confirmation.
- **Responsive Navigation**: Full mobile/tablet responsiveness with collapsible sidebars.
- **Clickable Prompt Chips**: Preloaded benchmark examples for instant testing.
- **Session History**: Fast switching between past queries in the current session.

---

## Project Architecture

```text
User Browser (HTML5 / Modern Dark CSS3 / Vanilla JavaScript)
                        │
                        ▼  HTTP POST /api/chat  (JSON)
               Flask Backend Server (app.py)
                        │
                        ▼
            Unified NLP Pipeline (nlp/__init__.py)
     ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
     ▼             ▼             ▼             ▼             ▼             ▼
1. Tokenizer  2. Stemmer    3. Lemmatizer  4. NER Engine  5. Sentiment  6. Chatbot
(word_tokenize) (PorterStemmer) (WordNet+POS) (spaCy+NLTK)   (VADER)     (Intent+Rules)
     │             │             │             │             │             │
     └─────────────┴─────────────┼─────────────┴─────────────┴─────────────┘
                                 ▼
                     Unified JSON Response
                                 │
                        ▲────────┘
                        ▼
         Interactive Chat Bubbles + Live 5-Card Dashboard
```

---

## Technologies Used

- **Backend**: Python 3.10+ (tested on Python 3.13)
- **Web Framework**: Flask, Flask-CORS
- **WSGI Production Server**: Gunicorn
- **NLP Libraries**:
  - `nltk` (Tokenizers, PorterStemmer, WordNet, POS Tagger, MaxEnt NE Chunker, VADER)
  - `spacy` + `en_core_web_sm` (Statistical Named Entity Recognition)
  - `numpy`
- **Frontend**:
  - HTML5 (Semantic Structure)
  - CSS3 (Custom Glassmorphism, CSS Variables, Flexbox/Grid)
  - JavaScript (Vanilla ES6, Asynchronous Fetch API, Zero external UI libraries)

---

## Installation Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Sachin2716/NLP-Intelligent-Chatbot.git
cd NLP-Intelligent-Chatbot
```

### 2. Create and Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLP Corpora for Offline Operation

Run the automated corpora setup script once:

```bash
python setup_nltk.py
```

---

## How to Run Locally

### Start the Flask Server

```bash
python app.py
```

Access the application in your browser:
```text
http://127.0.0.1:5000
```

---

## Live Deployment Options

### Option 1: Deploy to Render (Recommended Free Hosting)
1. Fork or push this repository to your GitHub account: `https://github.com/Sachin2716/NLP-Intelligent-Chatbot`.
2. Visit [Render.com](https://render.com) and log in.
3. Click **New +** → **Blueprint** (or **Web Service**).
4. Select the `NLP-Intelligent-Chatbot` repository.
5. Render detects `render.yaml` automatically:
   - **Build Command**: `pip install -r requirements.txt && python setup_nltk.py`
   - **Start Command**: `gunicorn app:app`
6. Click **Deploy**. Your live URL will be active in minutes (e.g., `https://nlp-intelligent-chatbot.onrender.com`).

### Option 2: Deploy to Vercel
1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel` in the project root.
3. The project includes `vercel.json` configured for serverless Python WSGI execution.

---

## Screenshots & UI Preview

```text
┌─────────────────────────┬──────────────────────────────────┬─────────────────────────────┐
│ 🧠 Smart NLP            │  💬 Chat Stream                  │  📊 Live NLP Dashboard      │
│                         │                                  │                             │
│ [+ New Chat]            │  👤 "Sachin lives in Bengaluru   │  [1] Tokenization           │
│                         │      and works at Google."       │      9 tokens | 7 words     │
│ Recent Queries:         │                                  │                             │
│ • Sachin at Google      │  🤖 "Sachin is in great company  │  [2] Stemming (Porter)      │
│ • Children playing      │      working at Google in        │      works -> work          │
│ • Happy new job         │      Bengaluru! 🏢..."           │                             │
│                         │                                  │  [3] Lemmatization          │
│ 5 NLP Techniques:       │                                  │      lives -> live (Verb)   │
│ 1. Tokenization         │                                  │                             │
│ 2. Stemming             │                                  │  [4] Named Entity (NER)     │
│ 3. Lemmatization        │                                  │      Sachin -> PERSON       │
│ 4. NER                  │                                  │      Bengaluru -> LOCATION  │
│ 5. Sentiment            │                                  │      Google -> ORGANIZATION │
│                         │                                  │                             │
│ 🟢 100% Offline         │  [ Type your message...    ] [>] │  [5] Sentiment (VADER)      │
│    Zero External APIs   │                                  │      NEUTRAL (0.0000)       │
└─────────────────────────┴──────────────────────────────────┴─────────────────────────────┘
```

---

## Test Verification Suite

An automated test suite with 11 distinct scenarios is provided in `test_app.py`:

```bash
python test_app.py
```

### Verified Test Benchmark Results:

| Test Case | Tokens | Stemming Sample | Lemmatization Sample | Detected NER Entities | Sentiment Polarity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **"Hello, how are you?"** | 6 | `Hello` → `hello` | `Hello` → `Hello` (Noun) | *No entities detected* | `NEUTRAL` (0.0000) |
| **"Sachin is studying Artificial Intelligence in Bengaluru."** | 9 | `studying` → `studi` | `studying` → `study` (Verb) | `Sachin` (PERSON)<br>`Bengaluru` (LOCATION) | `POSITIVE` (+0.4767) |
| **"I am very happy because I got a new job."** | 11 | `happy` → `happi` | `happy` → `happy` (Adj) | *No entities detected* | `POSITIVE` (+0.6115) |
| **"This product is very bad and disappointing."** | 8 | `disappointing` → `disappoint` | `bad` → `bad` (Adj) | *No entities detected* | `NEGATIVE` (-0.8051) |
| **"The children are playing football."** | 6 | `playing` → `play` | `children` → `child` (Noun)<br>`playing` → `play` (Verb) | *No entities detected* | `POSITIVE` (+0.2023) |
| **"Google is an organization based in the technology industry."** | 10 | `Google` → `googl` | `Google` → `Google` (Noun) | `Google` (ORGANIZATION) | `NEUTRAL` (0.0000) |
| **"He paid $250 on January 15 at 3:30 PM with a 15% discount."** | 14 | `discount` → `discount` | `paid` → `pay` (Verb) | `January 15` (DATE)<br>`3:30 PM` (TIME)<br>`$250` (MONEY)<br>`15%` (PERCENTAGE) | `NEUTRAL` (0.0000) |

---

## Future Improvements

- [ ] Add Dependency Parsing and Constituent Tree diagrams in an interactive SVG canvas.
- [ ] Incorporate multi-lingual support (Spanish, French, Hindi) via Polyglot or Stanza.
- [ ] Add TF-IDF and Bag-of-Words vectorizer visualizer.
- [ ] Implement audio voice input and Text-to-Speech (TTS) using Web Speech API.

---

## License & Acknowledgments

- **License**: MIT Open Source License.
- Built with [NLTK](https://www.nltk.org/), [spaCy](https://spacy.io/), and [Flask](https://flask.palletsprojects.com/).
