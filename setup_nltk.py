"""NLTK and model setup script.
Downloads all necessary NLTK corpora and models locally for offline operation.
Run once during project installation or build step:
    python setup_nltk.py
"""

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import nltk

REQUIRED_PACKAGES = [
    'punkt',
    'punkt_tab',
    'wordnet',
    'omw-1.4',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',
    'maxent_ne_chunker',
    'maxent_ne_chunker_tab',
    'words',
    'vader_lexicon'
]


def setup_all():
    print("=" * 60)
    print("Setting up local NLP resources for SMART NLP Chatbot...")
    print("=" * 60)

    success_count = 0
    for pkg in REQUIRED_PACKAGES:
        try:
            print(f"Checking / downloading '{pkg}'...")
            nltk.download(pkg, quiet=True)
            success_count += 1
            print(f"  [OK] '{pkg}' ready.")
        except Exception as e:
            print(f"  [WARN] Failed to download '{pkg}': {e}")

    print("-" * 60)
    print(f"NLTK setup complete: {success_count}/{len(REQUIRED_PACKAGES)} packages verified.")

    # Check spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("  [OK] spaCy model 'en_core_web_sm' is installed and ready.")
    except Exception:
        print("  [INFO] spaCy 'en_core_web_sm' not found. Local NLTK & rule-based engine will be used.")

    print("=" * 60)
    print("NLP engines are ready for offline operation!")
    print("=" * 60)


if __name__ == "__main__":
    setup_all()
