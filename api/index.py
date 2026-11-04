"""Vercel Serverless Function Entry Point.
Routes incoming HTTP requests to the core Flask application.
Pre-configures NLTK data directories for the read-only serverless environment.
"""

import os
import sys
import tempfile
import nltk

# Disable pathsec enforcement so NLTK loads safely in serverless environments
try:
    from nltk import pathsec
    pathsec.ENFORCE = False
except Exception:
    pass

# Add project root directory to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Priority 1: Bundled nltk_data inside project repository
bundled_nltk = os.path.join(PROJECT_ROOT, 'nltk_data')
if os.path.exists(bundled_nltk) and bundled_nltk not in nltk.data.path:
    nltk.data.path.insert(0, bundled_nltk)

# Priority 2: Writable directory in /tmp for dynamic fallback
tmp_nltk = os.path.join(tempfile.gettempdir(), 'nltk_data')
try:
    os.makedirs(tmp_nltk, exist_ok=True)
    os.chmod(tmp_nltk, 0o777)
except Exception:
    pass
if tmp_nltk not in nltk.data.path:
    nltk.data.path.append(tmp_nltk)

# If bundled data is absent, attempt lazy download to /tmp
if not os.path.exists(bundled_nltk):
    for pkg in ['punkt_tab', 'wordnet', 'averaged_perceptron_tagger_eng', 'vader_lexicon']:
        try:
            nltk.download(pkg, download_dir=tmp_nltk, quiet=True)
        except Exception:
            pass

from app import app
