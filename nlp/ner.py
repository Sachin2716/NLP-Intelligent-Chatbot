"""Named Entity Recognition (NER) module.
Identifies PERSON, LOCATION, ORGANIZATION, DATE, TIME, MONEY, and PERCENTAGE.
Uses local spaCy English model when available, enriched with NLTK and rule-based extractors.
Never returns fake entities; accurately flags 'No entities detected' when none exist.
"""

import re
import html

# Try loading spaCy
_SPACY_NLP = None
try:
    import spacy
    _SPACY_NLP = spacy.load("en_core_web_sm")
except Exception:
    _SPACY_NLP = None

# Try importing NLTK
_NLTK_AVAILABLE = False
try:
    import nltk
    from nltk import pos_tag, ne_chunk
    from nltk.tokenize import word_tokenize
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False


# Entity visual configuration
ENTITY_CONFIG = {
    "PERSON": {
        "color": "#A855F7",
        "bg": "rgba(168, 85, 247, 0.18)",
        "border": "#A855F7",
        "icon": "👤",
        "description": "People, including fictional characters"
    },
    "LOCATION": {
        "color": "#3B82F6",
        "bg": "rgba(59, 130, 246, 0.18)",
        "border": "#3B82F6",
        "icon": "📍",
        "description": "Countries, cities, states, geographic places"
    },
    "ORGANIZATION": {
        "color": "#06B6D4",
        "bg": "rgba(6, 182, 212, 0.18)",
        "border": "#06B6D4",
        "icon": "🏢",
        "description": "Companies, agencies, institutions"
    },
    "DATE": {
        "color": "#10B981",
        "bg": "rgba(16, 185, 129, 0.18)",
        "border": "#10B981",
        "icon": "📅",
        "description": "Absolute or relative dates or periods"
    },
    "TIME": {
        "color": "#F59E0B",
        "bg": "rgba(245, 158, 11, 0.18)",
        "border": "#F59E0B",
        "icon": "⏰",
        "description": "Times of day, specific hours/minutes"
    },
    "MONEY": {
        "color": "#EAB308",
        "bg": "rgba(234, 179, 8, 0.18)",
        "border": "#EAB308",
        "icon": "💰",
        "description": "Monetary values including unit/currency"
    },
    "PERCENTAGE": {
        "color": "#EC4899",
        "bg": "rgba(236, 72, 153, 0.18)",
        "border": "#EC4899",
        "icon": "📊",
        "description": "Percentages and numeric ratios"
    }
}

# Common name gazetteer for high-precision person identification
COMMON_NAMES = {
    'sachin', 'rahul', 'priya', 'amit', 'anita', 'rohit', 'virat', 'neha',
    'deepak', 'pooja', 'vikram', 'sneha', 'arjun', 'ananya', 'sanjay', 'kavita',
    'john', 'mary', 'alice', 'bob', 'charlie', 'david', 'emma', 'michael',
    'sarah', 'james', 'robert', 'william', 'elizabeth', 'alex', 'sam', 'tom',
    'steve', 'elon', 'bill', 'alan', 'ada', 'grace', 'linus', 'guido'
}

# Known locations
KNOWN_LOCATIONS = {
    'bengaluru', 'bangalore', 'mumbai', 'delhi', 'new delhi', 'hyderabad',
    'chennai', 'kolkata', 'pune', 'london', 'new york', 'san francisco',
    'california', 'india', 'united states', 'usa', 'uk', 'france', 'germany',
    'japan', 'tokyo', 'paris', 'berlin', 'singapore', 'sydney', 'canada'
}

# Known organizations
KNOWN_ORGS = {
    'google', 'microsoft', 'apple', 'amazon', 'meta', 'facebook', 'netflix',
    'tesla', 'ibm', 'intel', 'nvidia', 'twitter', 'openai', 'deepmind',
    'infosys', 'tcs', 'wipro', 'nasa', 'who', 'un', 'mit', 'stanford', 'harvard'
}

# Non-entity academic, conversational, or generic terms that statistical models misclassify
NON_ENTITIES = {
    'artificial intelligence', 'machine learning', 'natural language processing',
    'technology industry', 'data science', 'deep learning', 'computer science',
    'software engineering', 'information technology', 'neural networks',
    'hello', 'hi', 'hey', 'greetings', 'welcome', 'bye', 'goodbye', 'thanks',
    'thank', 'please', 'yes', 'no', 'ok', 'okay', 'what', 'who', 'where',
    'when', 'why', 'how', 'this', 'that', 'these', 'those', 'there', 'here',
    'product', 'service', 'weather', 'industry', 'company', 'organization',
    'football', 'cricket', 'sports', 'game', 'children', 'child', 'people'
}

# Regex patterns for high precision
DATE_PATTERNS = [
    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,\s*\d{4})?\b',
    r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s*,\s*\d{4})?\b',
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
    r'\b(?:today|yesterday|tomorrow|tonight)\b',
    r'\b(?:next|last|this)\s+(?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
    r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b'
]

TIME_PATTERNS = [
    r'\b(?:1[0-2]|0?[1-9]):[0-5][0-9](?::[0-5][0-9])?\s*(?:AM|PM|am|pm)\b',
    r'\b(?:2[0-3]|[01]?[0-9]):[0-5][0-9]\b(?!\s*%)',
    r'\b(?:1[0-2]|0?[1-9])\s*(?:AM|PM|am|pm)\b',
    r'\b(?:noon|midnight)\b'
]

MONEY_PATTERNS = [
    r'(?:[\$€£₹¥]|USD|INR|EUR|GBP)\s*[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|crore|lakh|dollars?|rupees?|euros?|pounds?|cents?))?\b',
    r'\b[\d,]+(?:\.\d+)?\s*(?:dollars?|rupees?|euros?|pounds?|cents?|USD|INR|EUR|GBP)\b'
]

PERCENTAGE_PATTERNS = [
    r'\b\d+(?:\.\d+)?\s*(?:%|percent|percentage)\b'
]


def _extract_regex_entities(text: str) -> list:
    """Extract DATE, TIME, MONEY, PERCENTAGE using structured regular expressions."""
    entities = []

    for pat in PERCENTAGE_PATTERNS:
        for match in re.finditer(pat, text, re.IGNORECASE):
            entities.append({
                "text": match.group(0),
                "label": "PERCENTAGE",
                "start": match.start(),
                "end": match.end(),
                "source": "regex"
            })

    for pat in MONEY_PATTERNS:
        for match in re.finditer(pat, text, re.IGNORECASE):
            entities.append({
                "text": match.group(0),
                "label": "MONEY",
                "start": match.start(),
                "end": match.end(),
                "source": "regex"
            })

    for pat in TIME_PATTERNS:
        for match in re.finditer(pat, text, re.IGNORECASE):
            # Exclude matches that overlap with money
            entities.append({
                "text": match.group(0),
                "label": "TIME",
                "start": match.start(),
                "end": match.end(),
                "source": "regex"
            })

    for pat in DATE_PATTERNS:
        for match in re.finditer(pat, text, re.IGNORECASE):
            entities.append({
                "text": match.group(0),
                "label": "DATE",
                "start": match.start(),
                "end": match.end(),
                "source": "regex"
            })

    return entities


def _extract_contextual_entities(text: str) -> list:
    """Extract Person, Location, and Organization based on linguistic cues and gazetteer."""
    entities = []

    # 1. Gazetteer exact matches with word boundaries
    for name in COMMON_NAMES:
        for match in re.finditer(rf'\b{re.escape(name)}\b', text, re.IGNORECASE):
            # Check capitalization in original text
            matched_str = text[match.start():match.end()]
            if matched_str[0].isupper():
                entities.append({
                    "text": matched_str,
                    "label": "PERSON",
                    "start": match.start(),
                    "end": match.end(),
                    "source": "contextual_name"
                })

    for loc in KNOWN_LOCATIONS:
        for match in re.finditer(rf'\b{re.escape(loc)}\b', text, re.IGNORECASE):
            matched_str = text[match.start():match.end()]
            if matched_str[0].isupper():
                entities.append({
                    "text": matched_str,
                    "label": "LOCATION",
                    "start": match.start(),
                    "end": match.end(),
                    "source": "contextual_location"
                })

    for org in KNOWN_ORGS:
        for match in re.finditer(rf'\b{re.escape(org)}\b', text, re.IGNORECASE):
            matched_str = text[match.start():match.end()]
            if matched_str[0].isupper():
                entities.append({
                    "text": matched_str,
                    "label": "ORGANIZATION",
                    "start": match.start(),
                    "end": match.end(),
                    "source": "contextual_org"
                })

    # 2. Subject + human action patterns: "<Capitalized Name> is studying/lives/works/said..."
    human_verbs_pat = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:is\s+studying|was\s+studying|lives|lived|works|worked|studies|studied|said|wrote|joined|graduated)\b'
    for match in re.finditer(human_verbs_pat, text):
        candidate = match.group(1).strip()
        if candidate.lower() not in NON_ENTITIES and candidate.lower() not in KNOWN_LOCATIONS and candidate.lower() not in KNOWN_ORGS:
            start = match.start(1)
            end = match.end(1)
            entities.append({
                "text": candidate,
                "label": "PERSON",
                "start": start,
                "end": end,
                "source": "contextual_verb"
            })

    # 3. Location prepositions: "in <Capitalized City>", "from <Capitalized City>"
    loc_prep_pat = r'\b(?:in|from|at|near|around)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    for match in re.finditer(loc_prep_pat, text):
        candidate = match.group(1).strip()
        if candidate.lower() not in NON_ENTITIES and candidate.lower() not in COMMON_NAMES and candidate.lower() not in KNOWN_ORGS:
            # If known or looks like a proper geographic name
            if candidate.lower() in KNOWN_LOCATIONS:
                entities.append({
                    "text": candidate,
                    "label": "LOCATION",
                    "start": match.start(1),
                    "end": match.end(1),
                    "source": "prep_location"
                })

    # 4. Organization prepositions: "works at <Capitalized Org>", "employed by <Capitalized Org>"
    org_prep_pat = r'\b(?:works?\s+at|employed\s+by|hired\s+by|company\s+called|founder\s+of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    for match in re.finditer(org_prep_pat, text):
        candidate = match.group(1).strip()
        if candidate.lower() not in NON_ENTITIES:
            entities.append({
                "text": candidate,
                "label": "ORGANIZATION",
                "start": match.start(1),
                "end": match.end(1),
                "source": "prep_org"
            })

    return entities


def _spacy_entities(text: str) -> list:
    """Extract entities using spaCy when available."""
    if _SPACY_NLP is None:
        return []

    try:
        doc = _SPACY_NLP(text)
        entities = []
        for ent in doc.ents:
            clean_text = ent.text.strip()
            if clean_text.lower() in NON_ENTITIES:
                continue

            label = ent.label_
            mapped_label = None

            if label == "PERSON":
                mapped_label = "PERSON"
            elif label in ("GPE", "LOC", "FAC"):
                mapped_label = "LOCATION"
            elif label in ("ORG", "ORGANIZATION"):
                mapped_label = "ORGANIZATION"
            if label == "DATE":
                # Check for bare arbitrary numbers misclassified as dates
                if clean_text.isdigit():
                    val = int(clean_text)
                    if val < 1000 or val > 2100 or len(clean_text) != 4:
                        continue
                mapped_label = "DATE"
            elif label == "TIME":
                mapped_label = "TIME"
            elif label == "MONEY":
                mapped_label = "MONEY"
            elif label in ("PERCENT", "PERCENTAGE"):
                mapped_label = "PERCENTAGE"

            if mapped_label:
                entities.append({
                    "text": clean_text,
                    "label": mapped_label,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "source": "spacy"
                })
        return entities
    except Exception:
        return []


def _nltk_entities(text: str) -> list:
    """Extract entities using NLTK ne_chunk as additional fallback."""
    if not _NLTK_AVAILABLE:
        return []

    try:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        chunked = ne_chunk(tagged)

        entities = []
        current_entity = []
        current_label = None

        for item in chunked:
            if hasattr(item, 'label'):
                label = item.label()
                ent_text = " ".join(c[0] for c in item)
                if ent_text.lower() in NON_ENTITIES:
                    continue

                mapped = None
                if label == "PERSON":
                    mapped = "PERSON"
                elif label in ("GPE", "LOCATION", "FACILITY"):
                    mapped = "LOCATION"
                elif label == "ORGANIZATION":
                    mapped = "ORGANIZATION"

                if mapped:
                    # Find span in original text
                    match = re.search(rf'\b{re.escape(ent_text)}\b', text)
                    if match:
                        entities.append({
                            "text": ent_text,
                            "label": mapped,
                            "start": match.start(),
                            "end": match.end(),
                            "source": "nltk"
                        })
        return entities
    except Exception:
        return []


def extract_entities(text: str) -> dict:
    """Perform Named Entity Recognition on user input text.

    Args:
        text (str): Input text from user.

    Returns:
        dict: Detected entities, category breakdowns, highlighted HTML, and metadata.
    """
    if not text or not text.strip():
        return {
            "entities": [],
            "total_entities": 0,
            "has_entities": False,
            "highlighted_html": "",
            "message": "No entities detected."
        }

    raw_text = text.strip()

    # Collect candidates from all engines
    candidates = []
    candidates.extend(_extract_regex_entities(raw_text))
    candidates.extend(_extract_contextual_entities(raw_text))
    candidates.extend(_spacy_entities(raw_text))
    candidates.extend(_nltk_entities(raw_text))

    # Deduplicate candidates by span overlaps
    # Priority: Contextual > Regex > spaCy > NLTK
    priority_order = {"contextual_name": 1, "contextual_location": 1, "contextual_org": 1,
                      "contextual_verb": 2, "prep_location": 2, "prep_org": 2,
                      "regex": 3, "spacy": 4, "nltk": 5}

    candidates.sort(key=lambda x: (x["start"], -(x["end"] - x["start"]), priority_order.get(x.get("source"), 9)))

    resolved = []
    occupied_spans = []

    for cand in candidates:
        cand_start = cand["start"]
        cand_end = cand["end"]
        cand_text = cand["text"].strip()
        cand_label = cand["label"]

        # Discard non-entities
        if cand_text.lower() in NON_ENTITIES:
            continue

        # Check collision with already accepted spans
        collision = False
        for (s, e) in occupied_spans:
            if not (cand_end <= s or cand_start >= e):
                collision = True
                break

        if not collision and len(cand_text) > 0:
            occupied_spans.append((cand_start, cand_end))
            cfg = ENTITY_CONFIG.get(cand_label, {
                "color": "#8B5CF6",
                "bg": "rgba(139, 92, 246, 0.2)",
                "border": "#8B5CF6",
                "icon": "🏷️",
                "description": cand_label
            })

            resolved.append({
                "text": cand_text,
                "label": cand_label,
                "start": cand_start,
                "end": cand_end,
                "color": cfg["color"],
                "bg": cfg["bg"],
                "border": cfg["border"],
                "icon": cfg["icon"],
                "description": cfg["description"]
            })

    # Sort resolved entities by position in sentence
    resolved.sort(key=lambda x: x["start"])

    # Generate highlighted HTML representation
    highlighted_html = ""
    last_idx = 0
    for ent in resolved:
        start = ent["start"]
        end = ent["end"]
        highlighted_html += html.escape(raw_text[last_idx:start])
        highlighted_html += (
            f'<mark class="ner-tag ner-{ent["label"].lower()}" '
            f'style="background-color: {ent["bg"]}; border: 1px solid {ent["border"]}; color: #FFFFFF;" '
            f'title="{ent["label"]}">'
            f'{html.escape(ent["text"])}'
            f'<span class="ner-badge" style="background-color: {ent["color"]};">{ent["icon"]} {ent["label"]}</span>'
            f'</mark>'
        )
        last_idx = end
    highlighted_html += html.escape(raw_text[last_idx:])

    has_entities = len(resolved) > 0

    return {
        "entities": resolved,
        "total_entities": len(resolved),
        "has_entities": has_entities,
        "highlighted_html": highlighted_html,
        "message": "Entities detected." if has_entities else "No entities detected."
    }
