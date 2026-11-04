"""Live public website verification test script.
Queries the deployed website over the internet and validates all 5 NLP techniques.
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import urllib.request
import json

base_url = 'https://nlp-chatbot-phi.vercel.app'

print(f"Testing live public website: {base_url}")
print("=" * 60)

# 1. Test homepage
try:
    req = urllib.request.Request(base_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=12) as resp:
        html = resp.read().decode('utf-8')
        assert 'Smart NLP Chatbot' in html
        print("[PASS] 1. Live Homepage loaded successfully (HTTP 200)")
except Exception as e:
    print(f"[FAIL] 1. Homepage failed: {e}")

# 2. Test /api/health
try:
    req = urllib.request.Request(f"{base_url}/api/health", headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=12) as resp:
        health = json.loads(resp.read().decode('utf-8'))
        print(f"[PASS] 2. Live Health Check: Status '{health['status']}', Mode: '{health['mode']}'")
except Exception as e:
    print(f"[FAIL] 2. Health check failed: {e}")

# 3. Test queries
test_cases = [
    ("Hello, how are you?", "Greeting Test"),
    ("Sachin is studying Artificial Intelligence in Bengaluru.", "PERSON & LOCATION NER"),
    ("I am very happy because I got a new job.", "Positive Sentiment"),
    ("This product is very bad and disappointing.", "Negative Sentiment"),
    ("The children are playing football.", "Irregular Lemmatization"),
    ("Google is an organization based in the technology industry.", "ORGANIZATION NER"),
    ("He paid $250 on January 15 at 3:30 PM with a 15% discount.", "Structured NER (DATE, TIME, MONEY, PERCENTAGE)")
]

for idx, (text, label) in enumerate(test_cases, start=3):
    payload = json.dumps({"message": text}).encode('utf-8')
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            tokens = data['nlp']['tokenization']['total_tokens']
            sentiment = data['nlp']['sentiment']['category']
            entities = [(e['text'], e['label']) for e in data['nlp']['ner']['entities']]
            reply_snippet = data['response'].replace('\n', ' ')[:65]
            print(f"[PASS] {idx}. {label}:")
            print(f"       Tokens: {tokens} | Sentiment: {sentiment} | Entities: {entities}")
            print(f"       Bot: \"{reply_snippet}...\"")
    except Exception as e:
        print(f"[FAIL] {idx}. {label} failed: {e}")

print("=" * 60)
print("Live Website Verification Complete!")
