"""Comprehensive automated verification test suite for the Smart NLP Chatbot.
Tests all endpoints, all 5 NLP techniques, chatbot response engine, and edge cases.
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import json
import unittest

sys.path.append('.')
from app import app
from nlp import process_text_pipeline


class TestNlpChatbot(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        """Verify /api/health returns 200 with engine statuses."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("engines", data)
        self.assertIn("mode", data)
        print("[PASS] /api/health passed.")

    def test_index_page(self):
        """Verify root / returns 200 HTML content."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Smart NLP Chatbot", res.data)
        self.assertIn(b"Live NLP Dashboard", res.data)
        print("[PASS] / (index.html) passed.")

    def test_test1_greeting(self):
        """TEST 1: 'Hello, how are you?'"""
        msg = "Hello, how are you?"
        res = self.client.post("/api/chat", json={"message": msg})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["nlp"]["tokenization"]["total_tokens"], 6)
        self.assertEqual(data["nlp"]["ner"]["entities"], [])
        self.assertEqual(data["nlp"]["sentiment"]["category"], "NEUTRAL")
        self.assertIn("doing great", data["response"].lower())
        print("[PASS] Test 1 (Greeting) passed.")

    def test_test2_sachin_bengaluru(self):
        """TEST 2: 'Sachin is studying Artificial Intelligence in Bengaluru.'"""
        msg = "Sachin is studying Artificial Intelligence in Bengaluru."
        res = self.client.post("/api/chat", json={"message": msg})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        entities = {e["text"]: e["label"] for e in data["nlp"]["ner"]["entities"]}
        self.assertEqual(entities.get("Sachin"), "PERSON")
        self.assertEqual(entities.get("Bengaluru"), "LOCATION")
        
        # Lemmatization of 'studying' -> 'study'
        lemmas = {r["original"].lower(): r["lemma"].lower() for r in data["nlp"]["lemmatization"]["results"]}
        self.assertEqual(lemmas.get("studying"), "study")
        print("[PASS] Test 2 (Sachin studying AI in Bengaluru) passed.")

    def test_test3_positive_sentiment(self):
        """TEST 3: 'I am very happy because I got a new job.'"""
        msg = "I am very happy because I got a new job."
        res = self.client.post("/api/chat", json={"message": msg})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["nlp"]["sentiment"]["category"], "POSITIVE")
        self.assertGreater(data["nlp"]["sentiment"]["compound"], 0.4)
        print("[PASS] Test 3 (Positive Sentiment) passed.")

    def test_test4_negative_sentiment(self):
        """TEST 4: 'This product is very bad and disappointing.'"""
        msg = "This product is very bad and disappointing."
        res = self.client.post("/api/chat", json={"message": msg})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["nlp"]["sentiment"]["category"], "NEGATIVE")
        self.assertLess(data["nlp"]["sentiment"]["compound"], -0.4)
        print("[PASS] Test 4 (Negative Sentiment) passed.")

    def test_test5_children_playing(self):
        """TEST 5: 'The children are playing football.'"""
        msg = "The children are playing football."
        res = self.client.post("/api/chat", json={"message": msg})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        
        # Verify lemmatization: children -> child, playing -> play
        lemmas = {r["original"].lower(): r["lemma"].lower() for r in data["nlp"]["lemmatization"]["results"]}
        self.assertEqual(lemmas.get("children"), "child")
        self.assertEqual(lemmas.get("playing"), "play")
        
        # Verify stemming: playing -> play
        stems = {r["original"].lower(): r["stemmed"].lower() for r in data["nlp"]["stemming"]["results"]}
        self.assertEqual(stems.get("playing"), "play")
        print("[PASS] Test 5 (Children playing football) passed.")

    def test_test6_google_org(self):
        """TEST 6: 'Google is an organization based in the technology industry.'"""
        msg = "Google is an organization based in the technology industry."
        res = self.client.post("/api/chat", json={"message": msg})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        entities = {e["text"]: e["label"] for e in data["nlp"]["ner"]["entities"]}
        self.assertEqual(entities.get("Google"), "ORGANIZATION")
        print("[PASS] Test 6 (Google Organization) passed.")

    def test_sachin_lives_bengaluru_works_google(self):
        """Example from prompt: 'Sachin lives in Bengaluru and works at Google.'"""
        msg = "Sachin lives in Bengaluru and works at Google."
        res = self.client.post("/api/chat", json={"message": msg})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        entities = {e["text"]: e["label"] for e in data["nlp"]["ner"]["entities"]}
        self.assertEqual(entities.get("Sachin"), "PERSON")
        self.assertEqual(entities.get("Bengaluru"), "LOCATION")
        self.assertEqual(entities.get("Google"), "ORGANIZATION")
        print("[PASS] Prompt example (Sachin lives in Bengaluru and works at Google) passed.")

    def test_entities_structured_types(self):
        """Verify DATE, TIME, MONEY, PERCENTAGE detection."""
        msg = "He paid $250 on January 15 at 3:30 PM with a 15% discount."
        res = self.client.post("/api/chat", json={"message": msg})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        labels = {e["label"] for e in data["nlp"]["ner"]["entities"]}
        self.assertIn("MONEY", labels)
        self.assertIn("DATE", labels)
        self.assertIn("TIME", labels)
        self.assertIn("PERCENTAGE", labels)
        print("[PASS] Structured entities (DATE, TIME, MONEY, PERCENTAGE) passed.")

    def test_edge_cases(self):
        """Verify empty, numbers, single words, punctuation."""
        # Empty string
        res = self.client.post("/api/chat", json={"message": ""})
        self.assertEqual(res.status_code, 200)
        
        # Single word: Better
        res2 = self.client.post("/api/chat", json={"message": "Better"})
        self.assertEqual(res2.status_code, 200)
        d2 = res2.get_json()
        self.assertEqual(d2["nlp"]["lemmatization"]["results"][0]["lemma"].lower(), "good")
        
        # Arbitrary number
        res3 = self.client.post("/api/chat", json={"message": "12345"})
        self.assertEqual(res3.status_code, 200)
        d3 = res3.get_json()
        self.assertEqual(d3["nlp"]["ner"]["entities"], [])
        
        # Punctuation only
        res4 = self.client.post("/api/chat", json={"message": "!@#$%^&*()"})
        self.assertEqual(res4.status_code, 200)
        print("[PASS] Edge cases passed.")


if __name__ == "__main__":
    unittest.main()
