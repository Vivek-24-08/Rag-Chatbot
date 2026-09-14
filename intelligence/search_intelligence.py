"""Conservative, local query understanding and feedback storage."""
from dataclasses import dataclass
import json
import os
import re
import sqlite3
import time
import uuid

INTENTS = {
    "costs": ("copay", "coinsurance", "deductible", "premium", "cost", "out of pocket"),
    "coverage": ("cover", "covered", "benefit", "eligible"),
    "claims": ("claim", "bill", "reimbursement"),
    "appeals": ("appeal", "grievance", "complaint", "denial"),
    "providers": ("provider", "doctor", "specialist", "network", "referral"),
    "pharmacy": ("drug", "medication", "prescription", "pharmacy", "formulary"),
}

@dataclass(frozen=True)
class ProcessedQuery:
    original: str
    search_text: str
    intent: str
    was_rewritten: bool

class SearchIntelligence:
    def __init__(self, database_path, enabled=True):
        self.database_path, self.enabled = database_path, enabled
        if enabled:
            os.makedirs(os.path.dirname(database_path) or ".", exist_ok=True)
            with sqlite3.connect(database_path) as db:
                db.executescript("""CREATE TABLE IF NOT EXISTS searches (id TEXT PRIMARY KEY, session_id TEXT, query TEXT, rewritten_query TEXT, intent TEXT, latency_ms REAL, sources TEXT, created_at REAL);
                CREATE TABLE IF NOT EXISTS feedback (id TEXT PRIMARY KEY, search_id TEXT, rating INTEGER, created_at REAL);""")

    def process(self, question, history=None):
        original = re.sub(r"\s+", " ", question).strip()
        scores = {intent: sum(term in original.lower() for term in terms) for intent, terms in INTENTS.items()}
        intent, score = max(scores.items(), key=lambda item: item[1])
        intent = intent if score else "general"
        previous = next((turn.get("content", "") for turn in reversed(history or []) if turn.get("role") == "user"), "")
        follow_up = bool(re.match(r"^(what about|how about|and |does (it|that|this)|is (it|that|this))\b", original.lower()))
        return ProcessedQuery(original, f"{previous} Follow-up: {original}" if follow_up and previous else original, intent, follow_up and bool(previous))

    def record_search(self, session_id, processed, latency_ms, sources):
        event_id = str(uuid.uuid4())
        if self.enabled:
            with sqlite3.connect(self.database_path) as db:
                db.execute("INSERT INTO searches VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (event_id, session_id, processed.original, processed.search_text, processed.intent, latency_ms, json.dumps(list(dict.fromkeys(sources))), time.time()))
        return event_id

    def record_feedback(self, search_id, rating):
        if self.enabled and search_id and rating in (-1, 1):
            with sqlite3.connect(self.database_path) as db:
                db.execute("INSERT INTO feedback VALUES (?, ?, ?, ?)", (str(uuid.uuid4()), search_id, rating, time.time()))

    def preferred_sources(self, session_id):
        if not self.enabled:
            return set()
        with sqlite3.connect(self.database_path) as db:
            rows = db.execute("SELECT s.sources FROM searches s JOIN feedback f ON s.id=f.search_id WHERE s.session_id=? GROUP BY s.id HAVING SUM(f.rating)>0", (session_id,)).fetchall()
        return {source for row in rows for source in json.loads(row[0])}

    def summary(self):
        if not self.enabled:
            return {"searches": 0, "positive_rate": None}
        with sqlite3.connect(self.database_path) as db:
            searches = db.execute("SELECT COUNT(*) FROM searches").fetchone()[0]
            ratings = [row[0] for row in db.execute("SELECT rating FROM feedback")]
        return {"searches": searches, "positive_rate": sum(value > 0 for value in ratings) / len(ratings) if ratings else None}
