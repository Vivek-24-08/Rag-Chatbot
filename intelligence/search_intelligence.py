"""Rule-based routing and opt-in, query-free analytics; not a trained learning model."""
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
import json
import re
import sqlite3
import time
import uuid
from utils.logger import get_logger
logger = get_logger(__name__)
INTENTS = {
    "costs": ("copay", "coinsurance", "deductible", "premium", "cost", "out of pocket"),
    "coverage": ("cover", "covered", "benefit", "eligible"),
    "claims": ("claim", "bill", "reimbursement"),
    "appeals": ("appeal", "grievance", "complaint", "denial"),
    "providers": ("provider", "doctor", "specialist", "network", "referral"),
    "pharmacy": ("drug", "medication", "prescription", "pharmacy", "formulary"),
}
FOLLOW = re.compile(r"^(what about|how about|and\b|does (it|that|this)\b|is (it|that|this)\b)", re.I)

@dataclass(frozen=True)
class ProcessedQuery:
    original: str
    search_text: str
    intent: str
    was_rewritten: bool
    needs_clarification: bool = False

    @property
    def weights(self):
        return (0.8, 1.2) if self.intent in {"costs", "pharmacy", "appeals"} else (1.0, 1.0)

class SearchIntelligence:
    def __init__(self, database_path, enabled=False, retention_days=30):
        base = Path(database_path)
        self.database_path = str(base.with_name(base.stem + "-v2.db"))
        self.enabled = enabled
        self.retention_days = retention_days
        self.last_error = ""
        if enabled:
            try:
                base.parent.mkdir(parents=True, exist_ok=True)
                with self._db() as db:
                    db.executescript("""
                    CREATE TABLE IF NOT EXISTS events(
                        id TEXT PRIMARY KEY, session TEXT, intent TEXT, latency REAL,
                        sources TEXT, status TEXT, created REAL);
                    CREATE TABLE IF NOT EXISTS ratings(
                        event TEXT PRIMARY KEY REFERENCES events(id) ON DELETE CASCADE,
                        rating INTEGER CHECK(rating IN (-1,1)));
                    CREATE INDEX IF NOT EXISTS event_time ON events(created);
                    """)
                self.prune()
            except Exception as exc:
                self._failed(exc)
                self.enabled = False

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.database_path, timeout=5)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _failed(self, exc):
        self.last_error = type(exc).__name__
        logger.warning("analytics_unavailable error=%s", self.last_error)

    def process(self, question, history=None):
        original = re.sub(r"\s+", " ", question).strip()
        if not original or len(original) > 4000:
            raise ValueError("Enter a question between 1 and 4,000 characters.")
        previous = [t.get("content", "") for t in (history or []) if t.get("role") == "user"][-6:]
        antecedents = []
        if FOLLOW.match(original):
            for turn in reversed(previous):
                antecedents.insert(0, turn[:1000])
                if not FOLLOW.match(turn):
                    break
        rewritten = bool(antecedents)
        text = " Previous question: ".join(antecedents + [original]) if rewritten else original
        scores = {intent: sum(bool(re.search(r"\b" + re.escape(term) + r"s?\b", text.lower()))
                              for term in terms) for intent, terms in INTENTS.items()}
        intent, score = max(scores.items(), key=lambda item: item[1])
        return ProcessedQuery(original, text, intent if score else "general", rewritten,
                              bool(FOLLOW.match(original) and not antecedents))

    def prune(self):
        if self.enabled:
            with self._db() as db:
                db.execute("DELETE FROM events WHERE created < ?", (time.time() - self.retention_days * 86400,))

    def record_search(self, session_id, processed, latency_ms, sources, status="answered"):
        if not self.enabled:
            return ""
        event_id = str(uuid.uuid4())
        try:
            self.prune()
            with self._db() as db:
                db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                           (event_id, session_id, processed.intent, latency_ms,
                            json.dumps(sorted(set(sources))), status, time.time()))
            return event_id
        except Exception as exc:
            self._failed(exc)
            return ""

    def record_feedback(self, search_id, rating, session_id=None):
        if rating not in (-1, 1):
            raise ValueError("Feedback must be +1 or -1.")
        if not self.enabled or not search_id:
            return False
        try:
            with self._db() as db:
                event = db.execute("SELECT session FROM events WHERE id=?", (search_id,)).fetchone()
                if not event or (session_id is not None and event[0] != session_id):
                    return False
                db.execute("INSERT OR REPLACE INTO ratings VALUES (?,?)", (search_id, rating))
            return True
        except Exception as exc:
            self._failed(exc)
            return False

    def preferred_sources(self, session_id):
        if not self.enabled or not session_id:
            return {}
        try:
            self.prune()
            with self._db() as db:
                rows = db.execute("""SELECT sources,rating FROM events JOIN ratings ON events.id=ratings.event
                                     WHERE session=?""", (session_id,)).fetchall()
            totals = {}
            for sources, rating in rows:
                for identity in json.loads(sources):
                    totals[identity] = totals.get(identity, 0) + rating
            return {key: max(-1, min(1, value)) for key, value in totals.items()}
        except Exception as exc:
            self._failed(exc)
            return {}

    def forget_session(self, session_id):
        if self.enabled:
            try:
                with self._db() as db:
                    db.execute("DELETE FROM events WHERE session=?", (session_id,))
            except Exception as exc:
                self._failed(exc)

    def summary(self):
        empty = {"enabled": self.enabled, "searches": 0, "positive_rate": None,
                 "average_latency_ms": None, "no_evidence": 0, "errors": 0, "intents": {}}
        if not self.enabled:
            return empty
        try:
            self.prune()
            with self._db() as db:
                rows = db.execute("SELECT intent,latency,status FROM events").fetchall()
                ratings = [r[0] for r in db.execute("SELECT rating FROM ratings")]
            empty.update(searches=len(rows), positive_rate=sum(v > 0 for v in ratings)/len(ratings) if ratings else None,
                         average_latency_ms=sum(r[1] for r in rows)/len(rows) if rows else None,
                         no_evidence=sum(r[2] == "no_evidence" for r in rows),
                         errors=sum(r[2] == "error" for r in rows))
            for intent, _, _ in rows:
                empty["intents"][intent] = empty["intents"].get(intent, 0) + 1
            return empty
        except Exception as exc:
            self._failed(exc)
            return empty
