"""Read-only installation diagnostics. Never displays credential values."""
import importlib.util
import sqlite3
import sys

def main():
    failed = False
    print("Python:", sys.version.split()[0])
    if not (3, 11) <= sys.version_info[:2] <= (3, 12):
        print("Use a fresh Python 3.11 or 3.12 environment (not Anaconda base 3.9.7).")
        failed = True
    for module in ("streamlit", "fitz", "chromadb", "langchain_openai", "langchain_text_splitters",
                   "sentence_transformers", "tiktoken", "dotenv"):
        present = importlib.util.find_spec(module) is not None
        print(module + ": " + ("installed" if present else "MISSING"))
        failed |= not present
    try:
        db = sqlite3.connect(":memory:")
        db.execute("CREATE VIRTUAL TABLE test USING fts5(text)")
        db.close()
        print("SQLite FTS5: ready")
        from utils.config import settings
        if settings.chat_provider == "none":
            print("Chat: intentionally disabled; document search remains available")
        else:
            settings.validate_chat()
            print("Chat configuration: present (credentials and connection not tested)")
    except Exception as exc:
        print("Configuration/SQLite check failed:", type(exc).__name__)
        print("Check .env against .env.example. No secret values are printed.")
        failed = True
    return int(failed)

if __name__ == "__main__":
    raise SystemExit(main())
