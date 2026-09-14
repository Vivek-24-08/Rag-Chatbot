# ==============================================================================
# app.py
# ------------------------------------------------------------------------------
# PROJECT ENTRY POINT.
#
# HOW TO RUN THIS APP:
#       streamlit run app.py
#
# WHAT THIS FILE DOES:
#   This file is intentionally tiny. Its only job is to import and launch
#   the actual UI code that lives in frontend/streamlit_ui.py. Keeping
#   app.py minimal (rather than putting all the UI logic directly here)
#   makes the project easier to navigate: anyone opening the repo can see
#   at a glance that the "front door" is app.py, while the real
#   implementation is organized by responsibility into ingestion/,
#   chunking/, embeddings/, vectorstore/, retrieval/, rag/, prompts/, and
#   frontend/.
#
# WHY A SEPARATE frontend/streamlit_ui.py INSTEAD OF PUTTING EVERYTHING HERE:
#   Streamlit runs whatever .py file you pass to `streamlit run` as a
#   script from top to bottom. Separating the UI into its own module means
#   you could, in principle, later add a second frontend (e.g., a FastAPI
#   backend for a different UI) that reuses the same rag/, retrieval/, etc.
#   modules without duplicating any pipeline logic.
# ==============================================================================

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# load_dotenv() reads the ".env" file in the project root and copies its
# key=value pairs into the process's environment variables (os.environ).
# This keeps secrets (like API keys) OUT of the source code.
load_dotenv()


@dataclass
class Config:
    """
    A simple, typed container for all app settings.

    Using a dataclass (instead of scattering os.getenv() calls everywhere)
    gives us:
      - autocomplete / type hints in your editor
      - one obvious place to see every setting the app depends on
      - easy defaults if a .env value is missing
    """

    # --- OpenAI credentials & model choices ---------------------------------
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    # --- Vector store (ChromaDB) settings ------------------------------------
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "insurance_docs")

    # --- Chunking settings ----------------------------------------------------
    # CHUNK_SIZE: how many characters go into each chunk of text.
    # CHUNK_OVERLAP: how many characters two consecutive chunks share, so a
    #                sentence that spans a chunk boundary isn't lost entirely.
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 200))

    # --- Retrieval settings -----------------------------------------------------
    # RETRIEVAL_TOP_K: how many of the most-similar chunks we hand to the LLM
    # as context for each question.
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", 5))

    # --- File storage paths -------------------------------------------------
    pdf_upload_dir: str = os.getenv("PDF_UPLOAD_DIR", "data/pdfs")

    def validate(self) -> None:
        """
        Raise a clear, human-readable error early if required settings are
        missing, instead of letting the app crash later with a cryptic
        "401 Unauthorized" deep inside the OpenAI SDK.
        """
        if not self.openai_api_key or self.openai_api_key.startswith("sk-REPLACE"):
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add "
                "your real OpenAI API key before running the app."
            )


# A single, shared Config instance that every other module can import:
#     from utils.config import settings
settings = Config()

# Launch the Streamlit UI only when running inside the Databricks App
# container (where Streamlit and OPENAI_API_KEY are available).
# When run directly in the workspace editor, this is skipped.
try:
    import streamlit.runtime
    if streamlit.runtime.exists():
        import frontend.streamlit_ui
        frontend.streamlit_ui.run_app()
except Exception:
    pass  # Not in a Streamlit context
