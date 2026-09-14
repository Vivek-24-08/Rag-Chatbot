# ==============================================================================
# utils/config.py
# ------------------------------------------------------------------------------
# WHAT THIS FILE DOES:
#   Centralizes all configuration in one place. Every other module in this
#   project imports settings (API keys, model names, folder paths, etc.) from
#   here instead of reading environment variables directly.
#
# WHY THIS MATTERS (GenAI beginners):
#   RAG apps have a LOT of "knobs" you'll want to tune while experimenting:
#     - which LLM to call
#     - which embedding model to use
#     - how big each text chunk should be
#     - how many chunks to retrieve per question
#   Putting them all in one Config object means you can tune the whole
#   pipeline's behavior by editing ONE file (or even just the .env file)
#   instead of hunting through every .py file for a hard-coded number.
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
    retrieval_candidate_k: int = int(os.getenv("RETRIEVAL_CANDIDATE_K", 20))
    hybrid_rrf_k: int = int(os.getenv("HYBRID_RRF_K", 60))
    hybrid_vector_weight: float = float(os.getenv("HYBRID_VECTOR_WEIGHT", 1.0))
    hybrid_lexical_weight: float = float(os.getenv("HYBRID_LEXICAL_WEIGHT", 1.0))

    # Search-quality data remains local and can be disabled through .env.
    search_analytics_db: str = os.getenv("SEARCH_ANALYTICS_DB", "data/search_analytics.db")
    search_analytics_enabled: bool = os.getenv("SEARCH_ANALYTICS_ENABLED", "true").lower() == "true"
    feedback_source_boost: float = float(os.getenv("FEEDBACK_SOURCE_BOOST", 0.15))
    lexical_index_path: str = os.getenv("LEXICAL_INDEX_PATH", "data/lexical_search.db")

    # --- File storage paths -------------------------------------------------
    pdf_upload_dir: str = os.getenv("PDF_UPLOAD_DIR", "data/pdfs")

    # --- Embedding provider ---------------------------------------------------
    # "openai" uses OpenAI text-embedding-3-large (requires API key + credits)
    # "local"  uses HuggingFace all-MiniLM-L6-v2 (free, runs on CPU, no key)
    # Default: auto-detect based on whether OPENAI_API_KEY is set.
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "")

    # Local embedding model (only used when embedding_provider == "local")
    local_embedding_model: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    # --- Chat (LLM) provider ---------------------------------------------------
    # "openai"     uses OpenAI GPT-4o (requires API key + credits)
    # "databricks" uses Databricks Foundation Model APIs (free within workspace)
    # Default: auto-detect based on whether OPENAI_API_KEY is set.
    chat_provider: str = os.getenv("CHAT_PROVIDER", "")

    # Databricks serving endpoint (only used when chat_provider == "databricks")
    databricks_chat_endpoint: str = os.getenv(
        "DATABRICKS_CHAT_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct"
    )

    def __post_init__(self):
        """Default to local embeddings + Databricks LLM for zero-cost testing."""
        if not self.embedding_provider:
            self.embedding_provider = "local"
        if not self.chat_provider:
            self.chat_provider = "databricks"

    @property
    def has_openai_key(self) -> bool:
        """Check whether a real OpenAI API key is configured."""
        return bool(
            self.openai_api_key
            and not self.openai_api_key.startswith("sk-REPLACE")
        )

    def validate(self) -> None:
        """
        Raise a clear, human-readable error early if required settings are
        missing, instead of letting the app crash later with a cryptic
        "401 Unauthorized" deep inside the OpenAI SDK.
        """
        if not self.has_openai_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add "
                "your real OpenAI API key before running the app."
            )


# A single, shared Config instance that every other module can import:
#     from utils.config import settings
settings = Config()
