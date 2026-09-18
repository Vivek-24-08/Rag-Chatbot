"""Validated settings. Relative paths are always relative to the repository."""
import os
import math
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

def env(name, default="", cast=str):
    return field(default_factory=lambda: cast(os.getenv(name, default)))

def boolean(value):
    if str(value).lower() not in {"true", "false"}:
        raise ValueError("Boolean settings must be true or false.")
    return str(value).lower() == "true"

@dataclass
class Config:
    openai_api_key: str = env("OPENAI_API_KEY")
    openrouter_api_key: str = env("OPENROUTER_API_KEY")
    chat_provider: str = env("CHAT_PROVIDER", "none")
    chat_model: str = env("OPENAI_CHAT_MODEL", "gpt-4o")
    openrouter_model: str = env("OPENROUTER_MODEL")
    databricks_chat_endpoint: str = env("DATABRICKS_CHAT_ENDPOINT")
    embedding_provider: str = env("EMBEDDING_PROVIDER", "local")
    embedding_model: str = env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    local_embedding_model: str = env("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_persist_dir: str = env("CHROMA_PERSIST_DIR", "data/chroma_db")
    chroma_collection_name: str = env("CHROMA_COLLECTION_NAME", "insurance_docs")
    lexical_index_path: str = env("LEXICAL_INDEX_PATH", "data/lexical_search.db")
    search_analytics_db: str = env("SEARCH_ANALYTICS_DB", "data/search_analytics.db")
    pdf_upload_dir: str = env("PDF_UPLOAD_DIR", "data/pdfs")
    chunk_size: int = env("CHUNK_SIZE", "1000", int)
    chunk_overlap: int = env("CHUNK_OVERLAP", "200", int)
    retrieval_top_k: int = env("RETRIEVAL_TOP_K", "5", int)
    retrieval_candidate_k: int = env("RETRIEVAL_CANDIDATE_K", "20", int)
    hybrid_rrf_k: int = env("HYBRID_RRF_K", "60", int)
    hybrid_vector_weight: float = env("HYBRID_VECTOR_WEIGHT", "1", float)
    hybrid_lexical_weight: float = env("HYBRID_LEXICAL_WEIGHT", "1", float)
    feedback_source_boost: float = env("FEEDBACK_SOURCE_BOOST", "0.15", float)
    max_vector_distance: float = env("MAX_VECTOR_DISTANCE", "0.65", float)
    context_tokens: int = env("CONTEXT_TOKENS", "3500", int)
    max_answer_tokens: int = env("MAX_ANSWER_TOKENS", "800", int)
    request_timeout: int = env("REQUEST_TIMEOUT", "60", int)
    search_analytics_enabled: bool = env("SEARCH_ANALYTICS_ENABLED", "false", boolean)
    retention_days: int = env("ANALYTICS_RETENTION_DAYS", "30", int)
    ocr_enabled: bool = env("OCR_ENABLED", "false", boolean)

    def __post_init__(self):
        if self.chat_provider not in {"none", "openai", "openrouter", "databricks"}:
            raise ValueError("CHAT_PROVIDER must be none, openai, openrouter or databricks.")
        if self.embedding_provider not in {"local", "openai"}:
            raise ValueError("EMBEDDING_PROVIDER must be local or openai.")
        for name in ("chunk_size", "retrieval_top_k", "retrieval_candidate_k", "hybrid_rrf_k",
                     "context_tokens", "max_answer_tokens", "request_timeout", "retention_days"):
            if getattr(self, name) <= 0:
                raise ValueError(name + " must be positive.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE.")
        for name in ("hybrid_vector_weight", "hybrid_lexical_weight", "feedback_source_boost", "max_vector_distance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + " must be finite and non-negative.")
        if not self.hybrid_vector_weight + self.hybrid_lexical_weight:
            raise ValueError("At least one hybrid weight must be positive.")
        if self.feedback_source_boost > 0.25:
            raise ValueError("FEEDBACK_SOURCE_BOOST is a fraction, limited to 0.25.")
        for name in ("chroma_persist_dir", "lexical_index_path", "search_analytics_db", "pdf_upload_dir"):
            path = Path(getattr(self, name))
            setattr(self, name, str(path if path.is_absolute() else ROOT / path))

    @property
    def active_chat_model(self):
        return {"openrouter": self.openrouter_model, "databricks": self.databricks_chat_endpoint,
                "openai": self.chat_model, "none": "Not configured"}[self.chat_provider]

    @property
    def active_embedding_model(self):
        return self.local_embedding_model if self.embedding_provider == "local" else self.embedding_model

    @property
    def has_openai_key(self):
        return bool(self.openai_api_key and "REPLACE" not in self.openai_api_key)

    def validate(self):
        if not self.has_openai_key:
            raise ValueError("Set OPENAI_API_KEY in .env for OpenAI embeddings.")

    def validate_chat(self):
        if self.chat_provider == "none":
            raise ValueError("Select a chat provider in .env. Document search is available without one.")
        if self.chat_provider == "openai":
            self.validate()
        if self.chat_provider == "openrouter" and (not self.openrouter_api_key or not self.openrouter_model):
            raise ValueError("Set OPENROUTER_API_KEY and OPENROUTER_MODEL in .env.")
        if self.chat_provider == "databricks" and not self.databricks_chat_endpoint:
            raise ValueError("Set DATABRICKS_CHAT_ENDPOINT and configure Databricks authentication.")

settings = Config()
