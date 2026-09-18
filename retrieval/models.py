"""Dependency-light types shared by retrieval, prompts and the UI."""
from dataclasses import dataclass

@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    page_number: int
    similarity_score: float
    vector_distance: float = None
    lexical_score: float = 0.0
    chunk_id: str = ""
    document_id: str = ""
    vector_rank: int = None
    lexical_rank: int = None
    base_score: float = 0.0
    page_label: str = ""
