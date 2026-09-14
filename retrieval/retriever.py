"""Hybrid semantic and keyword retrieval for insurance documents."""
from dataclasses import dataclass
import re
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from utils.config import settings
from utils.logger import get_logger
from vectorstore.chroma_manager import ChromaManager

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    page_number: int
    similarity_score: float  # Fused RRF score; higher is better.
    vector_distance: float = None
    lexical_score: float = 0.0


class Retriever:
    def __init__(self, chroma_manager: ChromaManager):
        self.chroma_manager = chroma_manager

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _key(document: Document) -> Tuple[str, str, int]:
        return document.page_content, str(document.metadata.get("source", "unknown")), int(document.metadata.get("page", 0))

    def _bm25_search(self, question: str, limit: int):
        return self.chroma_manager.lexical_search(question, limit)

    def retrieve(self, question: str, top_k: int = None) -> List[RetrievedChunk]:
        k = top_k or settings.retrieval_top_k
        candidate_k = max(k, settings.retrieval_candidate_k)
        vector_results = self.chroma_manager.similarity_search_with_scores(question, candidate_k)
        lexical_results = self._bm25_search(question, candidate_k)
        fused: Dict[Tuple[str, str, int], Dict] = {}

        for rank, (document, distance) in enumerate(vector_results, 1):
            result = fused.setdefault(self._key(document), {"document": document, "score": 0.0, "distance": None, "lexical": 0.0})
            result["score"] += settings.hybrid_vector_weight / (settings.hybrid_rrf_k + rank)
            result["distance"] = float(distance)
        for rank, (document, lexical_score) in enumerate(lexical_results, 1):
            result = fused.setdefault(self._key(document), {"document": document, "score": 0.0, "distance": None, "lexical": 0.0})
            result["score"] += settings.hybrid_lexical_weight / (settings.hybrid_rrf_k + rank)
            result["lexical"] = float(lexical_score)

        ranked = sorted(fused.values(), key=lambda result: result["score"], reverse=True)[:k]
        return [RetrievedChunk(
            text=result["document"].page_content,
            source_file=result["document"].metadata.get("source", "unknown"),
            page_number=result["document"].metadata.get("page", 0),
            similarity_score=result["score"], vector_distance=result["distance"], lexical_score=result["lexical"],
        ) for result in ranked]

    def has_documents(self) -> bool:
        return self.chroma_manager.document_count() > 0
