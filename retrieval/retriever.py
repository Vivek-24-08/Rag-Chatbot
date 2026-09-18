"""Hybrid retrieval: independent candidates, deduplication, bounded adaptation."""
import hashlib
import math
import re
from retrieval.models import RetrievedChunk
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)
STOP = set("a an the is are do does what how about my your for of to and or it this that in".split())

def terms(text):
    return set(re.findall(r"[^\W_]+", text.lower())) - STOP

class Retriever:
    def __init__(self, chroma_manager, config=None):
        self.chroma_manager = chroma_manager
        self.config = config or settings
        self.last_warnings = []

    def retrieve(self, question, top_k=None, document_ids=None, weights=None, preferences=None):
        if not question or not question.strip():
            raise ValueError("Enter a question.")
        k = self.config.retrieval_top_k if top_k is None else top_k
        if type(k) is not int or k <= 0:
            raise ValueError("top_k must be a positive integer.")
        weights = weights or (self.config.hybrid_vector_weight, self.config.hybrid_lexical_weight)
        if len(weights) != 2 or any(not math.isfinite(w) or w < 0 for w in weights) or not sum(weights):
            raise ValueError("Require two finite non-negative weights with a positive sum.")
        count = max(k, self.config.retrieval_candidate_k)
        branches = []
        self.last_warnings = []
        for index, method in enumerate((self.chroma_manager.similarity_search_with_scores, self.chroma_manager.lexical_search)):
            if not weights[index]:
                branches.append([])
                continue
            try:
                branches.append(method(question, count, document_ids=document_ids))
            except Exception as exc:
                branches.append([])
                self.last_warnings.append(method.__name__ + " unavailable")
                logger.warning("retrieval_branch_failed error=%s", type(exc).__name__)
        if len(self.last_warnings) == sum(weight > 0 for weight in weights):
            raise RuntimeError("All enabled search indexes are unavailable.")
        fused = {}
        query_terms = terms(question)
        for branch, results in enumerate(branches):
            if weights[branch] == 0:
                continue
            seen = set()
            for rank, (doc, raw_score) in enumerate(results, 1):
                if not math.isfinite(raw_score):
                    continue
                meta = doc.metadata
                cid = meta.get("chunk_id") or hashlib.sha256(
                    (str(meta) + doc.page_content).encode()).hexdigest()
                if cid in seen:
                    continue
                seen.add(cid)
                # Evidence gate: lexical results must share informative words;
                # semantic-only evidence must meet the configured cosine distance.
                if branch == 1 and not query_terms.intersection(terms(doc.page_content)):
                    continue
                if branch == 0 and raw_score > self.config.max_vector_distance:
                    continue
                item = fused.setdefault(cid, RetrievedChunk(
                    doc.page_content, meta.get("source","unknown"), meta.get("page",0), 0.0,
                    chunk_id=cid, document_id=meta.get("document_id",""),
                    page_label=meta.get("page_label","")))
                item.base_score += weights[branch] / (self.config.hybrid_rrf_k + rank)
                if branch == 0:
                    item.vector_distance, item.vector_rank = raw_score, rank
                else:
                    item.lexical_score, item.lexical_rank = raw_score, rank
        for item in fused.values():
            # At most +/- 25%; applied to all candidates, before top-K truncation.
            preference = max(-1, min(1, (preferences or {}).get(item.document_id, 0)))
            item.similarity_score = item.base_score * (1 + self.config.feedback_source_boost * preference)
        ranked = sorted(fused.values(), key=lambda x: (-x.similarity_score, x.chunk_id))
        selected = []
        for item in ranked:
            if any(item.document_id == other.document_id and
                   len(terms(item.text) & terms(other.text)) / max(1, len(terms(item.text) | terms(other.text))) > 0.9
                   for other in selected):
                continue
            selected.append(item)
            if len(selected) == k:
                break
        return selected

    def has_documents(self):
        return self.chroma_manager.document_count() > 0
