"""Page-aware chunks with explicit document identity and token-safe splitting."""
from dataclasses import dataclass, field
import hashlib
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config import settings

@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

class DocumentChunker:
    def __init__(self, chunk_size=None, chunk_overlap=None, token_counter=None, token_limit=None):
        self.chunk_size = settings.chunk_size if chunk_size is None else chunk_size
        self.chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Require 0 <= overlap < positive chunk size.")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""], length_function=len)
        self.token_counter, self.token_limit = token_counter, token_limit

    def _fit(self, text):
        if not self.token_counter or self.token_counter(text) <= self.token_limit:
            return [text]
        if len(text) < 2:
            raise ValueError("Embedding token limit is too small.")
        mid = len(text) // 2
        return self._fit(text[:mid]) + self._fit(text[mid:])

    def chunk_pages(self, pages):
        chunks = []
        for page in pages:
            # Loader supplies a content hash; test/custom callers get a content-based fallback.
            identity = page.metadata.get("document_id") or hashlib.sha256(
                json.dumps([page.source_file, page.text], ensure_ascii=False).encode()).hexdigest()
            pieces = [part for piece in self.splitter.split_text(page.text)
                      for part in self._fit(piece) if part.strip()]
            for i, text in enumerate(pieces):
                metadata = {**page.metadata, "source": page.source_file, "page": page.page_number,
                            "document_id": identity, "chunk_index_on_page": i}
                cid = hashlib.sha256(json.dumps(
                    [identity, page.page_number, i, text], ensure_ascii=False).encode()).hexdigest()
                metadata["chunk_id"] = cid
                chunks.append(Chunk(text, metadata, cid))
        return chunks
