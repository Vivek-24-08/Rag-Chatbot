# ==============================================================================
# vectorstore/chroma_manager.py
# ------------------------------------------------------------------------------
# STEP 5 of the RAG pipeline: "Store embeddings in vector database"
#
# WHAT IS A VECTOR DATABASE? (GenAI beginners)
#   A normal database is great at exact lookups ("find the row where
#   id = 42"). A VECTOR database is built for a different kind of lookup:
#   "find the vectors that are mathematically closest to THIS vector."
#   That "closeness" search is called a SIMILARITY SEARCH, and it's the
#   engine behind RAG retrieval — see retrieval/retriever.py for Step 6.
#
# WHY ChromaDB SPECIFICALLY:
#   ChromaDB is open-source, runs entirely locally (no external service or
#   account needed), and persists its data to a folder on disk
#   (data/chroma_db/ in this project), which is perfect for a local-machine
#   VS Code prototype like this one. It also integrates directly with
#   LangChain, so we don't have to hand-write vector math ourselves.
#
# WHAT GETS STORED PER CHUNK:
#   For every text chunk we store THREE things together:
#     1. the embedding vector (for similarity search)
#     2. the original chunk text (so we can show/send it to the LLM later)
#     3. metadata: source filename + page number (so we can cite it later)
#
# HOW THIS FITS INTO THE BIGGER PICTURE:
#   chunks + vectors --(this file)--> persisted to disk in data/chroma_db/
#   Later, at question-answering time, retrieval/retriever.py re-opens this
#   same persisted database and searches it.
# ==============================================================================

import os
import re
import sqlite3
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from chunking.text_splitter import Chunk
from embeddings.embedding_service import EmbeddingService
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ChromaManager:
    """
    Manages a single persistent ChromaDB collection: creating it, adding
    document chunks to it, and exposing it for similarity search.
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        # Reuse a shared EmbeddingService if one is passed in (e.g., by the
        # RAG pipeline that also needs it for embedding the user's
        # question) — otherwise create a fresh one.
        self.embedding_service = embedding_service or EmbeddingService()

        # Make sure the folder Chroma will write its on-disk database files
        # into actually exists before we try to use it.
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)

        # LangChain's Chroma wrapper handles:
        #   - opening (or creating) the on-disk database at persist_directory
        #   - calling our embedding model whenever .add_documents() or
        #     .similarity_search() is used
        self.vectorstore = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=self.embedding_service.get_langchain_embeddings(),
            persist_directory=settings.chroma_persist_dir,
        )
        self._initialize_lexical_index()

        logger.info(
            f"ChromaManager ready. Collection='{settings.chroma_collection_name}', "
            f"persist_dir='{settings.chroma_persist_dir}'"
        )

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        Embed and store a batch of text chunks in ChromaDB.

        Args:
            chunks: list of Chunk objects from chunking/text_splitter.py

        Returns:
            The number of chunks successfully added.
        """
        if not chunks:
            logger.warning("add_chunks() called with an empty chunk list — nothing to do")
            return 0

        # LangChain's Chroma store expects LangChain "Document" objects:
        # page_content (the text) + metadata (a dict). Internally, when we
        # call add_documents(), Chroma will:
        #   1. call self.embedding_service to turn each chunk's text into a
        #      vector (this is Step 4 happening again, but batched)
        #   2. write {vector, text, metadata, id} into the on-disk database
        documents = [
            Document(page_content=chunk.text, metadata=chunk.metadata)
            for chunk in chunks
        ]
        ids = [chunk.chunk_id for chunk in chunks]

        logger.info(f"Embedding and storing {len(documents)} chunks in ChromaDB...")
        self.vectorstore.add_documents(documents=documents, ids=ids)
        self._upsert_lexical_chunks(chunks)
        logger.info("Chunks stored successfully (persisted to disk automatically)")

        return len(documents)

    def get_retriever(self, top_k: int = None):
        """
        Return a LangChain "retriever" object configured for similarity
        search. A retriever is just a standardized interface LangChain uses
        so that different vector stores (Chroma, Pinecone, FAISS, etc.) can
        all be swapped in and out of a RAG chain without changing other code.

        Args:
            top_k: how many of the most similar chunks to return per query.
                   Defaults to settings.retrieval_top_k from utils/config.py.
        """
        k = top_k or settings.retrieval_top_k
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def similarity_search_with_scores(self, query: str, top_k: int = None):
        """
        Directly run a similarity search and return chunks WITH their
        similarity scores. Useful when we want to show the user "how
        confident" the retrieval was, or filter out weak matches.

        Args:
            query: the (already-plain-text) user question
            top_k: number of chunks to retrieve

        Returns:
            List of (Document, score) tuples. For Chroma, a LOWER score
            means MORE similar (it's a distance metric, not a percentage).
        """
        k = top_k or settings.retrieval_top_k
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def get_all_documents(self) -> List[Document]:
        """Return persisted chunks for local BM25 keyword retrieval."""
        stored = self.vectorstore._collection.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(stored.get("documents", []), stored.get("metadatas", []))
            if text
        ]

    def _lexical_connection(self):
        os.makedirs(os.path.dirname(settings.lexical_index_path) or ".", exist_ok=True)
        return sqlite3.connect(settings.lexical_index_path)

    def _initialize_lexical_index(self) -> None:
        """Create a persistent FTS5 index and rebuild it only when out of sync."""
        with self._lexical_connection() as connection:
            connection.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search
                              USING fts5(chunk_id UNINDEXED, content, source UNINDEXED, page UNINDEXED)""")
            indexed_count = connection.execute("SELECT COUNT(*) FROM chunk_search").fetchone()[0]
            vector_count = self.vectorstore._collection.count()
            if indexed_count == vector_count:
                return
            connection.execute("DELETE FROM chunk_search")
            stored = self.vectorstore._collection.get(include=["documents", "metadatas"])
            connection.executemany(
                "INSERT INTO chunk_search VALUES (?, ?, ?, ?)",
                [(chunk_id, text, (metadata or {}).get("source", "unknown"), str((metadata or {}).get("page", 0)))
                 for chunk_id, text, metadata in zip(stored.get("ids", []), stored.get("documents", []), stored.get("metadatas", []))
                 if text],
            )

    def _upsert_lexical_chunks(self, chunks: List[Chunk]) -> None:
        with self._lexical_connection() as connection:
            for chunk in chunks:
                connection.execute("DELETE FROM chunk_search WHERE chunk_id = ?", (chunk.chunk_id,))
                connection.execute("INSERT INTO chunk_search VALUES (?, ?, ?, ?)",
                                   (chunk.chunk_id, chunk.text, chunk.metadata.get("source", "unknown"), str(chunk.metadata.get("page", 0))))

    def lexical_search(self, query: str, top_k: int) -> List[tuple]:
        """Use the persisted FTS5/BM25 index; this does not scan every chunk."""
        terms = re.findall(r"[a-z0-9]+", query.lower())
        if not terms:
            return []
        match_query = " OR ".join(terms)
        with self._lexical_connection() as connection:
            rows = connection.execute("""SELECT content, source, page, -bm25(chunk_search) AS score
                                         FROM chunk_search WHERE chunk_search MATCH ?
                                         ORDER BY bm25(chunk_search) LIMIT ?""", (match_query, top_k)).fetchall()
        return [(Document(page_content=text, metadata={"source": source, "page": int(page)}), float(score))
                for text, source, page, score in rows]

    def document_count(self) -> int:
        """Return how many chunks are currently stored in the collection."""
        return self.vectorstore._collection.count()

    def clear_collection(self) -> None:
        """
        Delete ALL vectors in the current collection. Useful when a user
        wants to re-upload documents from scratch rather than accumulating
        duplicates across sessions.
        """
        logger.warning(f"Clearing all vectors from collection '{settings.chroma_collection_name}'")
        existing_ids = self.vectorstore._collection.get()["ids"]
        if existing_ids:
            self.vectorstore._collection.delete(ids=existing_ids)
        with self._lexical_connection() as connection:
            connection.execute("DELETE FROM chunk_search")
        logger.info("Collection cleared")
