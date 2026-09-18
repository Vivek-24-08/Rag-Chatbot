"""SQLite is authoritative; Chroma is a repairable, collection-scoped derived index.

Document replacement + FTS updates commit together. Vector sync holds a SQLite
write lock, so searches cannot observe an ingestion half-way through the update.
Old stores are left untouched; v2 uses its own namespace and requires re-ingestion.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import uuid
from langchain_core.documents import Document
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class ChromaManager:
    def __init__(self, embedding_service=None, config=None):
        self.config = config or settings
        if embedding_service is None:
            from embeddings.embedding_service import EmbeddingService
            embedding_service = EmbeddingService(self.config)
        self.embedding_service = embedding_service
        scope = hashlib.sha256((self.config.chroma_persist_dir + "/" +
                                self.config.chroma_collection_name).encode()).hexdigest()[:16]
        base = Path(self.config.lexical_index_path)
        self.db_path = base.with_name(base.stem + "-v2-" + scope + ".db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.collection_name = "rag-v2-" + scope
        self._collection = None
        self._client = None
        self.last_vector_error = ""
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS documents(
                id TEXT PRIMARY KEY, source TEXT, plan TEXT, year TEXT, version TEXT,
                kind TEXT, payload BLOB, updated REAL DEFAULT (strftime('%s','now')))""")
            db.execute("""CREATE TABLE IF NOT EXISTS chunks(
                id TEXT PRIMARY KEY, doc_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                text TEXT NOT NULL, metadata TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id)")
            try:
                db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(chunk_id UNINDEXED, content)")
            except sqlite3.OperationalError:
                raise ValueError("SQLite FTS5 is unavailable. Use the documented Python 3.11 environment.")
            version = db.execute("SELECT value FROM state WHERE key='schema'").fetchone()
            if version and version[0] != "2":
                raise ValueError("Unsupported document database schema; use a matching application version.")
            db.execute("INSERT OR IGNORE INTO state VALUES ('schema','2')")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(str(self.db_path), timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @property
    def collection(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings
            self._client = chromadb.PersistentClient(path=str(Path(self.config.chroma_persist_dir) / "v2"),
                                               settings=Settings(anonymized_telemetry=False))
        return self._client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"})

    def _sync(self, db):
        saved = db.execute("SELECT value FROM state WHERE key='revision'").fetchone()
        revision = saved[0] if saved else "empty"
        signature = revision + ":" + self.embedding_service.fingerprint
        collection = self.collection
        meta = collection.metadata or {}
        count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if meta.get("revision") == signature and collection.count() == count:
            return
        # Recreate the derived index when embedding dimensions/model change.
        if meta.get("embedding_model") != self.embedding_service.fingerprint:
            self._client.delete_collection(self.collection_name)
            collection = self.collection
        collection.modify(metadata={"revision": "dirty", "embedding_model": self.embedding_service.fingerprint})
        current_ids = collection.get(include=[])["ids"]
        for offset in range(0, len(current_ids), 256):
            collection.delete(ids=current_ids[offset:offset + 256])
        cursor = db.execute("SELECT id,text,metadata FROM chunks ORDER BY id")
        while True:
            batch = cursor.fetchmany(64)
            if not batch:
                break
            vectors = self.embedding_service.embed_documents([row[1] for row in batch])
            collection.upsert(ids=[r[0] for r in batch], documents=[r[1] for r in batch],
                              metadatas=[json.loads(r[2]) for r in batch], embeddings=vectors)
        collection.modify(metadata={"revision": signature, "embedding_model": self.embedding_service.fingerprint})
        db.execute("INSERT OR REPLACE INTO state VALUES ('vector_revision',?)", (signature,))

    def sync_vectors(self):
        try:
            with self._db() as db:
                db.execute("BEGIN IMMEDIATE")
                self._sync(db)
            self.last_vector_error = ""
            return True
        except Exception as exc:
            self.last_vector_error = type(exc).__name__
            logger.warning("vector_sync_failed error=%s", self.last_vector_error)
            return False

    def replace_document(self, record, chunks, payload=b""):
        if not chunks:
            raise ValueError("Document contains no indexable chunks.")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            old = [r[0] for r in db.execute("SELECT id FROM chunks WHERE doc_id=?", (record["id"],))]
            db.executemany("DELETE FROM search WHERE chunk_id=?", [(cid,) for cid in old])
            db.execute("DELETE FROM documents WHERE id=?", (record["id"],))
            db.execute("""INSERT INTO documents(id,source,plan,year,version,kind,payload)
                          VALUES (?,?,?,?,?,?,?)""",
                       tuple(record[k] for k in ("id", "source", "plan", "year", "version", "kind")) + (payload,))
            for chunk in chunks:
                meta = {**chunk.metadata, "document_id": record["id"], "chunk_id": chunk.chunk_id,
                        "plan": record["plan"], "year": record["year"], "source": record["source"]}
                db.execute("INSERT INTO chunks VALUES (?,?,?,?)",
                           (chunk.chunk_id, record["id"], chunk.text, json.dumps(meta)))
                db.execute("INSERT INTO search VALUES (?,?)", (chunk.chunk_id, chunk.text))
            db.execute("INSERT OR REPLACE INTO state VALUES ('revision',?)", (uuid.uuid4().hex,))
        # Canonical commit survives vector failures; search can fall back to FTS.
        self.sync_vectors()
        return len(chunks)

    def add_chunks(self, chunks):
        """Compatibility API for programmatic ingestion; grouped by document ID."""
        groups = {}
        for chunk in chunks:
            groups.setdefault(chunk.metadata["document_id"], []).append(chunk)
        for identity, batch in groups.items():
            meta = batch[0].metadata
            self.replace_document({"id": identity, "source": meta.get("source", "unknown"),
                                   "plan": meta.get("plan", ""), "year": meta.get("year", ""),
                                   "version": meta.get("version", identity), "kind": "text"}, batch)
        return len(chunks)

    def list_documents(self):
        with self._db() as db:
            rows = db.execute("""SELECT d.id,d.source,d.plan,d.year,d.version,d.kind,count(c.id)
                                 FROM documents d LEFT JOIN chunks c ON d.id=c.doc_id GROUP BY d.id
                                 ORDER BY d.source""").fetchall()
        return [dict(zip(("id","source","plan","year","version","kind","chunks"), row)) for row in rows]

    def document_payload(self, document_id):
        with self._db() as db:
            row = db.execute("SELECT payload FROM documents WHERE id=?", (document_id,)).fetchone()
        return row[0] if row else None

    def remove_document(self, document_id):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            ids = db.execute("SELECT id FROM chunks WHERE doc_id=?", (document_id,)).fetchall()
            db.executemany("DELETE FROM search WHERE chunk_id=?", ids)
            db.execute("DELETE FROM documents WHERE id=?", (document_id,))
            db.execute("INSERT OR REPLACE INTO state VALUES ('revision',?)", (uuid.uuid4().hex,))
        self.sync_vectors()

    def clear_collection(self):
        for record in self.list_documents():
            self.remove_document(record["id"])

    def document_count(self):
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def get_all_documents(self):
        with self._db() as db:
            rows = db.execute("SELECT text,metadata FROM chunks").fetchall()
        return [Document(page_content=t, metadata=json.loads(m)) for t,m in rows]

    def similarity_search_with_scores(self, query, top_k=None, document_ids=None):
        k = top_k or self.config.retrieval_top_k
        if document_ids == [] or not self.document_count():
            return []
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            self._sync(db)
            where = {"document_id": {"$in": document_ids}} if document_ids else None
            result = self.collection.query(query_embeddings=[self.embedding_service.embed_query(query)],
                                           n_results=min(k, self.document_count()),
                                           where=where, include=["documents","metadatas","distances"])
        return [(Document(page_content=t, metadata=m), float(d)) for t,m,d in
                zip(result["documents"][0], result["metadatas"][0], result["distances"][0])]

    def lexical_search(self, query, top_k, document_ids=None):
        # Quoted Unicode tokens cannot become FTS operators. Punctuation is consistently split.
        terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query.lower(), re.UNICODE)))[:80]
        if not terms or document_ids == []:
            return []
        match = " OR ".join('"' + term + '"' for term in terms)
        sql = """SELECT c.text,c.metadata,-bm25(search) FROM search
                 JOIN chunks c ON c.id=search.chunk_id WHERE search MATCH ?"""
        args = [match]
        if document_ids:
            sql += " AND c.doc_id IN (" + ",".join("?" for _ in document_ids) + ")"
            args.extend(document_ids)
        sql += " ORDER BY bm25(search),c.id LIMIT ?"
        args.append(top_k)
        with self._db() as db:
            rows = db.execute(sql, args).fetchall()
        return [(Document(page_content=t, metadata=json.loads(m)), float(score)) for t,m,score in rows]

    def get_retriever(self, top_k=None):
        from retrieval.retriever import Retriever
        manager = self
        class Adapter:
            def invoke(self, query):
                return Retriever(manager, config=manager.config).retrieve(query, top_k=top_k)
        return Adapter()
