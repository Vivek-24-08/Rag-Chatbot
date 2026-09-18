"""Offline regressions: synthetic documents, no live model or API credentials."""
import json
from types import SimpleNamespace
import pytest
from langchain_core.documents import Document
from utils.config import Config
from chunking.text_splitter import Chunk, DocumentChunker
from ingestion.pdf_loader import PageContent
from ingestion.source_loader import load_source
from retrieval.retriever import Retriever
from retrieval.models import RetrievedChunk
from intelligence.search_intelligence import SearchIntelligence
from prompts.prompt_template import validate_answer
from vectorstore.chroma_manager import ChromaManager

class FakeEmbedding:
    fingerprint = "test-v1"
    token_limit = 256
    def token_count(self, text):
        return len(text.split())
    def embed_documents(self, texts):
        return [[float("deductible" in t.lower()), float("appeal" in t.lower()), 0.1] for t in texts]
    def embed_query(self, text):
        return self.embed_documents([text])[0]

@pytest.fixture
def config(tmp_path):
    return Config(chroma_persist_dir=str(tmp_path / "vectors"),
                  lexical_index_path=str(tmp_path / "lexical.db"),
                  search_analytics_db=str(tmp_path / "analytics.db"),
                  chat_provider="none", search_analytics_enabled=False)

def item(identity="doc1", text="The deductible is $100.", cid="chunk1", page=1):
    return Chunk(text, {"document_id": identity, "source": identity + ".txt", "page": page}, cid)

def record(identity="doc1", plan="A", year="2026"):
    return dict(id=identity, source=identity + ".txt", plan=plan, year=year, version="v1", kind=".txt")

@pytest.fixture
def manager(config, monkeypatch):
    manager = ChromaManager(FakeEmbedding(), config)
    monkeypatch.setattr(manager, "sync_vectors", lambda: False)
    return manager

def test_config_validation_and_zero_overlap():
    assert Config(chunk_overlap=0).chunk_overlap == 0
    for kwargs in ({"chunk_size":0}, {"chunk_overlap":1000}, {"hybrid_vector_weight":-1},
                   {"feedback_source_boost":1}, {"max_vector_distance":float("nan")}):
        with pytest.raises(ValueError):
            Config(**kwargs)

def test_chunk_ids_differ_for_versions_and_token_limit():
    chunker = DocumentChunker(100, 0, lambda x: len(x), 10)
    a = chunker.chunk_pages([PageContent("hello world " * 5, 1, "same.txt", {"document_id":"a"})])
    b = chunker.chunk_pages([PageContent("hello world " * 5, 1, "same.txt", {"document_id":"b"})])
    assert all(len(c.text) <= 10 for c in a)
    assert {c.chunk_id for c in a}.isdisjoint(c.chunk_id for c in b)

def test_source_formats(tmp_path):
    for suffix, content in ((".txt","Coverage"), (".md","# Coverage"),
                            (".json",'[{"question":"Copay?","answer":"$10"}]')):
        path = tmp_path / ("doc" + suffix)
        path.write_text(content, encoding="utf-8")
        assert load_source(path)[0].text
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_source(path)

def test_atomic_replacement_and_scope(manager, config):
    manager.replace_document(record(), [item()], b"original")
    manager.replace_document(record(), [item(text="Appeal deadline is 30 days.", cid="changed")], b"replacement")
    assert manager.document_count() == 1
    assert manager.lexical_search("deductible", 10) == []
    assert len(manager.lexical_search("appeal", 10)) == 1
    assert manager.document_payload("doc1") == b"replacement"
    config.chroma_collection_name = "another"
    assert ChromaManager(FakeEmbedding(), config).document_count() == 0

def test_failed_transaction_preserves_old_document(manager):
    manager.replace_document(record(), [item()])
    with pytest.raises(Exception):
        manager.replace_document(record(), [item(cid="same"), item(cid="same")])
    assert manager.lexical_search("deductible", 2)
    assert manager.document_count() == 1

def test_fts_filter_unicode_and_delete(manager):
    manager.replace_document(record(), [item(text="café deductible")])
    manager.replace_document(record("doc2"), [item("doc2", "other deductible", "c2")])
    assert len(manager.lexical_search('café OR " ) -', 5, ["doc1"])) == 1
    assert len(manager.lexical_search("deductible", 5, ["doc2"])) == 1
    assert manager.lexical_search("deductible", 5, []) == []
    manager.remove_document("doc1")
    assert manager.document_payload("doc1") is None
    assert manager.document_count() == 1

def test_routing_followup_feedback_privacy(tmp_path):
    intelligence = SearchIntelligence(str(tmp_path / "events.db"), True)
    query = intelligence.process("What about out of network?", [
        {"role":"user","content":"What is the specialist copay?"},
        {"role":"user","content":"And referrals?"}])
    assert "specialist" in query.search_text and "referrals" in query.search_text
    assert query.intent == "providers"
    assert intelligence.process("What is the copay?").weights != (1,1)
    assert intelligence.process("And that?").needs_clarification
    assert intelligence.process("billingham").intent == "general"
    event = intelligence.record_search("session", query, 1, ["doc1"])
    assert intelligence.record_feedback(event, 1, "wrong") is False
    assert intelligence.record_feedback(event, 1, "session")
    assert intelligence.preferred_sources("session") == {"doc1":1}
    intelligence.record_feedback(event, -1, "session")
    assert intelligence.preferred_sources("session") == {"doc1":-1}
    assert intelligence.summary()["positive_rate"] == 0
    with intelligence._db() as db:
        assert len(db.execute("SELECT * FROM ratings").fetchall()) == 1
        assert "specialist" not in str(db.execute("SELECT * FROM events").fetchall())
    intelligence.forget_session("session")
    assert intelligence.summary()["searches"] == 0

def test_analytics_failure_does_not_raise(tmp_path):
    analytics = SearchIntelligence(str(tmp_path / "a.db"), True)
    analytics.database_path = str(tmp_path / "missing" / "db")
    assert analytics.record_search("s", analytics.process("copay"), 1, []) == ""
    assert analytics.preferred_sources("s") == {}

def doc(cid, identity, text="deductible evidence"):
    return Document(page_content=text, metadata={"chunk_id":cid,"document_id":identity,"source":identity,"page":1})

def test_rrf_dedup_and_feedback_before_topk(config):
    a, b = doc("a","a"), doc("b","b","deductible amount")
    fake = SimpleNamespace(similarity_search_with_scores=lambda *a,**k: [(b,.2)],
                           lexical_search=lambda *args,**k: [(a,1),(a,1),(b,.5)])
    retriever = Retriever(fake, config)
    results = retriever.retrieve("deductible", 1, preferences={"a":-1,"b":1})
    assert results[0].chunk_id == "b"
    assert results[0].similarity_score <= results[0].base_score*1.25
    with pytest.raises(ValueError):
        retriever.retrieve("deductible", 0)

def test_branch_failure_and_evidence_gate(config):
    def fail(*a,**k):
        raise RuntimeError("test")
    fake = SimpleNamespace(similarity_search_with_scores=fail,
                           lexical_search=lambda *a,**k: [(doc("a","a"),1)])
    retriever = Retriever(fake, config)
    assert retriever.retrieve("deductible")
    assert retriever.last_warnings
    fake.lexical_search = fail
    with pytest.raises(RuntimeError):
        retriever.retrieve("deductible")
    fake.similarity_search_with_scores = lambda *a,**k: [(doc("a","a"),.99)]
    fake.lexical_search = lambda *a,**k: []
    assert retriever.retrieve("deductible") == []

def test_citation_validation():
    chunks = [RetrievedChunk("deductible $100","policy.txt",1,.03)]
    answer, cited = validate_answer('{"claims":[{"text":"Deductible: $100.","citations":[1]}]}', chunks)
    assert "policy.txt" in answer and cited == chunks
    for bad in ('not json','{"claims":[{"text":"Claim","citations":[2]}]}',
                '{"claims":[{"text":"Claim","citations":[]}]}',
                '{"claims":[{"text":"Claim","citations":[true]}]}'):
        with pytest.raises(ValueError):
            validate_answer(bad, chunks)

def test_pipeline_followup_and_plan_guard(manager, config, monkeypatch):
    from rag.rag_pipeline import RAGPipeline
    manager.replace_document(record(), [item()])
    captured = []
    class Chat:
        def invoke(self, messages):
            captured.append(messages[-1].content)
            return SimpleNamespace(content='{"claims":[{"text":"The deductible is $100.","citations":[1]}]}')
    pipeline = RAGPipeline(config, manager, Chat())
    monkeypatch.setattr(pipeline.retriever, "retrieve", lambda *a,**k: [
        RetrievedChunk("Deductible $100","policy.txt",1,.03,document_id="doc1")])
    result = pipeline.ask("And out of network?", history=[{"role":"user","content":"What is the deductible?"}])
    assert result.sources and "deductible" in captured[0]
    manager.replace_document(record("doc2", "B"), [item("doc2",cid="two")])
    assert "one plan" in pipeline.ask("deductible").answer

def test_real_chroma_repair_and_model_change(config):
    manager = ChromaManager(FakeEmbedding(), config)
    manager.replace_document(record(), [item()])
    assert not manager.last_vector_error
    assert manager.similarity_search_with_scores("deductible", 1)
    manager.replace_document(record(), [item(text="Appeal within 30 days", cid="new")])
    assert manager.collection.get()["ids"] == ["new"]
    manager.embedding_service.fingerprint = "test-v2"
    assert manager.sync_vectors()
    assert manager.collection.metadata["embedding_model"] == "test-v2"
    manager.clear_collection()
    assert manager.document_count() == manager.collection.count() == 0
