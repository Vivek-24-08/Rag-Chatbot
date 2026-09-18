"""Ingestion and answer orchestration. Providers are initialized only when needed."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import time
from chunking.text_splitter import DocumentChunker
from embeddings.embedding_service import EmbeddingService
from ingestion.source_loader import load_source
from intelligence.search_intelligence import SearchIntelligence
from prompts.prompt_template import SYSTEM_PROMPT, build_user_prompt, select_context, validate_answer
from retrieval.retriever import Retriever
from vectorstore.chroma_manager import ChromaManager
from utils.config import settings
from utils.logger import get_logger
logger = get_logger(__name__)

@dataclass
class RAGResponse:
    answer: str
    sources: list = field(default_factory=list)
    search_id: str = ""
    search_intent: str = "general"
    warnings: list = field(default_factory=list)

class RAGPipeline:
    def __init__(self, config=None, manager=None, llm=None):
        self.config = config or settings
        self.embedding_service = manager.embedding_service if manager else EmbeddingService(self.config)
        self.chroma_manager = manager or ChromaManager(self.embedding_service, self.config)
        self.retriever = Retriever(self.chroma_manager, self.config)
        self.search_intelligence = SearchIntelligence(self.config.search_analytics_db,
            self.config.search_analytics_enabled, self.config.retention_days)
        self.llm = llm
        self.last_ingestion_errors = []
        self.last_ingestion_warnings = []

    def _chat(self):
        if self.llm is not None:
            return self.llm
        self.config.validate_chat()
        if self.config.chat_provider == "databricks":
            from langchain_databricks import ChatDatabricks
            self.llm = ChatDatabricks(endpoint=self.config.databricks_chat_endpoint,
                                     temperature=0, max_tokens=self.config.max_answer_tokens)
        else:
            from langchain_openai import ChatOpenAI
            options = dict(model=self.config.active_chat_model, temperature=0,
                           max_tokens=self.config.max_answer_tokens,
                           timeout=self.config.request_timeout, max_retries=2)
            if self.config.chat_provider == "openrouter":
                options.update(api_key=self.config.openrouter_api_key, base_url="https://openrouter.ai/api/v1")
            else:
                options["api_key"] = self.config.openai_api_key
            self.llm = ChatOpenAI(**options)
        return self.llm

    def ingest_documents(self, file_paths, plan="", year="", replace_document_id=None):
        if replace_document_id and len(file_paths) != 1:
            raise ValueError("Replace one document at a time.")
        total = 0
        self.last_ingestion_errors = []
        self.last_ingestion_warnings = []
        for value in file_paths:
            path = Path(value)
            stage = "reading source"
            try:
                if path.stat().st_size > 25 * 1024 * 1024:
                    raise ValueError("File exceeds the 25 MB limit.")
                pages = load_source(path, self.config.ocr_enabled)
                self.last_ingestion_warnings.extend(sorted({
                    p.metadata["extraction_warning"] for p in pages if p.metadata.get("extraction_warning")}))
                payload = path.read_bytes()
                version = hashlib.sha256(payload).hexdigest()
                identity = replace_document_id or hashlib.sha256(
                    json.dumps([version, plan.strip(), str(year).strip()], ensure_ascii=False).encode()).hexdigest()
                if replace_document_id and identity not in {d["id"] for d in self.chroma_manager.list_documents()}:
                    raise ValueError("Replacement target no longer exists.")
                for page in pages:
                    page.metadata.update(document_id=identity, plan=plan.strip(), year=str(year).strip())
                stage = "loading embedding tokenizer and splitting"
                # Token limit comes from the actual configured embedding tokenizer.
                chunker = DocumentChunker(self.config.chunk_size, self.config.chunk_overlap,
                    self.embedding_service.token_count, self.embedding_service.token_limit)
                chunks = chunker.chunk_pages(pages)
                record = dict(id=identity, source=path.name, plan=plan.strip(), year=str(year).strip(),
                              version=version, kind=path.suffix.lower())
                stage = "saving document"
                total += self.chroma_manager.replace_document(record, chunks, payload)
            except Exception as exc:
                # Do not expose parser/provider exceptions, which may contain private content.
                self.last_ingestion_errors.append({"source": path.name, "error": type(exc).__name__, "stage": stage})
                logger.warning("ingestion_failed error=%s", type(exc).__name__)
        return total

    def ask(self, question, session_id="", history=None, document_ids=None):
        processed = self.search_intelligence.process(question, history)
        if processed.needs_clarification:
            return RAGResponse("Which benefit or service does your follow-up question refer to?")
        documents = self.chroma_manager.list_documents()
        selected = [d for d in documents if document_ids is None or d["id"] in document_ids]
        if not selected:
            return RAGResponse("Upload and select at least one document to search.")
        plans = {(d["plan"], d["year"]) for d in selected}
        if len(plans) > 1:
            return RAGResponse("Please select documents for one plan and year. This prevents mixing different coverage rules.")
        started = time.perf_counter()
        cited, warnings, status = [], [], "answered"
        try:
            base = (self.config.hybrid_vector_weight, self.config.hybrid_lexical_weight)
            weights = tuple(a*b for a,b in zip(base, processed.weights))
            chunks = self.retriever.retrieve(processed.search_text, document_ids=[d["id"] for d in selected],
                        weights=weights, preferences=self.search_intelligence.preferred_sources(session_id))
            warnings.extend(self.retriever.last_warnings)
            chunks = select_context(chunks, self.config.context_tokens)
            if not chunks:
                status = "no_evidence"
                answer = "I couldn't find sufficient evidence in the selected documents. Try a more specific question or upload the relevant plan document."
            elif self.config.chat_provider == "none" and self.llm is None:
                status = "search_only"
                answer = "Search results are available below. Configure a chat provider in .env to generate an answer."
                cited = chunks
            else:
                from langchain_core.messages import SystemMessage, HumanMessage
                response = self._chat().invoke([SystemMessage(content=SYSTEM_PROMPT),
                                HumanMessage(content=build_user_prompt(processed.search_text, chunks))])
                answer, cited = validate_answer(response.content, chunks)
                if not cited:
                    status = "no_evidence"
        except Exception as exc:
            logger.warning("answer_failed error=%s", type(exc).__name__)
            status = "error"
            answer = ("I couldn't produce a verified answer. Check the provider/model configuration and connection, "
                      "then try again. No unverified model answer was displayed.")
        event = self.search_intelligence.record_search(session_id, processed,
                    (time.perf_counter()-started)*1000, [c.document_id for c in cited] if status == "answered" else [], status)
        if self.search_intelligence.last_error:
            warnings.append("Search analytics are unavailable; this does not prevent answering.")
        logger.info("search_complete event=%s status=%s", event or "not-recorded", status)
        return RAGResponse(answer, cited, event, processed.intent, warnings)
