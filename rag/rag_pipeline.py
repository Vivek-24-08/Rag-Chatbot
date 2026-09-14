# ==============================================================================
# rag/rag_pipeline.py
# ------------------------------------------------------------------------------
# This is the ORCHESTRATOR of the whole RAG pipeline. It ties together
# every step, Steps 6 through 9:
#     Step 6: Perform similarity search        (via retrieval/retriever.py)
#     Step 7: Retrieve top matching chunks      (via retrieval/retriever.py)
#     Step 8: Send retrieved chunks to the LLM  (via prompts/prompt_template.py)
#     Step 9: Generate a grounded response      (via OpenAI GPT-4o)
#
# WHAT IS "RAG" IN ONE SENTENCE? (GenAI beginners)
#   Retrieval-Augmented Generation = instead of asking an LLM to answer
#   purely from what it memorized during training, we first RETRIEVE the
#   most relevant snippets of TRUSTED, up-to-date documents, and then ask
#   the LLM to GENERATE an answer using ONLY those snippets. This is what
#   lets a general-purpose model like GPT-4o answer accurately about a
#   SPECIFIC insurance plan's specific rules — something it was never trained on.
#
# HOW LANGCHAIN IS USED HERE:
#   We use LangChain's ChatOpenAI wrapper to call GPT-4o. LangChain
#   standardizes how you send "messages" (system/user/assistant turns) to
#   different LLM providers, so if you ever wanted to swap GPT-4o for
#   another model, most of this code wouldn't need to change.
#
# ALSO PROVIDES: ingest_documents(), which runs Steps 1-5 (upload -> extract
# -> chunk -> embed -> store) for newly uploaded PDFs, before any questions
# can be answered.
# ==============================================================================

from dataclasses import dataclass, field
import time
from typing import List

from chunking.text_splitter import DocumentChunker
from embeddings.embedding_service import EmbeddingService
from ingestion.pdf_loader import PDFLoader
from intelligence.search_intelligence import SearchIntelligence
from prompts.prompt_template import SYSTEM_PROMPT, build_user_prompt
from retrieval.retriever import Retriever, RetrievedChunk
from vectorstore.chroma_manager import ChromaManager
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """
    The final answer package returned to the Streamlit frontend: the
    generated text plus the exact chunks it was grounded in, so the UI can
    render citations underneath the chat bubble.
    """

    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)
    search_id: str = ""
    search_intent: str = "general"


class RAGPipeline:
    """
    The single entry point the Streamlit frontend talks to. It exposes just
    two methods:
      - ingest_documents(file_paths): run Steps 1-5 on newly uploaded PDFs
      - ask(question): run Steps 6-9 to answer a user's question
    """

    def __init__(self):
        # --- Shared building blocks, wired together once at startup -------
        self.pdf_loader = PDFLoader()
        self.chunker = DocumentChunker()
        self.embedding_service = EmbeddingService()
        self.chroma_manager = ChromaManager(embedding_service=self.embedding_service)
        self.retriever = Retriever(self.chroma_manager)
        self.search_intelligence = SearchIntelligence(
            settings.search_analytics_db, settings.search_analytics_enabled
        )

        # Initialize the LLM based on the configured chat provider.
        # Document ingestion works independently (uses local or OpenAI embeddings).
        self.llm = None
        if settings.chat_provider == "databricks":
            from langchain_databricks import ChatDatabricks
            self.llm = ChatDatabricks(
                endpoint=settings.databricks_chat_endpoint,
                temperature=0,
            )
            logger.info(f"RAGPipeline initialized with Databricks LLM "
                        f"(endpoint='{settings.databricks_chat_endpoint}')")
        elif settings.chat_provider == "openai" and settings.has_openai_key:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=settings.chat_model,
                temperature=0,
                openai_api_key=settings.openai_api_key,
            )
            logger.info("RAGPipeline initialized with OpenAI LLM")
        else:
            logger.info("RAGPipeline initialized WITHOUT LLM. "
                        "Document ingestion is available; chat requires a provider.")

    # -------------------------------------------------------------------
    # INGESTION: Steps 1 -> 5 (upload, extract, chunk, embed, store)
    # -------------------------------------------------------------------
    def ingest_documents(self, file_paths: List[str]) -> int:
        """
        Run the full ingestion pipeline on a list of uploaded PDF file paths.

        Step 1 (Upload) happens in the Streamlit UI (frontend/streamlit_ui.py),
        which saves uploaded files to disk and passes their paths here.

        Args:
            file_paths: paths to PDF files already saved on disk, e.g.
                        ["data/pdfs/Aetna_EOC_2026.pdf"]

        Returns:
            The number of chunks successfully embedded and stored.
        """
        logger.info(f"Starting ingestion for {len(file_paths)} file(s)")

        # Step 2: Extract text from PDFs
        pages = self.pdf_loader.load_multiple_pdfs(file_paths)
        if not pages:
            logger.warning("No text could be extracted from the uploaded PDF(s)")
            return 0

        # Step 3: Chunk documents
        chunks = self.chunker.chunk_pages(pages)

        # Step 4 + 5: Generate embeddings & store them in ChromaDB
        # (embedding happens INSIDE add_chunks(), via the embedding_function
        # that was wired into ChromaManager's Chroma vectorstore)
        stored_count = self.chroma_manager.add_chunks(chunks)

        logger.info(f"Ingestion complete: {stored_count} chunks stored")
        return stored_count

    # -------------------------------------------------------------------
    # QUESTION ANSWERING: Steps 6 -> 9 (retrieve, augment, generate)
    # -------------------------------------------------------------------
    def ask(self, question: str, session_id: str = "", history: List[dict] = None) -> RAGResponse:
        """
        Answer a user's question using only the uploaded documents.

        Args:
            question: the user's natural-language question

        Returns:
            A RAGResponse containing the generated answer text and the
            source chunks it was grounded in (for citation display).
        """
        # Guard clause: if nothing has been uploaded/indexed yet, don't
        # even call the LLM — just tell the user clearly what to do.
        if not self.retriever.has_documents():
            return RAGResponse(
                answer=(
                    "I don't have any insurance documents to search yet. "
                    "Please upload a PDF (like your Evidence of Coverage or "
                    "Summary of Benefits and Coverage) using the sidebar first."
                ),
                sources=[],
            )

        # Step 6 + 7: similarity search -> top-K retrieved chunks
        processed_query = self.search_intelligence.process(question, history)
        started = time.perf_counter()
        retrieved_chunks = self.retriever.retrieve(processed_query.search_text)
        preferred_sources = self.search_intelligence.preferred_sources(session_id)
        if preferred_sources:
            retrieved_chunks.sort(key=lambda chunk: chunk.similarity_score + (
                settings.feedback_source_boost if chunk.source_file in preferred_sources else 0
            ), reverse=True)
        search_id = self.search_intelligence.record_search(
            session_id, processed_query, (time.perf_counter() - started) * 1000,
            [chunk.source_file for chunk in retrieved_chunks],
        )

        # Guard: if no LLM is available, return a clear message.
        if self.llm is None:
            return RAGResponse(
                answer=(
                    "⚠️ The OpenAI API key is not configured or has no credits. "
                    "Document ingestion works with local embeddings, but "
                    "answering questions requires a valid OpenAI API key. "
                    "Please add credits at https://platform.openai.com/settings/organization/billing/"
                ),
                sources=retrieved_chunks,
                search_id=search_id,
                search_intent=processed_query.intent,
            )

        # Step 8: Send retrieved chunks + question to the LLM.
        from langchain_core.messages import SystemMessage, HumanMessage
        user_prompt = build_user_prompt(question, retrieved_chunks)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        logger.info(f"Sending question to {settings.chat_model} with {len(retrieved_chunks)} context chunks")

        # Step 9: Generate a grounded response.
        # self.llm.invoke() sends the messages to OpenAI's chat completion
        # endpoint and returns an AIMessage; .content is the text answer.
        ai_message = self.llm.invoke(messages)
        answer_text = ai_message.content

        logger.info("Received grounded response from LLM")

        return RAGResponse(answer=answer_text, sources=retrieved_chunks,
                           search_id=search_id, search_intent=processed_query.intent)
