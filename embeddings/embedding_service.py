# ==============================================================================
# embeddings/embedding_service.py
# ------------------------------------------------------------------------------
# STEP 4 of the RAG pipeline: "Generate embeddings"
#
# WHAT IS AN EMBEDDING? (GenAI beginners)
#   An embedding is a list of numbers (a "vector") that represents the
#   MEANING of a piece of text. For example, the sentences
#       "What is my annual deductible?"
#       "How much do I pay before insurance starts covering costs?"
#   are worded completely differently, but they mean almost the same thing.
#   A good embedding model will place both sentences' vectors close together
#   in a high-dimensional mathematical space — even though they share almost
#   no words in common.
#
#   This is the "magic" that makes RAG retrieval smarter than old-fashioned
#   keyword search (like Ctrl+F): it can find the RIGHT document chunk even
#   if the user's question uses totally different vocabulary than the
#   document itself.
#
# WHY text-embedding-3-large SPECIFICALLY:
#   It's OpenAI's most capable embedding model, producing 3072-dimensional
#   vectors. Higher dimensionality generally means it captures more nuance
#   in meaning — useful for dense, jargon-heavy insurance text (deductibles
#   vs. copays vs. coinsurance are subtly different concepts an embedding
#   model needs to distinguish).
#
# HOW THIS FITS INTO THE BIGGER PICTURE:
#   text chunks --(this file, called TWICE: once at ingest time, once per
#   user question)--> vectors --(vectorstore/chroma_manager.py)--> stored,
#   or compared for similarity search.
# ==============================================================================

from typing import List

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Provides text embeddings via either OpenAI (paid, requires API key) or
    a local HuggingFace model (free, runs on CPU).

    The provider is auto-detected from utils/config.py based on whether an
    OpenAI API key is present. Both providers expose the same interface so
    the rest of the pipeline doesn't need to care which one is active.
    """

    def __init__(self):
        if settings.embedding_provider == "openai":
            settings.validate()  # fail fast if API key missing
            from langchain_openai import OpenAIEmbeddings
            self.client = OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key,
            )
            logger.info(f"EmbeddingService initialized with OpenAI model='{settings.embedding_model}'")
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.client = HuggingFaceEmbeddings(
                model_name=settings.local_embedding_model,
            )
            logger.info(f"EmbeddingService initialized with local model='{settings.local_embedding_model}'")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a batch of text chunks into embedding vectors.
        Used during INGESTION (Step 4), once per uploaded document.

        Args:
            texts: list of chunk strings, e.g. ["The annual deductible is...", ...]

        Returns:
            A list of vectors (each vector is a list of floats), in the
            same order as the input texts.
        """
        logger.info(f"Embedding {len(texts)} document chunks...")
        vectors = self.client.embed_documents(texts)
        logger.info(f"Generated {len(vectors)} embeddings (dimension={len(vectors[0]) if vectors else 0})")
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """
        Convert a single piece of text — normally the USER'S QUESTION —
        into an embedding vector. Used during RETRIEVAL (Step 6), once per
        chat message the user sends.

        Args:
            text: the user's natural-language question, e.g.
                  "What's my copay for an ER visit?"

        Returns:
            A single embedding vector (list of floats).
        """
        return self.client.embed_query(text)

    def get_langchain_embeddings(self):
        """
        Expose the underlying LangChain OpenAIEmbeddings object directly.
        Useful because LangChain's Chroma vector store wrapper (used in
        vectorstore/chroma_manager.py) wants an embeddings OBJECT, not just
        our embed_documents()/embed_query() convenience methods, so it can
        call embeddings internally whenever needed.
        """
        return self.client
