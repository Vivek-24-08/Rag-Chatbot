# ==============================================================================
# chunking/text_splitter.py
# ------------------------------------------------------------------------------
# STEP 3 of the RAG pipeline: "Chunk documents"
#
# WHAT THIS FILE DOES:
#   Breaks each page's text into smaller overlapping "chunks" of text that
#   are small enough to (a) embed cheaply and (b) fit comfortably inside an
#   LLM prompt alongside a few other chunks.
#
# WHY CHUNKING IS NECESSARY (GenAI beginners):
#   You might think: "why not just embed each entire PDF page, or the whole
#   document, as one big vector?" Two reasons:
#     1. Embedding models and LLMs have limited context windows — you can't
#        cram a 300-page policy document into one API call.
#     2. Similarity search works best on FOCUSED pieces of text. If a page
#        discusses both "deductibles" and "prior authorization", embedding
#        the whole page as one vector blurs those two topics together,
#        making retrieval less precise. Smaller, focused chunks retrieve
#        more accurately for a specific question.
#
# WHY OVERLAP BETWEEN CHUNKS:
#   If a sentence like "The annual deductible is $1,500 per individual and
#   $3,000 per family" gets cut in half by a hard chunk boundary, neither
#   resulting chunk contains the complete fact. Overlapping the end of one
#   chunk with the start of the next greatly reduces the odds of splitting
#   an important fact in two.
#
# WE USE LangChain's RecursiveCharacterTextSplitter, WHICH WORKS BY:
#   Trying to split on paragraph breaks ("\n\n") first. If a resulting
#   piece is still too big, it falls back to splitting on single newlines,
#   then sentences, then words, then characters — always preferring the
#   split point that keeps the most semantic meaning intact.
# ==============================================================================

from dataclasses import dataclass, field
from typing import List

# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.pdf_loader import PageContent
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """
    A single chunk of text, ready to be embedded, along with the metadata
    it inherited from its source page (filename + page number). This
    metadata is what allows the final chatbot answer to say
    "(Source: Aetna_EOC_2026.pdf, page 14)".
    """

    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""  # assigned by DocumentChunker, e.g. "eoc.pdf_p14_c0"


class DocumentChunker:
    """
    Wraps LangChain's RecursiveCharacterTextSplitter with insurance-document
    friendly defaults, and preserves page-level metadata through the split.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        # Fall back to the values from utils/config.py (which itself falls
        # back to .env, which falls back to sensible hard-coded defaults).
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # separators are tried IN ORDER — LangChain first tries to split on
        # "\n\n" (paragraph breaks). Only if a chunk is still too long does
        # it move down the list to smaller and smaller split points.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        logger.info(
            f"DocumentChunker initialized (chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )

    def chunk_pages(self, pages: List[PageContent]) -> List[Chunk]:
        """
        Split every page's text into chunks, preserving source/page metadata
        on each resulting chunk.

        Args:
            pages: List of PageContent objects from ingestion/pdf_loader.py

        Returns:
            A flat list of Chunk objects across all input pages.
        """
        all_chunks: List[Chunk] = []

        for page in pages:
            # split_text() returns a list of plain strings — the actual
            # splitting logic (paragraph -> sentence -> word -> char) lives
            # inside LangChain's RecursiveCharacterTextSplitter.
            text_pieces = self.splitter.split_text(page.text)

            for i, piece in enumerate(text_pieces):
                chunk_id = f"{page.source_file}_p{page.page_number}_c{i}"
                all_chunks.append(
                    Chunk(
                        text=piece,
                        metadata={
                            **page.metadata,
                            "chunk_index_on_page": i,
                        },
                        chunk_id=chunk_id,
                    )
                )

        logger.info(f"Split {len(pages)} pages into {len(all_chunks)} chunks")
        return all_chunks
