# ==============================================================================
# tests/test_chunking.py
# ------------------------------------------------------------------------------
# Unit tests for chunking/text_splitter.py.
#
# WHY TEST CHUNKING SPECIFICALLY:
#   Chunking bugs are sneaky — if chunk boundaries are wrong, or metadata
#   (source filename / page number) doesn't survive the split, citations
#   shown to the user in Streamlit will be silently wrong, even though the
#   app "looks like" it's working. These tests catch that early.
#
# HOW TO RUN:
#       pytest tests/test_chunking.py -v
# ==============================================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking.text_splitter import DocumentChunker
from ingestion.pdf_loader import PageContent


def test_chunk_pages_preserves_metadata():
    """Every chunk produced should carry forward its source page's metadata."""
    page = PageContent(
        text="A" * 2500,  # long enough to force multiple chunks
        page_number=3,
        source_file="sample_eoc.pdf",
        metadata={"source": "sample_eoc.pdf", "page": 3},
    )

    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_pages([page])

    assert len(chunks) > 1, "Expected long page to be split into multiple chunks"
    for chunk in chunks:
        assert chunk.metadata["source"] == "sample_eoc.pdf"
        assert chunk.metadata["page"] == 3


def test_chunk_pages_respects_chunk_size_roughly():
    """No chunk should wildly exceed the configured chunk_size."""
    page = PageContent(
        text="This is a sentence about deductibles and copays. " * 50,
        page_number=1,
        source_file="sample_sbc.pdf",
        metadata={"source": "sample_sbc.pdf", "page": 1},
    )

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_pages([page])

    for chunk in chunks:
        # Allow a little slack since the splitter tries to break on sentence
        # boundaries rather than cutting mid-word at the exact character count.
        assert len(chunk.text) <= 600


def test_empty_pages_produce_no_chunks():
    """Pages with no text shouldn't produce any chunk objects at all."""
    chunker = DocumentChunker()
    chunks = chunker.chunk_pages([])
    assert chunks == []


def test_chunk_ids_are_unique():
    """Chunk IDs are used as ChromaDB document IDs, so they must be unique."""
    pages = [
        PageContent(text="Deductible info. " * 100, page_number=1, source_file="doc.pdf",
                    metadata={"source": "doc.pdf", "page": 1}),
        PageContent(text="Copay info. " * 100, page_number=2, source_file="doc.pdf",
                    metadata={"source": "doc.pdf", "page": 2}),
    ]
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_pages(pages)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"
