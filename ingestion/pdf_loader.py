# ==============================================================================
# ingestion/pdf_loader.py
# ------------------------------------------------------------------------------
# STEP 1 & 2 of the RAG pipeline: "Upload PDF files" + "Extract text from PDFs"
#
# WHAT THIS FILE DOES:
#   Takes raw PDF files (e.g., Aetna's Evidence of Coverage, Summary of
#   Benefits and Coverage) and converts them into plain text that the rest
#   of the pipeline can work with — LLMs and embedding models understand
#   text, not PDF byte streams.
#
# WHY PyMuPDF (imported as "fitz"):
#   PyMuPDF is fast and preserves page structure well, which lets us tag
#   every chunk of text with the PAGE NUMBER it came from. That page number
#   is what eventually powers the "citations" feature in the Streamlit UI —
#   so a user asking "what's my deductible?" can see it was found on
#   page 14 of the EOC, not just take the chatbot's word for it.
#
# HOW THIS FITS INTO THE BIGGER RAG PICTURE:
#   PDF file  --(this file)-->  plain text + metadata  --(chunking)-->
#   text chunks  --(embeddings)-->  vectors  --(vectorstore)--> ChromaDB
# ==============================================================================

import os
from dataclasses import dataclass, field
from typing import List

import fitz  # PyMuPDF's import name is "fitz" (its historical codename)

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PageContent:
    """
    Holds the extracted text for a single PDF page, plus metadata about
    where it came from. We keep metadata attached to every piece of text
    from the very first step, so it can travel all the way through
    chunking -> embedding -> vector storage -> retrieval -> final citation.
    """

    text: str
    page_number: int          # 1-indexed page number (human-friendly)
    source_file: str          # original PDF filename, e.g. "Aetna_EOC_2026.pdf"
    metadata: dict = field(default_factory=dict)


class PDFLoader:
    """
    Loads one or more PDF files from disk and extracts their text,
    page by page, using PyMuPDF.
    """

    def load_pdf(self, file_path: str) -> List[PageContent]:
        """
        Extract text from a single PDF file, one PageContent object per page.

        Args:
            file_path: Path to a .pdf file on disk (e.g., "data/pdfs/eoc.pdf")

        Returns:
            A list of PageContent objects, one per non-empty page.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found at: {file_path}")

        filename = os.path.basename(file_path)
        pages: List[PageContent] = []

        # fitz.open() opens the PDF and gives us page-by-page access.
        # Using a "with" block ensures the file handle is always closed,
        # even if something goes wrong mid-extraction.
        with fitz.open(file_path) as doc:
            logger.info(f"Opened '{filename}' with {doc.page_count} pages")

            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)

                # get_text("text") returns plain reading-order text.
                # PyMuPDF also supports "blocks", "words", "html", etc.,
                # but plain text is what our embedding model needs.
                raw_text = page.get_text("text")
                cleaned_text = self._clean_text(raw_text)

                # Skip pages that are blank or contain only whitespace
                # (common on section-divider pages) — no point embedding
                # empty content, it just wastes API calls and storage.
                if not cleaned_text.strip():
                    continue

                pages.append(
                    PageContent(
                        text=cleaned_text,
                        page_number=page_index + 1,  # humans count from 1, not 0
                        source_file=filename,
                        metadata={"source": filename, "page": page_index + 1},
                    )
                )

        logger.info(f"Extracted text from {len(pages)} non-empty pages in '{filename}'")
        return pages

    def load_multiple_pdfs(self, file_paths: List[str]) -> List[PageContent]:
        """
        Convenience method to load and extract text from several PDFs at
        once (e.g., EOC + SBC + Medicare Managed Care Manual all uploaded
        together). Returns one combined list of PageContent objects.
        """
        all_pages: List[PageContent] = []
        for path in file_paths:
            try:
                all_pages.extend(self.load_pdf(path))
            except Exception as exc:
                # We log and continue rather than crashing the whole batch
                # if a single malformed/corrupt PDF is uploaded.
                logger.error(f"Failed to load '{path}': {exc}")
        return all_pages

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Light text cleanup applied to every extracted page:
          - collapse repeated whitespace/newlines that PDFs often produce
            (e.g., from multi-column layouts or table extraction quirks)
          - strip leading/trailing whitespace

        We keep this intentionally minimal: aggressive cleaning can
        accidentally delete meaningful content (like "$0 copay" formatting),
        which would hurt answer accuracy.
        """
        # Replace 3+ consecutive newlines with just 2 (paragraph break)
        import re
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse runs of spaces/tabs into a single space
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()
