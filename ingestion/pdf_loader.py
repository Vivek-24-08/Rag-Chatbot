"""PDF text/table extraction with optional Tesseract OCR and explicit failures."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import re
import fitz

@dataclass
class PageContent:
    text: str
    page_number: int
    source_file: str
    metadata: dict = field(default_factory=dict)

class PDFLoader:
    def __init__(self, ocr_enabled=False):
        self.ocr_enabled = ocr_enabled
        self.errors = []
        self.warnings = []

    def load_pdf(self, file_path):
        path = Path(file_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        pages = []
        self.warnings = []
        with fitz.open(path) as doc:
            if doc.needs_pass:
                raise ValueError("Password-protected PDF. Upload an unlocked copy.")
            for index, page in enumerate(doc):
                text = page.get_text("text", sort=True)
                if not text.strip() and self.ocr_enabled:
                    try:
                        text = page.get_text(textpage=page.get_textpage_ocr(full=True))
                    except Exception:
                        raise ValueError("OCR requires a working Tesseract installation and language data.")
                # Preserve table row/column relationships as an additional labeled block.
                try:
                    tables = page.find_tables().tables
                    for table in tables:
                        rows = table.extract()
                        text += "\nTABLE (each line is a row):\n" + "\n".join(
                            " | ".join(str(cell or "").replace("\n", " ") for cell in row) for row in rows)
                except Exception:
                    # Plain text remains usable when automatic table detection fails.
                    self.warnings.append(f"Automatic table extraction failed on physical page {index + 1}.")
                text = self._clean_text(text)
                if text:
                    pages.append(PageContent(text, index + 1, path.name, {
                        "source": path.name, "page": index + 1,
                        "page_label": page.get_label() or str(index + 1),
                        "document_id": digest, "version": digest,
                    }))
                else:
                    self.warnings.append(f"No readable text on physical page {index + 1}; check for a scanned page.")
        if not pages:
            raise ValueError("No text found. The PDF is blank or scanned; enable OCR for scanned PDFs.")
        if self.warnings:
            for item in pages:
                item.metadata["extraction_warning"] = " ".join(self.warnings)
        return pages

    def load_multiple_pdfs(self, file_paths):
        pages, self.errors = [], []
        for path in file_paths:
            try:
                pages.extend(self.load_pdf(path))
            except Exception as exc:
                self.errors.append({"source": Path(path).name, "error": type(exc).__name__})
        return pages

    @staticmethod
    def _clean_text(text):
        return re.sub(r"[ \t]{2,}", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()
