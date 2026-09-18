"""Local connectors for PDFs, UTF-8 text/Markdown and JSON FAQ exports."""
import json
import hashlib
from pathlib import Path
from ingestion.pdf_loader import PDFLoader, PageContent

def load_source(path, ocr_enabled=False):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return PDFLoader(ocr_enabled).load_pdf(path)
    version = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {"source": path.name, "document_id": version, "version": version, "page": 1}
    if path.suffix.lower() in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8-sig")
        pages = [PageContent(text, 1, path.name, meta)]
    elif path.suffix.lower() == ".json":
        records = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(records, list):
            raise ValueError("FAQ JSON must be a list of question/answer objects.")
        pages = []
        for i, row in enumerate(records, 1):
            if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
                raise ValueError("Each FAQ entry requires string question and answer fields.")
            pages.append(PageContent("Question: " + row["question"] + "\nAnswer: " + row["answer"],
                                     i, path.name, {**meta, "page": i}))
    else:
        raise ValueError("Supported sources: PDF, TXT, MD and JSON FAQ exports.")
    if not any(p.text.strip() for p in pages):
        raise ValueError("The document contains no text.")
    return pages
