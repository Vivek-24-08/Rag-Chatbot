"""Optional real local embedding smoke check. May download the configured model."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from utils.config import Config
from rag.rag_pipeline import RAGPipeline

def worker(folder):
    root = Path(folder)
    config = Config(chat_provider="none", embedding_provider="local",
        search_analytics_enabled=False, lexical_index_path=str(root/"index.db"),
        chroma_persist_dir=str(root/"vectors"))
    pipeline = RAGPipeline(config)
    document = root/"synthetic-plan.txt"
    document.write_text("The specialist copay is $25 per visit. The annual deductible is $100.",encoding="utf-8")
    count = pipeline.ingest_documents([document], "Synthetic", "2026")
    if not count or pipeline.chroma_manager.last_vector_error:
        raise RuntimeError("Local ingestion/vector indexing failed.")
    response = pipeline.ask("How much do I pay to see a specialist?")
    if not response.sources or response.warnings:
        raise RuntimeError("Local hybrid search did not return healthy results.")
    print(json.dumps(dict(chunks=count, results=len(response.sources),
          embedding_model=config.active_embedding_model, status="local embedding + hybrid retrieval passed")))

def main():
    # Chroma retains Windows file handles for the process lifetime. A worker
    # must exit before the parent can remove its own temporary test directory.
    with tempfile.TemporaryDirectory(prefix="rag-local-smoke-") as folder:
        subprocess.run([sys.executable, "-m", "scripts.smoke_local", "--worker", folder], check=True)
    return 0

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        raise SystemExit(main())
