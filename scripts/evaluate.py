"""Tiny reproducible FTS/hybrid plumbing evaluation, not a real-world accuracy claim."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from chunking.text_splitter import Chunk
from utils.config import Config
from vectorstore.chroma_manager import ChromaManager
from retrieval.retriever import Retriever

CASES = [
    ("What is the deductible?", "benefits", "The annual deductible is $100.", 1),
    ("How long to appeal?", "appeals", "Submit an appeal within 30 days.", 2),
    ("What is the pharmacy copay?", "pharmacy", "The pharmacy copay is $10.", 3),
]
def main():
    with tempfile.TemporaryDirectory(prefix="rag-eval-") as folder:
        config = Config(lexical_index_path=str(Path(folder)/"index.db"), chroma_persist_dir=str(Path(folder)/"vectors"))
        manager = ChromaManager(SimpleNamespace(fingerprint="evaluation"), config)
        manager.sync_vectors = lambda: False
        # Vector branch is deliberately empty: no downloaded model and no API call.
        manager.similarity_search_with_scores = lambda *a,**k: []
        for _, identity, text, page in CASES:
            manager.replace_document(dict(id=identity, source=identity+".txt", plan="Synthetic", year="2026",
                    version="1",kind=".txt"), [Chunk(text,dict(source=identity+".txt",page=page,document_id=identity),identity)])
        retriever = Retriever(manager, config)
        hits, reciprocal, details = 0, 0.0, []
        for question, expected, _, page in CASES:
            results = retriever.retrieve(question, top_k=3)
            rank = next((i for i,c in enumerate(results,1) if c.document_id==expected and c.page_number==page),None)
            hits += rank is not None
            reciprocal += 1/rank if rank else 0
            details.append(dict(question=question,expected_document=expected,expected_page=page,rank=rank))
        absent = not retriever.retrieve("astronomy telescope galaxy")
        print(json.dumps(dict(scope="synthetic lexical retrieval only", cases=len(CASES),
              recall_at_3=hits/len(CASES),mrr_at_3=reciprocal/len(CASES),
              unanswerable_query_empty=absent,details=details),indent=2))
        return 0 if hits == len(CASES) and absent else 1

if __name__ == "__main__":
    raise SystemExit(main())
