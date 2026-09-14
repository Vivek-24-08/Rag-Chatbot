# 🏥 Insurance Virtual Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers insurance
questions **strictly from documents you upload** — Evidence of Coverage
(EOC), Summary of Benefits and Coverage (SBC), the Medicare Managed Care
Manual, or any other plan PDF. Every answer is grounded in retrieved
document text and cited with a source filename + page number.

## How it works (10 steps)

| # | Step | Where |
|---|------|-------|
| 1 | Upload PDF files | `frontend/streamlit_ui.py` |
| 2 | Extract text from PDFs | `ingestion/pdf_loader.py` |
| 3 | Chunk documents | `chunking/text_splitter.py` |
| 4 | Generate embeddings | `embeddings/embedding_service.py` |
| 5 | Store embeddings in vector database | `vectorstore/chroma_manager.py` |
| 6 | Perform similarity search | `retrieval/retriever.py` |
| 7 | Retrieve top matching chunks | `retrieval/retriever.py` |
| 8 | Send retrieved chunks to the LLM | `prompts/prompt_template.py` |
| 9 | Generate a grounded response | `rag/rag_pipeline.py` |
| 10 | Display answer with citations | `frontend/streamlit_ui.py` |

`rag/rag_pipeline.py` is the orchestrator that wires steps 1–9 together;
`app.py` is the single entry point that launches the Streamlit UI.

## Tech stack

- **Frontend:** Streamlit
- **LLM:** OpenAI GPT-4o
- **Embeddings:** OpenAI `text-embedding-3-large`
- **Framework:** LangChain
- **Vector DB:** ChromaDB (local, persisted to disk)
- **PDF parsing:** PyMuPDF

## Folder structure

```
rag-chatbot/
├── app.py                       # Entry point: `streamlit run app.py`
├── requirements.txt
├── .env.example                 # Copy to .env and add your OpenAI key
├── .gitignore
│
├── data/
│   ├── pdfs/                    # Uploaded source PDFs land here
│   └── chroma_db/                # Persisted vector database
│
├── ingestion/
│   └── pdf_loader.py             # Step 1-2: PDF -> plain text (PyMuPDF)
│
├── chunking/
│   └── text_splitter.py          # Step 3: text -> overlapping chunks
│
├── embeddings/
│   └── embedding_service.py      # Step 4: text -> vectors (OpenAI)
│
├── vectorstore/
│   └── chroma_manager.py         # Step 5: store/search vectors (ChromaDB)
│
├── retrieval/
│   └── retriever.py              # Step 6-7: similarity search -> top-K chunks
│
├── rag/
│   └── rag_pipeline.py           # Orchestrates ingestion + Step 8-9 (LLM call)
│
├── prompts/
│   └── prompt_template.py        # Grounded, citation-enforcing system prompt
│
├── frontend/
│   └── streamlit_ui.py           # Step 1 & 10: upload UI + chat + citations
│
├── utils/
│   ├── config.py                  # Centralized settings (reads .env)
│   └── logger.py                  # Shared logging setup
│
└── tests/
    ├── test_chunking.py
    └── test_prompt_template.py
```

## Setup

1. **Clone/copy this project**, then create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your API key:**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and paste in your real `OPENAI_API_KEY` (from
   https://platform.openai.com/api-keys). You can also adjust the chat
   model, embedding model, chunk size, and top-K retrieval count here.

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   This opens the app in your browser (usually `http://localhost:8501`).

## Using the app

1. In the sidebar, upload one or more PDFs (EOC, SBC, Medicare Managed
   Care Manual, etc.) and click **"Ingest Documents"**. This runs Steps
   1–5: text extraction, chunking, embedding, and storage in ChromaDB.
2. Once you see **"✅ Knowledge base ready"**, ask a question in the chat
   box at the bottom — e.g., *"What is my annual deductible?"* or
   *"Do I need a referral to see a specialist?"*
3. Each answer includes an expandable **"📚 Sources"** panel showing which
   document and page every fact came from, so you can verify it against
   the original PDF.
4. Use **"🗑️ Clear knowledge base"** in the sidebar to wipe all indexed
   documents and start fresh.

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

These tests check chunking behavior and prompt construction — they do
**not** call the OpenAI API, so they run for free without an API key.

## Key design decisions

- **Grounded-only answers:** The system prompt (`prompts/prompt_template.py`)
  strictly instructs GPT-4o to answer only from retrieved document chunks,
  and to say "I don't know" (and point the user to Member Services) rather
  than guess — important for an insurance use case where wrong answers
  about coverage or deadlines can have real consequences.
- **Citations by design:** Every chunk carries `source_file` +
  `page_number` metadata from the moment it's extracted (Step 2) all the
  way through to the final answer (Step 10), so nothing has to be
  "guessed" or reconstructed after the fact.
- **Local-first:** ChromaDB persists to a local folder (`data/chroma_db/`),
  so the whole pipeline (except the OpenAI API calls themselves) runs on
  your machine with no external database service to set up.

## Extending this project

- **New document types:** Add more PDFs to the sidebar uploader — no code
  changes needed, since `ingestion/pdf_loader.py` and the rest of the
  pipeline are format-agnostic beyond "it's a PDF."
- **Swap the vector DB:** Because `retrieval/retriever.py` and
  `rag/rag_pipeline.py` depend only on `ChromaManager`'s small interface
  (`add_chunks`, `similarity_search_with_scores`, `document_count`,
  `clear_collection`), you could implement an alternative manager (e.g.,
  for Pinecone or FAISS) and swap it in with minimal changes elsewhere.
- **Swap the LLM:** Change `OPENAI_CHAT_MODEL` in `.env`, or replace
  `ChatOpenAI` in `rag/rag_pipeline.py` with another LangChain chat model
  integration.

## Disclaimer

This assistant summarizes uploaded plan documents; it does not provide
legal, medical, or financial advice, and it does not make coverage
decisions. Always confirm important details with Member Services or
your official plan documents.
