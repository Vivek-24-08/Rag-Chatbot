# Insurance Document Assistant — hybrid retrieval and guided search

A local, single-owner Streamlit chatbot. It searches uploaded plan documents using semantic vectors **and** keyword search, then asks your selected model to answer with checked source references.

This is a learning/demo application, not a coverage decision system. It cannot guarantee that a model's interpretation is correct. Always verify important answers with the official plan document or insurer.

## Start here on Windows

Use the **inner folder** containing `app.py` and `requirements.txt`:

```powershell
cd "$env:USERPROFILE\Downloads\aetna-rag-chatbot(1)\aetna-rag-chatbot(1)"
```

Use Python 3.11 or 3.12. Do not use the older Anaconda base Python 3.9.7 for this app. If this project's `.venv` is already installed, skip environment creation and installation.

For a fresh installation with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-windows-py312.txt
.\.venv\Scripts\python.exe -m pip check
```

The install is sizeable because local embeddings use PyTorch. It does not change Anaconda base. The Windows/Python 3.12 lock file records the tested complete environment. For Python 3.11 or other operating systems, use requirements-dev.txt; that file pins direct dependencies but not all transitive dependencies.

### Configure your OpenRouter key

1. Copy `.env.example` to a file named exactly `.env`, **only if .env does not already exist**.
2. Open `.env` in a text editor.
3. Set these three entries:

```dotenv
CHAT_PROVIDER=openrouter
OPENROUTER_API_KEY=your_actual_key_here
OPENROUTER_MODEL=the_exact_model_id_from_your_OpenRouter_account
```

Use a chat model capable of following JSON instructions. Do not assume a model is free or available without checking your account. Never paste your key into source code, chat, screenshots, or GitHub.

Keep `EMBEDDING_PROVIDER=local`. OpenRouter is used for answers, not embeddings. The local embedding model downloads on first use; afterward it can use its cache. If using OpenAI embeddings instead, set `EMBEDDING_PROVIDER=openai` and `OPENAI_API_KEY` separately.

Without a key, set `CHAT_PROVIDER=none` for document search only.

### Check and run

```powershell
.\.venv\Scripts\python.exe -m scripts.doctor
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open [the local chatbot](http://localhost:8501). Leave the terminal running; press Ctrl+C there to stop. Restart after changing `.env`.

The doctor checks installation and configuration presence, not whether an API key has credits or is valid. Tests do not call paid APIs.

### Use the chatbot

1. Enter a plan name and year in the sidebar.
2. Upload a readable PDF, UTF-8 TXT/Markdown file, or JSON FAQ export.
3. Click **Index documents**. The first local model download can take time.
4. Select the documents you want to search. Select one plan/year at a time.
5. Ask a specific question, such as “What is the specialist copay?”
6. Open **Evidence and citations** and inspect/download the original document.
7. Follow up with “What about out of network?” The previous user questions are included in the search and model question.

A JSON FAQ export looks like:

```json
[{"question":"What is the copay?","answer":"The specialist copay is $20."}]
```

The upload limit is 25 MB per file. Uploads are staged in temporary folders, then removed automatically. Original document bytes are retained in the local document database for source downloads.

To replace a document, choose its replacement target before indexing one new upload. To delete, select a document and tick the confirmation box. Clear knowledge base removes the current index's documents, not unrelated files or legacy stores.

## What happens inside

1. **Configuration:** one validated settings object loads `.env`; relative data paths are anchored to this repository, not the terminal's current directory.
2. **Ingestion:** PDF text/table extraction or a local text/FAQ connector produces pages with source, physical page number, content version and document identity.
3. **Chunking:** page-aware splitting, configurable character overlap and tokenizer-aware subdivision prevent embedding truncation. Page boundaries are preserved for reliable citations.
4. **Storage:** SQLite holds documents, original bytes, chunks and an FTS5 keyword index. Replacement is transactional. Chroma is a derived vector index that can be rebuilt.
5. **Query processing:** bounded conversational context resolves simple follow-ups. Word-boundary rules identify costs, coverage, claims, appeals, providers and pharmacy topics.
6. **Routing:** costs/pharmacy/appeals increase the keyword branch's relative weight. Document selection filters **both** search branches.
7. **Hybrid ranking:** independently retrieve candidates, remove duplicates and combine ranks using weighted Reciprocal Rank Fusion: `weight / (60 + rank)`. RRF scores are ranking scores, not confidence probabilities.
8. **Personalization:** optional session feedback adjusts cited-document scores by at most 15% by default, before final top-K selection. Changing a rating replaces it; negative ratings can reverse preferences.
9. **Evidence and answer:** weak semantic-only matches are rejected by a configurable cosine-distance threshold. Only bounded evidence is sent to the model. The model returns structured claims/citations.
10. **Validation and display:** invalid JSON, missing citations or references outside the supplied evidence are rejected. Citations are rendered from trusted metadata, not model-invented filenames.
11. **Monitoring:** optional local aggregate analytics track intent, end-to-end answer latency, no-evidence cases, errors and feedback.

These “intelligent” features are transparent rules and bounded feedback adjustments. They are **not** a trained intent classifier, fine-tuning, reinforcement learning, or a guarantee of semantic accuracy.

## Components

| Location | Responsibility |
|---|---|
| `app.py` | Single UI entry point and safe startup error |
| `utils/config.py` | Validated provider, retrieval and privacy settings |
| `ingestion/` | PDF, text/Markdown and FAQ loading |
| `chunking/text_splitter.py` | Stable chunk identity and token-safe splitting |
| `embeddings/embedding_service.py` | Shared local model or optional OpenAI embeddings |
| `vectorstore/chroma_manager.py` | Authoritative SQLite/FTS and repairable Chroma index |
| `retrieval/retriever.py` | Candidate fusion, evidence gate, deduplication, feedback ranking |
| `intelligence/search_intelligence.py` | Follow-ups, routing, consent-based analytics and feedback |
| `prompts/prompt_template.py` | Untrusted-document boundaries and citation validation |
| `rag/rag_pipeline.py` | Ingestion, plan guard, lazy providers and answer orchestration |
| `frontend/streamlit_ui.py` | Upload, selection, chat, feedback and document lifecycle |
| `scripts/` | Diagnostics, synthetic evaluation and aggregate reports |
| `tests/` | Offline regression tests including real local Chroma with fake embeddings |

## Defaults and tuning

See `.env.example` for every supported environment setting.

- Without configuration, chat is `none`; embeddings are local MiniLM.
- Chunks: 1,000 characters with 200 overlap; local tokenizer limit is also enforced.
- Retrieve 20 candidates per branch, then choose 5 results.
- RRF constant: 60; base semantic/keyword weights: 1/1.
- Maximum semantic cosine distance: 0.65. Tune on representative labelled questions, not intuition.
- Feedback multiplier fraction: 0.15; configuration cannot exceed 0.25.
- Context budget: 3,500 tokens, enforced with a conservative UTF-8 byte upper bound for byte-based tokenizers. This can select fewer chunks than exact tokenization. System instructions and the bounded question are additional; choose a model with adequate context.
- Answer limit: 800 tokens; OpenAI-compatible request timeout: 60 seconds, up to 2 retries.
- Analytics: off; opt-in retention: 30 days.
- OCR: off. Enabling it requires a separately installed Tesseract/language-data setup supported by PyMuPDF.

For large document collections, indexing currently rebuilds the derived vector collection after changes. Normal searches only check revision/count; they do not fetch all chunk texts to synchronize. Background incremental indexing is future scalability work.

## Feedback and regular performance reviews

Set `SEARCH_ANALYTICS_ENABLED=true` only if you want local feedback persistence and metrics. Restart the app. Feedback is attributed to documents actually cited by a validated answer, not every search candidate. Search-only results receive no document preference credit.

Stored fields: random browser-session identifier, intent, duration, cited document IDs, outcome, rating and timestamp. **Question text, rewritten questions, document text and API keys are not stored in analytics.** Retention cleanup runs when analytics is used. Clearing the chat deletes that session's analytics and starts a fresh session.

Generate a report:

```powershell
.\.venv\Scripts\python.exe -m scripts.analytics_report
```

To make reviews regular, Windows Task Scheduler can run the environment's Python with arguments `-m scripts.analytics_report` and the inner project folder as “Start in.” No operating-system scheduled task is installed automatically. Review low feedback, no-evidence rate, errors and latency, then add failing questions to the evaluation set before changing ranking.

Analytics failure does not block answers. Browser sessions do not provide durable user accounts; personalized feedback is deliberately session-scoped.

## Evaluation and verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.evaluate
```

The evaluation prints recall@3 and reciprocal rank against three synthetic questions with expected document/page labels, plus an unanswerable query. It evaluates **keyword/retrieval plumbing only**, not real semantic-model quality or generated-answer correctness. Unit tests use fake embeddings/chat responses; one regression exercises a real Chroma index.

An optional real-model check is available with `python -m scripts.smoke_local`. It may download the embedding model, uses only a temporary synthetic plan, and never calls a paid chat API.

CI runs the tests and evaluation on Python 3.11/3.12. Before a real demonstration, test with your actual documents and chosen OpenRouter model, inspect every citation, try an unanswered question and confirm conflicting/multiple plans are handled safely.

## Existing data and migration

This upgrade uses a new v2 SQLite namespace and a `v2` subfolder for Chroma. **Re-upload documents through the new interface.** Existing legacy files/indexes are left untouched, not silently migrated or deleted.

The current store links original bytes, chunks and document identities. Replacing/deleting a document updates SQLite and FTS together; vector failure leaves a repairable derived index. Use **Repair vectors**. Search can fall back to keyword matching with a warning. A change in embedding model rebuilds the current derived collection so incompatible dimensions are not mixed.

Back up the local data directory before migrations. Deleting from the app is not a forensic secure erase of backups, legacy databases or disk free space.

## Privacy and deployment boundaries

- The server binds to localhost. This is a **single-owner installation**: all browser sessions on it share the knowledge base. Plan selection is not access control.
- Do not expose it publicly without authentication, per-user storage/authorization, TLS, rate limits and durable backups. `app.yaml` is a non-secret example, not production deployment certification.
- Local embeddings keep embedding text on your machine. OpenAI embeddings send text to OpenAI. Chat sends the question, relevant follow-up context and selected snippets to the configured provider; OpenRouter may route these to its underlying model provider.
- Do not upload private medical/personal information without the necessary permissions and provider agreements.
- Logs contain controlled event/status/error-class fields, not raw provider exceptions or user questions. Third-party libraries may have their own logging; do not enable debug tracing on sensitive data.
- `.env`, local databases, documents, environments and cache files are ignored for future Git additions. Ignore rules do not remove files already committed or secrets in Git history.
- Never push keys. Rotate any credential previously published.

## Known limits

Citations are checked for valid references, **not proven entailment** of each claim. Prompt-injection instructions are treated as untrusted data, but prompting is not a complete security boundary. PDF table/OCR extraction needs human review; complex layouts and scanned documents may be incomplete. Cross-page passages are not automatically merged, preserving page attribution. There is no web crawler, enterprise connector, trained reranker, durable user profile or production tenant isolation.

OpenRouter/OpenAI are the supported chat setup paths here. The existing Databricks adapter is lazy/optional and requires a separately compatible `langchain-databricks` environment and workspace authentication; it is not part of the tested standard installation.

## Troubleshooting

- **No module named …**: use `.venv\Scripts\python.exe` for both installation and launch; avoid mixing Jupyter's Anaconda kernel with this environment.
- **Python not found / Microsoft Store message**: install Python 3.11/3.12, then create the environment with the Python launcher. Do not keep retrying the broken alias.
- **No documents after upgrade**: re-upload into the v2 store.
- **No answer but search results**: chat is set to `none`.
- **Could not produce a verified answer**: check key/model/credits/network; the model may have returned malformed JSON or invalid citations.
- **Keyword-only warning**: check local model download/dependencies, then repair vectors.
- **No readable PDF text**: use a text-based PDF or configure OCR. Password-protected PDFs must be unlocked before upload.
- **Jupyter**: use a terminal for Streamlit. Notebook test output does not launch a web application.
