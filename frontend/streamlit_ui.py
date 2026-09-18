"""Local single-owner UI. Do not expose this application as a public multi-user service."""
import tempfile
import uuid
from pathlib import Path
import streamlit as st
from rag.rag_pipeline import RAGPipeline
from utils.config import settings

def render_message(message, pipeline):
    with st.chat_message(message["role"]):
        # Plain text avoids rendering document/model-controlled hyperlinks or remote images.
        st.text(message["content"])
        for warning in message.get("warnings", []):
            st.warning(warning)
        if message.get("sources"):
            with st.expander("Evidence and citations"):
                for source in message["sources"]:
                    st.text(f"{source.source_file} — physical page/entry {source.page_number}")
                    st.caption(f"Hybrid ranking score: {source.similarity_score:.4f} (not confidence)")
                    st.text(source.text)
                    payload = pipeline.chroma_manager.document_payload(source.document_id)
                    if payload:
                        st.download_button("Download original", payload, file_name=source.source_file,
                            key="download_" + message["id"] + "_" + source.chunk_id)
        if message.get("search_id"):
            left, right = st.columns(2)
            for column, label, rating in ((left, "Helpful", 1), (right, "Not helpful", -1)):
                if column.button(label, key=message["id"] + str(rating)):
                    saved = pipeline.search_intelligence.record_feedback(
                        message["search_id"], rating, st.session_state.session_id)
                    if saved:
                        message["feedback"] = rating
                    else:
                        st.warning("Feedback could not be saved.")
            if message.get("feedback"):
                st.caption("Feedback saved. You can change your rating.")

def run_app():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
    pipeline = st.session_state.pipeline
    manager = pipeline.chroma_manager
    st.title("Insurance Document Assistant")
    st.caption("Search your documents with meaning + keywords. Verify important coverage details with your insurer.")
    with st.sidebar:
        st.header("Documents")
        st.caption("Local, single-owner workspace. Documents are shared by all browser sessions on this installation.")
        plan = st.text_input("Plan name")
        year = st.text_input("Plan year")
        files = st.file_uploader("PDF, text, Markdown or FAQ JSON", type=["pdf","txt","md","json"],
                                 accept_multiple_files=True)
        records = manager.list_documents()
        labels = {d["id"]: f'{d["source"]} | {d["plan"] or "unlabelled"} | {d["year"] or "no year"} | {d["version"][:8]}' for d in records}
        replacement = st.selectbox("Replace an existing document (optional)", [""] + list(labels),
                                    format_func=lambda x: labels.get(x, "Add as new"))
        if st.button("Index documents", disabled=not files):
            if replacement and len(files) != 1:
                st.error("Select one upload when replacing a document.")
            else:
                with st.spinner("Reading and indexing..."):
                    total, failures = 0, []
                    for uploaded in files:
                        # Each upload gets a separate temporary directory; never use an upload's path.
                        with tempfile.TemporaryDirectory(prefix="rag-upload-") as staging:
                            name = uploaded.name.replace("\\", "/").split("/")[-1]
                            path = Path(staging) / name
                            path.write_bytes(uploaded.getvalue())
                            total += pipeline.ingest_documents([path], plan, year, replacement or None)
                            failures.extend(pipeline.last_ingestion_errors)
                            for warning in pipeline.last_ingestion_warnings:
                                st.warning(warning)
                    if total:
                        st.success(f"Stored {total} chunks. Reselect documents below if needed.")
                    for failure in failures:
                        st.error(f'{failure["source"]}: failed while {failure["stage"]} ({failure["error"]}). Check file format, readable text, size and embedding model availability.')
                    if manager.last_vector_error:
                        st.warning("Vector indexing failed. Saved keyword search remains available; use Repair vectors.")
                records = manager.list_documents()
                labels = {d["id"]: f'{d["source"]} | {d["plan"]} | {d["year"]} | {d["version"][:8]}' for d in records}
        selected = st.multiselect("Documents to search", list(labels), default=list(labels),
                                  format_func=lambda x: labels[x])
        st.caption(f"{len(records)} documents / {manager.document_count()} chunks stored")
        if manager.last_vector_error:
            st.warning("The vector index needs repair. Keyword search can still use the saved documents.")
        if st.button("Repair vectors"):
            if manager.sync_vectors():
                st.success("Vector index ready.")
            else:
                st.error("Vector repair failed. Check embedding dependencies/model access.")
        delete = st.selectbox("Document to remove", [""] + list(labels), format_func=lambda x: labels.get(x, "None"))
        confirm = st.checkbox("Confirm deletion from this application's current document store")
        if st.button("Remove selected document", disabled=not delete or not confirm):
            manager.remove_document(delete)
            st.session_state.messages = []
            st.rerun()
        if st.button("Clear current knowledge base", disabled=not confirm):
            manager.clear_collection()
            st.session_state.messages = []
            pipeline.search_intelligence.forget_session(st.session_state.session_id)
            st.rerun()
        if st.button("Clear this chat and its feedback"):
            pipeline.search_intelligence.forget_session(st.session_state.session_id)
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
        st.caption(f"Chat: {settings.chat_provider} / {settings.active_chat_model}")
        st.caption(f"Embeddings: {settings.embedding_provider} / {settings.active_embedding_model}")
        with st.expander("Search performance"):
            summary = pipeline.search_intelligence.summary()
            if not summary["enabled"]:
                st.caption("Analytics/feedback persistence is off. Enable SEARCH_ANALYTICS_ENABLED in .env to opt in.")
            else:
                st.json(summary)
                st.caption("No question text is stored. Feedback adjusts cited documents within this browser session.")
    for message in st.session_state.messages:
        render_message(message, pipeline)
    question = st.chat_input("Ask about the selected plan", max_chars=4000)
    if question and question.strip():
        history = list(st.session_state.messages)
        st.session_state.messages.append(dict(role="user", content=question, id=str(uuid.uuid4())))
        with st.spinner("Searching and checking citations..."):
            response = pipeline.ask(question, st.session_state.session_id, history, selected)
        st.session_state.messages.append(dict(role="assistant", content=response.answer,
            sources=response.sources, search_id=response.search_id, warnings=response.warnings, id=str(uuid.uuid4())))
        st.rerun()
