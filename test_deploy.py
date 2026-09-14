import streamlit as st

st.set_page_config(page_title="Test", page_icon="✅")
st.title("✅ Aetna RAG Chatbot - Deployment Test")
st.write("If you can see this, the Streamlit container is working!")

# Test basic imports
import sys
st.write(f"Python: {sys.version}")

try:
    from utils.config import settings
    st.success("✅ utils.config imported successfully")
except Exception as e:
    st.error(f"❌ utils.config failed: {e}")

try:
    from langchain_openai import ChatOpenAI
    st.success("✅ langchain_openai imported successfully")
except Exception as e:
    st.error(f"❌ langchain_openai failed: {e}")

try:
    import chromadb
    st.success(f"✅ chromadb {chromadb.__version__} imported successfully")
except Exception as e:
    st.error(f"❌ chromadb failed: {e}")

try:
    from rag.rag_pipeline import RAGPipeline
    st.success("✅ rag.rag_pipeline imported successfully")
except Exception as e:
    st.error(f"❌ rag.rag_pipeline failed: {e}")
