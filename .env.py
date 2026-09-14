# ==============================================================================
# .env.py
# ------------------------------------------------------------------------------
# HOW TO USE THIS FILE:
#   1. Copy it:      cp .env.example .env
#   2. Open ".env" and paste in your real OpenAI API key.
#   3. NEVER commit the real ".env" file to git — it contains secrets.
#      (This project's .gitignore already excludes ".env" for you.)
#
# The app reads these values at startup using python-dotenv (see
# utils/config.py), so nothing here is hard-coded into the source code.
# ==============================================================================

# Your OpenAI API key. Get one at https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-REPLACE_WITH_YOUR_OWN_KEY

# Chat model used to generate the final answer (GPT-4o = strong reasoning + low latency)
OPENAI_CHAT_MODEL=gpt-4o

# Embedding model used to turn text into vectors for similarity search
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Folder where the Chroma vector database is persisted to disk
CHROMA_PERSIST_DIR=data/chroma_db

# Name of the Chroma "collection" (a named table of vectors) for this project
CHROMA_COLLECTION_NAME=aetna_insurance_docs

# Chunking configuration (see chunking/text_splitter.py for how these are used)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Number of chunks to retrieve per question ("top-k" retrieval)
RETRIEVAL_TOP_K=5

# Hybrid semantic + keyword retrieval and local feedback learning
RETRIEVAL_CANDIDATE_K=20
HYBRID_RRF_K=60
HYBRID_VECTOR_WEIGHT=1.0
HYBRID_LEXICAL_WEIGHT=1.0
SEARCH_ANALYTICS_DB=data/search_analytics.db
SEARCH_ANALYTICS_ENABLED=true
FEEDBACK_SOURCE_BOOST=0.15
LEXICAL_INDEX_PATH=data/lexical_search.db
