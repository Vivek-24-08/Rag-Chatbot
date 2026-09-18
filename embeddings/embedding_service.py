"""Lazy, shared embedding models with explicit token limits."""
from functools import lru_cache
from utils.config import settings

@lru_cache(maxsize=2)
def local_model(name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device="cpu")

class EmbeddingService:
    def __init__(self, config=None):
        self.config = config or settings
        self._client = None

    @property
    def fingerprint(self):
        return self.config.embedding_provider + ":" + self.config.active_embedding_model + ":normalized-v1"

    @property
    def client(self):
        if self._client is None:
            if self.config.embedding_provider == "local":
                try:
                    self._client = local_model(self.config.local_embedding_model)
                except Exception:
                    raise ValueError("Local embedding model could not load. Check internet access for its first download.")
            else:
                self.config.validate()
                from langchain_openai import OpenAIEmbeddings
                self._client = OpenAIEmbeddings(model=self.config.embedding_model,
                                               api_key=self.config.openai_api_key,
                                               request_timeout=self.config.request_timeout, max_retries=2)
        return self._client

    @property
    def token_limit(self):
        return self.client.max_seq_length if self.config.embedding_provider == "local" else 8000

    def token_count(self, text):
        if self.config.embedding_provider == "local":
            return len(self.client.tokenizer.encode(text, add_special_tokens=True))
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))

    def embed_documents(self, texts):
        if not texts:
            return []
        if any(self.token_count(t) > self.token_limit for t in texts):
            raise ValueError("Text exceeds the embedding token limit; shorten the query or chunks.")
        if self.config.embedding_provider == "local":
            return self.client.encode(texts, normalize_embeddings=True).tolist()
        return self.client.embed_documents(texts)

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def get_langchain_embeddings(self):
        return self
