"""Single owner of the ChromaDB client, collection, and embedding model, so
every module's query-time embeddings match ingestion-time embeddings."""

import logging
import os

import chromadb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_memory")
COLLECTION_NAME = "news_sentiment"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_chroma_client = None
_news_collection = None
_embedder = None


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    return _chroma_client


def get_news_collection():
    global _news_collection
    if _news_collection is None:
        client = get_chroma_client()
        _news_collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return _news_collection


def get_embedder():
    """Lazily loads the sentence-transformers model -- not every process
    that imports this module needs the ~90MB model loaded at import time."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logging.info(f"Loading sentence-transformers embedding model: {EMBEDDING_MODEL_NAME} ...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info("Embedding model ready.")
    return _embedder


def embed_texts(texts):
    """Returns a list of dense vector embeddings (list[list[float]]) for the
    given list of strings, using the shared all-MiniLM-L6-v2 model."""
    if not texts:
        return []
    model = get_embedder()
    vectors = model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()


def embed_query(text: str):
    """Embeds a single query string with the exact same model used for
    ingestion -- required so RAG cosine-similarity search is meaningful."""
    return embed_texts([text])[0]
