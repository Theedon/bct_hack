import os

from langchain_chroma import Chroma

from src.core.embeddings import GeminiEmbeddings
from src.core.settings import settings

COLLECTION_NAME = "yelp_reviews"


def get_vectorstore() -> Chroma:
    if not os.path.exists(settings.CHROMA_PATH):
        raise FileNotFoundError(
            f"Chroma index not found at '{settings.CHROMA_PATH}'. "
            "Run `uv run python scripts/ingest.py` first."
        )
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=settings.CHROMA_PATH,
        embedding_function=GeminiEmbeddings(),
    )
