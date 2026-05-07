"""One-time script to embed yelp_agent_data.csv into a persistent Chroma index."""

import sys
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.settings import settings
from src.core.vectorstore import COLLECTION_NAME

CSV_PATH = "data/yelp_agent_data.csv"


def build_documents(df: pd.DataFrame) -> list[Document]:
    docs = []
    for _, row in df.iterrows():
        docs.append(
            Document(
                page_content=str(row["text"]),
                metadata={
                    "user_id": str(row.get("user_id", "")),
                    "user_name": str(row.get("user_name", "")),
                    "business_id": str(row.get("business_id", "")),
                    "biz_name": str(row.get("biz_name", "")),
                    "categories": str(row.get("categories", "")),
                    "biz_stars": float(row.get("biz_stars", 0)),
                    "stars_review": float(row.get("stars_review", 0)),
                    "date": str(row.get("date", "")),
                },
            )
        )
    return docs


def main() -> None:
    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} rows loaded.")

    docs = build_documents(df)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )

    print(f"Embedding and persisting to '{settings.CHROMA_PATH}'...")
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=settings.CHROMA_PATH,
    )
    print(f"Done. {len(docs)} documents indexed.")


if __name__ == "__main__":
    main()
