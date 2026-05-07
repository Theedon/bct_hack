"""One-time script to embed yelp_agent_data.csv into a persistent Chroma index."""

import sys
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.settings import settings
from src.core.vectorstore import COLLECTION_NAME

CSV_PATH = "data/yelp_agent_data.csv"
BATCH_SIZE = 100


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
    total = len(df)
    print(f"  {total} rows loaded.")

    docs = build_documents(df)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY.get_secret_value(),
    )

    batches = [docs[i : i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    print(f"Embedding {total} documents in {len(batches)} batches of {BATCH_SIZE}...")

    vectorstore = None
    with tqdm(total=total, unit="doc", desc="Indexing") as bar:
        for i, batch in enumerate(batches):
            if i == 0:
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    collection_name=COLLECTION_NAME,
                    persist_directory=settings.CHROMA_PATH,
                )
            else:
                vectorstore.add_documents(batch)
            bar.update(len(batch))

    print(f"\nDone. {total} documents persisted to '{settings.CHROMA_PATH}'.")


if __name__ == "__main__":
    main()
