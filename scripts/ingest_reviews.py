"""One-time script to embed data/yelp_review/train.csv into a persistent Chroma index."""

import sys
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.embeddings import GeminiEmbeddings
from src.core.settings import settings
from src.core.vectorstore import COLLECTION_NAME

CSV_PATH = "data/yelp_review/train.csv"
BATCH_SIZE = 100


def build_documents(df: pd.DataFrame) -> list[Document]:
    docs = []
    for _, row in df.iterrows():
        categories = str(row.get("categories", ""))
        biz_attributes_clean = str(row.get("biz_attributes_clean", ""))
        stars_review = float(row.get("stars_review", 0))
        text = str(row["text"])
        docs.append(
            Document(
                page_content=(
                    f"Category: {categories} | "
                    f"Attributes: {biz_attributes_clean[:150]} | "
                    f"Sentiment: {stars_review} stars | "
                    f"Review: {text[:200]}"
                ),
                metadata={
                    "user_id": str(row.get("user_id", "")),
                    "user_name": str(row.get("user_name", "")),
                    "user_review_count": int(row.get("user_review_count", 0)),
                    "average_stars": float(row.get("average_stars", 0)),
                    "user_elite_count": int(row.get("user_elite_count", 0)),
                    "user_fans": int(row.get("user_fans", 0)),
                    "business_id": str(row.get("business_id", "")),
                    "biz_name": str(row.get("biz_name", "")),
                    "biz_stars": float(row.get("biz_stars", 0)),
                    "categories": categories,
                    "biz_attributes_clean": biz_attributes_clean,
                    "stars_review": stars_review,
                    "date": str(row.get("date", "")),
                    "review_text": text,
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

    embeddings = GeminiEmbeddings()

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
