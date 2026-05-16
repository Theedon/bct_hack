"""One-time script to embed unique businesses from train.csv into a Chroma index."""

import sys
import time
from pathlib import Path

import pandas as pd
import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.embeddings import GeminiEmbeddings
from src.core.settings import settings
from src.core.vectorstore import BUSINESS_COLLECTION

CSV_PATH = "data/yelp_review/train.csv"
BATCH_SIZE = 100
MAX_SNIPPETS = 3
SNIPPET_CHARS = 80
ATTRIBUTE_CHARS = 200


def _top_snippets(texts: list[str], n: int = MAX_SNIPPETS) -> list[str]:
    """Pick the n shortest non-empty review texts and truncate each."""
    cleaned = [t.strip().replace("\n", " ") for t in texts if isinstance(t, str) and t.strip()]
    cleaned.sort(key=len)
    return [t[:SNIPPET_CHARS] for t in cleaned[:n]]


def build_documents(df: pd.DataFrame) -> list[Document]:
    docs: list[Document] = []
    for business_id, group in df.groupby("business_id"):
        first = group.iloc[0]
        biz_name = str(first.get("biz_name", ""))
        categories = str(first.get("categories", ""))
        attributes = str(first.get("biz_attributes_clean", ""))[:ATTRIBUTE_CHARS]
        biz_stars = float(first.get("biz_stars", 0))

        review_texts = group["text"].dropna().astype(str).tolist()
        snippets = _top_snippets(review_texts)
        vibe = " // ".join(snippets) if snippets else "n/a"

        review_stars = group["stars_review"].dropna().astype(float)
        avg_user_stars = float(review_stars.mean()) if len(review_stars) else 0.0

        page_content = (
            f"Name: {biz_name} | "
            f"Category: {categories} | "
            f"Attributes: {attributes} | "
            f"Vibe: {vibe}"
        )

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "business_id": str(business_id),
                    "biz_name": biz_name,
                    "categories": categories,
                    "biz_attributes_clean": attributes,
                    "biz_stars": biz_stars,
                    "avg_user_stars": round(avg_user_stars, 2),
                    "review_count": int(len(group)),
                },
            )
        )
    return docs


def main() -> None:
    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} review rows loaded.")

    docs = build_documents(df)
    total = len(docs)
    print(f"  {total} unique businesses.")

    embeddings = GeminiEmbeddings()
    batches = [docs[i : i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    print(f"Embedding {total} documents in {len(batches)} batches of {BATCH_SIZE}...")

    def _with_retry(fn, *args, **kwargs):
        delay = 5
        for attempt in range(6):
            try:
                return fn(*args, **kwargs)
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429 and attempt < 5:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise

    vectorstore: Chroma | None = None
    with tqdm(total=total, unit="doc", desc="Indexing") as bar:
        for i, batch in enumerate(batches):
            if i == 0:
                vectorstore = _with_retry(
                    Chroma.from_documents,
                    documents=batch,
                    embedding=embeddings,
                    collection_name=BUSINESS_COLLECTION,
                    persist_directory=settings.CHROMA_PATH,
                )
            else:
                assert vectorstore is not None
                _with_retry(vectorstore.add_documents, batch)
            bar.update(len(batch))
            time.sleep(1.0)

    print(f"\nDone. {total} documents persisted to '{settings.CHROMA_PATH}'.")


if __name__ == "__main__":
    main()
