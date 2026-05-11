"""CLI to evaluate the review-generation agent against held-out test examples."""

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd

from src.agent.graph import graph

TEST_CSV = "data/yelp_review/test.csv"

OUTPUT_COLUMNS = [
    # Original test row
    "user_id", "user_name", "user_review_count", "average_stars",
    "user_elite_count", "user_fans", "business_id", "biz_name",
    "biz_stars", "categories", "biz_attributes_clean", "stars_review",
    "date",
    # Ground truth (held out from agent)
    "actual_review",
    # Agent outputs
    "predicted_rating", "draft_review",
    "user_manifesto", "reasoning_log", "new_experience", "retrieved_reviews",
]


def _run_agent(row: dict) -> dict:
    return graph.invoke({
        "messages": [],
        "user_id": str(row["user_id"]),
        "user_name": str(row["user_name"]),
        "user_review_count": int(row["user_review_count"]),
        "average_stars": float(row["average_stars"]),
        "user_elite_count": int(row["user_elite_count"]),
        "user_fans": int(row["user_fans"]),
        "business_id": str(row["business_id"]),
        "biz_name": str(row["biz_name"]),
        "categories": str(row["categories"]),
        "biz_attributes_clean": str(row["biz_attributes_clean"]),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the review agent on test examples.")
    parser.add_argument("--n", type=int, default=5, help="Number of test examples to run (default: 5)")
    parser.add_argument("--output", type=str, default="results/output.csv", help="Output CSV path")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between runs (default: 1.0)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TEST_CSV)
    sample = df.head(args.n)
    total = len(sample)

    print(f"Running agent on {total} test examples → {output_path}")
    print(f"Delay between runs: {args.delay}s\n")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for i, (_, row) in enumerate(sample.iterrows(), start=1):
            print(f"[{i}/{total}] {row['user_name']} → {row['biz_name']} ...", end=" ", flush=True)
            try:
                state = _run_agent(row)
                writer.writerow({
                    "user_id": row["user_id"],
                    "user_name": row["user_name"],
                    "user_review_count": row["user_review_count"],
                    "average_stars": row["average_stars"],
                    "user_elite_count": row["user_elite_count"],
                    "user_fans": row["user_fans"],
                    "business_id": row["business_id"],
                    "biz_name": row["biz_name"],
                    "biz_stars": row["biz_stars"],
                    "categories": row["categories"],
                    "biz_attributes_clean": row["biz_attributes_clean"],
                    "stars_review": row["stars_review"],
                    "date": row["date"],
                    "actual_review": row["text"],
                    "predicted_rating": state["predicted_rating"],
                    "draft_review": state["draft_review"],
                    "user_manifesto": state["user_manifesto"],
                    "reasoning_log": state["reasoning_log"],
                    "new_experience": state["new_experience"],
                    "retrieved_reviews": json.dumps(state["retrieved_reviews"]),
                })
                f.flush()
                print(f"done (predicted: {state['predicted_rating']}★  actual: {row['stars_review']}★)")
            except Exception as e:
                print(f"ERROR: {e}")
                writer.writerow({
                    "user_id": row["user_id"],
                    "user_name": row["user_name"],
                    "user_review_count": row["user_review_count"],
                    "average_stars": row["average_stars"],
                    "user_elite_count": row["user_elite_count"],
                    "user_fans": row["user_fans"],
                    "business_id": row["business_id"],
                    "biz_name": row["biz_name"],
                    "biz_stars": row["biz_stars"],
                    "categories": row["categories"],
                    "biz_attributes_clean": row["biz_attributes_clean"],
                    "stars_review": row["stars_review"],
                    "date": row["date"],
                    "actual_review": row["text"],
                    "predicted_rating": "",
                    "draft_review": f"ERROR: {e}",
                    "user_manifesto": "",
                    "reasoning_log": "",
                    "new_experience": "",
                    "retrieved_reviews": "[]",
                })
                f.flush()

            if i < total and args.delay > 0:
                time.sleep(args.delay)

    print(f"\nDone. Results saved to {output_path}")


if __name__ == "__main__":
    main()
