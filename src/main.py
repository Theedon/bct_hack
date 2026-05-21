"""CLI to evaluate the review-generation agent against held-out test examples."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

from rouge_score import rouge_scorer

import pandas as pd

from src.agent.reviewer.graph import graph

TEST_CSV = "data/yelp_review/test.csv"

OUTPUT_COLUMNS = [
    # Original test row
    "user_id",
    "user_name",
    "user_review_count",
    "average_stars",
    "user_elite_count",
    "user_fans",
    "business_id",
    "biz_name",
    "biz_stars",
    "categories",
    "biz_attributes_clean",
    "stars_review",
    "date",
    # Ground truth (held out from agent)
    "actual_review",
    # Agent outputs
    "predicted_rating",
    "draft_review",
    "user_manifesto",
    "reasoning_log",
    "new_experience",
    "retrieved_reviews",
]


def _run_agent(row: dict) -> dict:
    inputs = {
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
    }

    final_state = inputs.copy()

    print()  # Add a newline so logs start below the header

    for output in graph.stream(inputs):  # type: ignore
        for node_name, state_update in output.items():
            print(f"  → [{node_name}]", flush=True)
            for key, value in state_update.items():
                if key not in inputs and value:
                    if isinstance(value, str):
                        preview = value.strip().replace("\n", " ")
                        if len(preview) > 100:
                            preview = preview[:100] + "..."
                        print(f"      {key}: {preview}")
                    elif isinstance(value, list):
                        print(f"      {key}: {len(value)} items")
                    else:
                        print(f"      {key}: {value}")
            final_state.update(state_update)

    return final_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the review agent on test examples."
    )
    parser.add_argument(
        "--n", type=int, default=5, help="Number of test examples to run (default: 5)"
    )
    parser.add_argument(
        "--output", type=str, default="results/output.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between runs (default: 1.0)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TEST_CSV)
    sample = df.head(args.n)
    total = len(sample)

    print(f"Running agent on {total} test examples → {output_path}")
    print(f"Delay between runs: {args.delay}s\n")

    rows_written: list[dict] = []

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for i, (_, row) in enumerate(sample.iterrows(), start=1):
            print(
                f"[{i}/{total}] {row['user_name']} → {row['biz_name']} ...",
                flush=True,
            )
            try:
                state = _run_agent(row)
                out_row = {
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
                }
                writer.writerow(out_row)
                rows_written.append(out_row)
                f.flush()
                print(
                    f"done (predicted: {state['predicted_rating']}★  actual: {row['stars_review']}★)"
                )
            except Exception as e:
                print(f"ERROR: {e}")
                error_row = {
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
                }
                writer.writerow(error_row)
                rows_written.append(error_row)
                f.flush()

            if i < total and args.delay > 0:
                time.sleep(args.delay)

    print(f"\nDone. Results saved to {output_path}")

    # Evaluation summary
    valid = [r for r in rows_written if r["predicted_rating"] != ""]
    if valid:
        predicted = [float(r["predicted_rating"]) for r in valid]
        actual = [float(r["stars_review"]) for r in valid]
        n = len(valid)
        rmse = math.sqrt(sum((p - a) ** 2 for p, a in zip(predicted, actual)) / n)
        mae = sum(abs(p - a) for p, a in zip(predicted, actual)) / n

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = [
            scorer.score(r["actual_review"], r["draft_review"])["rougeL"].fmeasure
            for r in valid
            if r["draft_review"] and not str(r["draft_review"]).startswith("ERROR")
        ]
        mean_rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

        print(f"\n{'─' * 42}")
        print(f"Rating  RMSE    : {rmse:.4f}  (n={n})")
        print(f"Rating  MAE     : {mae:.4f}")
        print(f"Text    ROUGE-L : {mean_rouge_l:.4f}")
        print(f"{'─' * 42}")


if __name__ == "__main__":
    main()
