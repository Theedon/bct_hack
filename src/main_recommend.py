"""CLI to batch-evaluate the recommendation agent against held-out test examples."""

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd

from src.agent.recommender.graph import recommend_graph

TEST_CSV = "data/yelp_review/test.csv"

OUTPUT_COLUMNS = [
    "user_id", "user_name", "user_review_count", "average_stars",
    "user_elite_count", "user_fans",
    "cold_start",
    "num_test_businesses", "num_liked_test_businesses",
    "hit_at_k", "liked_hit_at_k",
    "recommendations",
    "user_manifesto", "reasoning_log",
]


def _run_agent(row: pd.Series, k: int) -> dict:
    return recommend_graph.invoke({  # type: ignore[arg-type]
        "user_id": str(row["user_id"]),
        "user_name": str(row["user_name"]),
        "user_review_count": int(row["user_review_count"]),
        "average_stars": float(row["average_stars"]),
        "user_elite_count": int(row["user_elite_count"]),
        "user_fans": int(row["user_fans"]),
        "query": "",
        "k": k,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-evaluate Task B recommender on test users.")
    parser.add_argument("--n", type=int, default=None, help="Number of users to evaluate (default: all)")
    parser.add_argument("--k", type=int, default=5, help="Recommendations per user (default: 5)")
    parser.add_argument("--output", type=str, default="results/output_recommend.csv", help="Output CSV path")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between users (default: 1.0)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TEST_CSV)
    users = [uid for uid in df["user_id"].unique()]
    if args.n is not None:
        users = users[: args.n]
    total = len(users)

    print(f"Evaluating recommender on {total} users (k={args.k}) → {output_path}")
    print(f"Delay between users: {args.delay}s\n")

    rows_written: list[dict] = []

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for i, user_id in enumerate(users, start=1):
            user_rows = df[df["user_id"] == user_id]
            first = user_rows.iloc[0]

            all_test_biz = set(user_rows["business_id"].astype(str).tolist())
            liked_test_biz = set(
                user_rows[user_rows["stars_review"] >= 4]["business_id"].astype(str).tolist()
            )

            print(
                f"[{i}/{total}] {first['user_name']} "
                f"({len(all_test_biz)} test biz, {len(liked_test_biz)} liked) ...",
                end=" ",
                flush=True,
            )

            try:
                state = _run_agent(first, args.k)
                recs = state.get("recommendations", [])
                rec_ids = {r["business_id"] for r in recs}

                hit = int(bool(rec_ids & all_test_biz))
                liked_hit = int(bool(rec_ids & liked_test_biz))

                row = {
                    "user_id": user_id,
                    "user_name": first["user_name"],
                    "user_review_count": first["user_review_count"],
                    "average_stars": first["average_stars"],
                    "user_elite_count": first["user_elite_count"],
                    "user_fans": first["user_fans"],
                    "cold_start": state.get("cold_start", ""),
                    "num_test_businesses": len(all_test_biz),
                    "num_liked_test_businesses": len(liked_test_biz),
                    "hit_at_k": hit,
                    "liked_hit_at_k": liked_hit,
                    "recommendations": json.dumps(recs),
                    "user_manifesto": state.get("user_manifesto", ""),
                    "reasoning_log": state.get("reasoning_log", ""),
                }
                print(f"done  hit={hit}  liked_hit={liked_hit}  cold={state.get('cold_start')}")
            except Exception as e:
                print(f"ERROR: {e}")
                row = {
                    "user_id": user_id,
                    "user_name": first["user_name"],
                    "user_review_count": first["user_review_count"],
                    "average_stars": first["average_stars"],
                    "user_elite_count": first["user_elite_count"],
                    "user_fans": first["user_fans"],
                    "cold_start": "",
                    "num_test_businesses": len(all_test_biz),
                    "num_liked_test_businesses": len(liked_test_biz),
                    "hit_at_k": "",
                    "liked_hit_at_k": "",
                    "recommendations": "[]",
                    "user_manifesto": f"ERROR: {e}",
                    "reasoning_log": "",
                }

            writer.writerow(row)
            f.flush()
            rows_written.append(row)

            if i < total and args.delay > 0:
                time.sleep(args.delay)

    # Summary
    valid = [r for r in rows_written if r["hit_at_k"] != ""]
    warm = [r for r in valid if r["cold_start"] is False or r["cold_start"] == "False"]
    cold = [r for r in valid if r["cold_start"] is True or r["cold_start"] == "True"]

    def _hit_rate(subset: list[dict], key: str) -> str:
        if not subset:
            return "n/a"
        return f"{sum(int(r[key]) for r in subset) / len(subset):.1%} ({len(subset)} users)"

    print(f"\n{'─' * 50}")
    print(f"hit@{args.k}        overall : {_hit_rate(valid, 'hit_at_k')}")
    print(f"hit@{args.k}        warm    : {_hit_rate(warm, 'hit_at_k')}")
    print(f"hit@{args.k}        cold    : {_hit_rate(cold, 'hit_at_k')}")
    print(f"liked_hit@{args.k}  overall : {_hit_rate(valid, 'liked_hit_at_k')}")
    print(f"liked_hit@{args.k}  warm    : {_hit_rate(warm, 'liked_hit_at_k')}")
    print(f"liked_hit@{args.k}  cold    : {_hit_rate(cold, 'liked_hit_at_k')}")
    print(f"{'─' * 50}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
