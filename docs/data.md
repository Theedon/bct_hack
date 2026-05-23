# Dataset — Creation & Preprocessing

> This document describes how `data/yelp_review/train.csv` and `data/yelp_review/test.csv`
> were produced. It is the authoritative reference for the data pipeline — update it if
> sampling parameters, filter thresholds, or split logic change.

---

## Source

**Kaggle Yelp Academic Dataset**
- URL: `https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset`
- Full dataset: 6,990,280 reviews across 3 JSON files

| File | Contents |
|---|---|
| `yelp_academic_dataset_review.json` | Reviews (review_id, user_id, business_id, stars, text, date) |
| `yelp_academic_dataset_business.json` | Business metadata (name, city, state, categories, attributes, stars) |
| `yelp_academic_dataset_user.json` | User metadata (name, review_count, elite, fans, average_stars) |

Full preprocessing code: `notebooks/Yelp_EDA.ipynb` (run on Google Colab)

---

## Sampling

Random sampling with `random.seed(42)` for reproducibility:

| Source file | Sample size |
|---|---|
| Reviews | 500,000 |
| Users | 300,000 |
| Businesses | 100,000 |

Sampled records were merged: reviews ← users (on `user_id`), then ← businesses (on `business_id`), inner joins only.

---

## Preprocessing

| Step | Detail |
|---|---|
| `biz_attributes_clean` | Business `attributes` dict flattened to `key: value, ...` string — LLM context requires plain text |
| `user_elite_count` | Count of years in the comma-separated `elite` field (e.g. `"2019,2020,2021"` → `3`) |
| Column renames | `stars` → `stars_review`, `name` (user) → `user_name`, `review_count` (user) → `user_review_count`, `fans` → `user_fans`, `name` (biz) → `biz_name`, `stars` (biz) → `biz_stars`, `city` → `biz_city`, `state` → `biz_state` |

---

## Train/Test Split

### Prolific user filter
Only users with **≥ 7 reviews** in the merged sample are included. Below this threshold
the preference manifesto has too little signal and cold-start dominates, making both Task A
and Task B evaluations meaningless.

### Per-user holdout split
For each qualifying user:
- **Test**: the single most recent review (sorted by `date`)
- **Train**: all earlier reviews

This guarantees every test user has indexed training history — no accidental cold-start
contamination in the eval set.

### Why not a random row split
A random split would assign some reviews from prolific users to test while leaving other
reviews from the same user in train — but for users who happen to fall entirely on the test
side of the split, they appear as cold-start users despite having rich history. This was the
baseline approach and produced universally cold-start eval sets. Replaced by the per-user
holdout.

---

## Final Dataset Statistics

| Metric | Value |
|---|---|
| Train rows | 4,569 |
| Test rows (unique users) | 429 |
| Avg train reviews per user | 10.7 |
| Users with 0 train reviews | 0 |

Test set was capped at 1,000 users during generation; after merge and prolific filter,
429 users remained.

---

## Golden Columns (both CSVs)

```
user_id, user_name, user_review_count, average_stars, user_elite_count, user_fans,
business_id, biz_name, biz_stars, biz_city, biz_state, categories,
biz_attributes_clean, stars_review, text, date
```

---

## Vectorstore Ingestion

The CSVs feed two separate ChromaDB collections:

| Collection | Script | Contents |
|---|---|---|
| `yelp_reviews` | `scripts/ingest_reviews.py` | One document per review row from `train.csv` |
| `yelp_businesses` | `scripts/ingest_businesses.py` | One document per unique business from `train.csv` + test-only businesses from `test.csv` (review text/stars stripped to prevent data leakage) |

Both scripts drop and recreate their collection on every run — re-running is safe and
idempotent.
