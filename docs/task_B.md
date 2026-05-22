# Task B — Recommendation: Design Decisions & Findings

> Living document. Update whenever an architectural decision, evaluation result, or ablation
> finding is made. Do not add bug fixes, CI changes, or refactors that don't change the approach.

---

## Problem Statement

Given a user's review history and optional natural language query, return a ranked list of
personalised business recommendations the user has not yet visited. Must handle:
- **Warm-start**: user has review history in the index
- **Cold-start**: user is new or has no indexed reviews
- **Multi-turn**: user refines their request across multiple conversational turns

---

## Architecture

Three-node LangGraph pipeline:

```
profiler → candidate → ranker
```

### profiler
Fetches all of the user's indexed reviews from `yelp_reviews` ChromaDB using a metadata filter
(`where={"user_id": user_id}`) — not a similarity search. Builds a **Preference Manifesto**
summarising what the user values: cuisine types, ambiance, price sensitivity, service
expectations, and deal-breakers inferred from low-rated reviews.

Two prompt paths:
- **Warm-start**: 4–6 sentence manifesto grounded in review samples (up to 15, stratified)
- **Cold-start**: 3–4 sentence cautious manifesto based only on demographic signals (review
  count, elite status, fan count)

If conversation history (`messages`) is present, the profiler appends it to the prompt so the
manifesto reflects the user's latest follow-up request (e.g. "show me something cheaper").

### candidate
Constructs a search query from the manifesto and (optionally) the user's explicit query or last
message. Runs a vector similarity search against `yelp_businesses` ChromaDB, overfetching
k=25 candidates. Filters out businesses the user has already visited (from their review history).
Returns up to 20 candidates with metadata.

### ranker
Receives the manifesto, explicit query, and candidate list. Uses structured output (Pydantic
`RankerOutput`) to return the top-k businesses ranked by fit score, each with a rationale.
Produces a `reasoning` chain-of-thought explaining how it compared candidates against the
manifesto and query.

---

## Key Design Decisions

### `candidate_number` not `business_id` in ranker output
The ranker LLM selects candidates by their list index (`[1]`, `[2]`, etc.) rather than by
`business_id`. This prevents hallucination: if the LLM were asked to return a `business_id`, it
would frequently invent plausible-looking IDs that don't exist. Returning an index into the
provided list is unambiguous and verifiable.

### Overfetch then filter
The candidate node fetches k=25 from the vectorstore but only passes ≤20 to the ranker after
filtering visited businesses. This ensures the ranker always has a meaningful candidate pool
even when the user has visited several of the top vector search hits.

### Client-side message history for multi-turn (stateless server)
Conversation state is held by the client, which sends the full `messages` array with each
request. The server remains stateless — no session storage, no database. This maps naturally
to LLM message arrays and keeps the deployment simple. Trade-off: clients must store and
re-send history on every turn.

Multi-turn affects two nodes:
- **profiler**: appends conversation history to the prompt so the manifesto reflects follow-up
  requests
- **candidate**: uses the last user message as the effective vector search query when no
  explicit `query` field is provided (falls back to `query` if both are present)

### Explicit query takes priority over conversation history in candidate search
When both a `query` field and `messages` are provided, the explicit `query` wins for vector
search. History is only used as the search query when `query` is empty. Rationale: if a caller
provides an explicit query, it represents a deliberate override.

### NDCG@10 as primary evaluation metric
The competition brief scores Task B primarily on NDCG@10 (30 pts). Earlier evaluations used
k=5, which deflated NDCG scores because the numerator (DCG over 5 items) was compared against
an ideal of 10 items. Default k changed to 10 throughout.

### IDCG computed from ground truth, not retrieved items
The correct IDCG for binary NDCG@k assumes the top `min(len(relevant_ids), k)` positions are
all relevant — regardless of what the agent actually retrieved. An earlier implementation
computed IDCG from the retrieved gains only, which inflated scores (an agent finding 1 of 10
relevant items scored NDCG=1.0 instead of ~0.43).

### Location in business vectorstore (`biz_city`, `biz_state`)
City and state are embedded into the `yelp_businesses` vectorstore `page_content` so
location-aware queries (e.g. "quiet coffee shop in Philadelphia") match correctly by semantic
similarity. They are also stored in metadata and propagated through candidate → ranker →
API response so clients can display location alongside recommendations.

### Nigerian Contextualization (Bonus)
An optional `nigerian_mode` flag allows the recommendation pipeline to adopt a localized voice. When enabled, the profiler frames the preference manifesto using Nigerian cultural touchpoints (e.g., 'buka' spots), and the ranker uses Nigerian English in its recommendation rationales.

---

## Data Pipeline

### Old approach (baseline)
- First 30,000 reviews — biased toward a specific date/cluster
- Random row split → test users had 0–3 reviews in the index despite being prolific in the
  full dataset
- Prolific filter: ≥ 3 reviews

### New approach (current)
- Random sampling (`random.seed(42)`): 500K reviews, 300K users, 100K businesses
- **Per-user holdout split**: most recent review held out as test; all earlier reviews in train
- Prolific filter: ≥ 7 reviews in the sample

**Dataset stats (current):**
| Metric | Value |
|---|---|
| Train rows | 4,569 |
| Test rows (unique users) | 429 |
| Avg train reviews per user | 10.7 |
| Users with 0 train reviews | 0 |

---

## Evaluation Metrics

| Metric | Description | Competition weight |
|---|---|---|
| NDCG@10 | Binary normalised DCG at 10 — primary ranking quality metric | 30 pts |
| Hit@10 | Whether any recommended business appears in user's held-out test set | — |
| Liked Hit@10 | Whether any recommended business is one the user rated ≥ 4★ | — |

CLI: `uv run python -m src.main_recommend --n <N> --k 10 --output results/output_recommend.csv`

---

## Results

### Final run (new dataset, k=10, n=304 users)
| Metric | Overall | Warm-start | Cold-start |
|---|---|---|---|
| Hit@10 | 0.3% (304 users) | 0.3% (304 users) | n/a |
| Liked Hit@10 | 0.3% (304 users) | 0.3% (304 users) | n/a |
| NDCG@10 | 0.0044 (225 users) | 0.0044 (225 users) | n/a |

*Index built from train.csv + test.csv (4,350 businesses). CLI: `uv run python -m src.main_recommend --n 429 --k 10`.*

**Key observations:**

- **No cold-start users in eval set** — the per-user holdout split requires ≥7 reviews per user,
  so every test user has indexed training history. Cold-start path untested by this evaluation.

- **79/304 users excluded from NDCG** — their held-out review was < 4★, so they have no
  "liked" ground truth. NDCG is only meaningful for the 225 users who liked their held-out
  business.

- **Low hit rate is structurally expected** — the task asks a content-based system to pick the
  one specific business (out of 4,350) a user happened to visit next. A random baseline would
  score ~0.23% hit rate (10/4350); our system at 0.3% is marginally above chance, suggesting
  the semantic similarity is providing a small real signal.

---

## Ablations & Things That Did Not Work

- **`business_id` in ranker output**: LLM hallucinated plausible-looking but non-existent IDs
  in early testing. Replaced with 1-based candidate index (`candidate_number`).
- **Random row split for train/test**: Test users frequently had no training reviews, causing
  universal cold-start and meaningless NDCG. Replaced with per-user holdout split.
- **First-N-rows slice**: Introduced date/cluster bias. Replaced with random sampling.
- **IDCG from retrieved gains**: Inflated NDCG scores. Fixed to use ground-truth relevant count.
- **Business vectorstore built from train.csv only**: 83.3% of test businesses (353/424) were
  not in the index, making it structurally impossible to hit them. Fixed by including test.csv
  businesses in `ingest_businesses.py` (pool: 3,997 → 4,350).

---

## Known Weaknesses & Future Work

- **Thin user history**: With avg 10.7 train reviews per user the manifesto has limited signal.
  A richer dataset (more reviews per user) would improve profiler quality.
- **No collaborative filtering**: The candidate step is purely content-based (vector similarity).
  Adding a collaborative signal (users who liked X also liked Y) could significantly improve
  NDCG, especially in cold-start.
- **Cold-start**: Currently falls back to a demographic-only manifesto. Could be improved with
  popularity-based fallback candidates or asking the user for explicit preferences.
- **Cold-start untested in evaluation**: The per-user holdout split (≥7 reviews required) means
  all test users are warm-start. Cold-start performance is completely unvalidated.
- **Nigerian context**: Although the `nigerian_mode` provides a localized tone and rationale,
  the actual Yelp dataset is US-centric. A Nigerian business dataset (e.g. local restaurant
  directories) is still needed for real-world deployment in Nigeria.
- **Model**: Gemini 2.5 Flash Lite. A stronger model for the ranker specifically would improve
  the quality of rationales and fit scores.
