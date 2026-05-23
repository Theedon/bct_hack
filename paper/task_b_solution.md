---
title: "Task B — Recommendation: Persona-Driven Retrieval with Conversational Refinement"
subtitle: "DSN × BCT LLM Agent Challenge — Solution Paper"
author: "Oluwatoyin Faniyan, Alimi David"
---

# 1. Introduction

Most recommendation systems treat users as static feature vectors — a list of past ratings
and interactions fed into a collaborative filtering model. Task B of this challenge asks for
something different: an agent that delivers personalised recommendations by reasoning about
who the user is, what they want, and how their preferences evolve across a conversation. The
system must handle cold-start users (no history), warm-start users (rich history), and
multi-turn refinement (the user narrows their request over successive turns).

Our system is a three-node LangGraph agent pipeline. A profiler builds a *Preference
Manifesto* from the user's review history — a concise description of what they value. A
candidate node retrieves semantically similar businesses from a ChromaDB vectorstore. A
ranker uses LLM-driven structured output to re-rank candidates by fit, producing a score and
natural-language rationale for each recommendation.

**Dataset.** We use the Kaggle Yelp Academic Dataset (6,990,280 reviews). **Frameworks and
models.** LangGraph for agent orchestration, ChromaDB for vector storage, Gemini 2.5 Flash
Lite (`gemini-2.5-flash-lite`) as the LLM, and Gemini Embedding 2
(`models/gemini-embedding-2`) for embeddings via the Google GenAI API.

# 2. Dataset and Preprocessing

From the full Yelp dataset we randomly sampled 500,000 reviews, 300,000 users, and 100,000
businesses (`random.seed(42)` for reproducibility). After merging on `user_id` and
`business_id` (inner joins), we filtered for prolific users with at least 7 reviews.

**Train/test split.** We use a per-user holdout: each user's most recent review is held out
as the test case. This guarantees every test user has indexed history for the profiler to work
with.

| Metric | Value |
|---|---|
| Train rows | 4,569 |
| Test rows (unique users) | 429 |
| Avg train reviews per user | 10.7 |
| Users with 0 train reviews | 0 |

**Business vectorstore.** The `yelp_businesses` collection is built from all unique
businesses in `train.csv` plus test-only businesses from `test.csv`. For test-only
businesses, the `text` and `stars_review` columns are nulled before ingestion to prevent
data leakage — held-out review content must not appear in the vectorstore embeddings or
aggregated metadata. This brings the total business pool from 3,997 to 4,350.

# 3. Architecture

```
profiler → candidate → ranker → END
```

**Profiler.** Fetches all of the user's indexed reviews from `yelp_reviews` using a metadata
filter (`where={"user_id": user_id}`) and builds a Preference Manifesto — a 4–6 sentence
summary of what the user values: cuisine types, ambiance preferences, price sensitivity,
service expectations, and deal-breakers inferred from low-rated reviews. Reviews are
stratified-sampled (up to 15, at least one low-rated and one high-rated) to prevent bias.

For cold-start users (no indexed reviews), the profiler switches to a cautious prompt that
infers only from demographic signals (review count, elite status, fan count) without
fabricating specific preferences.

If conversation history (`messages`) is present, the profiler appends it to the prompt so the
manifesto reflects the user's latest follow-up request (e.g., "show me something cheaper").

**Candidate.** Constructs a search query from the manifesto and (optionally) the user's
explicit query or last conversational message. Runs a vector similarity search against
`yelp_businesses`, overfetching k=25 candidates. Filters out businesses the user has already
visited (from their review history). Returns up to 20 candidates with full metadata to the
ranker.

**Ranker.** Receives the manifesto, explicit query, and candidate list. Uses Pydantic
structured output to return the top-k businesses ranked by fit score (0.0–1.0), each with a
natural-language rationale explaining why the business matches (or doesn't match) the user's
preferences. Also produces a `reasoning` chain-of-thought describing how it compared
candidates against the manifesto.

## Key Design Decisions

**`candidate_number` not `business_id` in ranker output.** The ranker selects candidates by
their 1-based list index rather than by `business_id`. In early testing, asking the LLM to
return `business_id` strings caused it to hallucinate plausible-looking but non-existent IDs.
Returning an index into the provided list is unambiguous and verifiable.

**Overfetch then filter.** The candidate node fetches k=25 from the vectorstore but only
passes up to 20 to the ranker after filtering visited businesses. This ensures the ranker
always has a meaningful candidate pool even when the user has visited several of the top
vector search hits.

**Client-side message history for multi-turn.** Conversation state is held by the client,
which sends the full `messages` array with each request. The server remains stateless — no
session storage, no database. This maps naturally to LLM message arrays and keeps deployment
simple. When both an explicit `query` field and `messages` are provided, the explicit query
takes priority for vector search — it represents a deliberate override.

**Preference Manifesto as a shared abstraction.** The manifesto concept appears in both
Task A (as a Persona Manifesto for writing style) and Task B (as a Preference Manifesto for
recommendation). In Task B, the manifesto serves as both the user profile for the ranker and
the basis for constructing the search query — a single intermediate representation that
eliminates the need for separate feature engineering.

# 4. Cold-Start and Multi-Turn Handling

## Cold-Start

The profiler uses two distinct prompt paths. For warm-start users, the manifesto is grounded
in actual review samples with specific preferences extracted from the data. For cold-start
users (no indexed reviews), the profiler switches to a cautious prompt that:

- Acknowledges the limited signal explicitly
- Infers only from demographic stats (e.g., high review count with Elite status suggests an
  active, discerning reviewer; zero reviews suggests a true newcomer)
- Does *not* fabricate cuisine, ambiance, or price preferences

**Captured cold-start demo.** User Amara — zero reviews, no Elite status, no fans. The system
correctly sets `cold_start: true` and generates a demographic-only manifesto:

> *"Amara appears to be a new user on the platform, with no recorded reviews or influence
> among other users. Her non-Elite status and zero average star rating suggest she is either
> just beginning her Yelp journey or has not yet engaged with the review system. As such, her
> preferences are currently undefined, offering no discernible pattern for analysis at this
> time."*

The resulting recommendations fall back to popularity- and quality-based ranking:

| # | Business | Location | Categories | Score |
|---|---|---|---|---|
| 1 | Bardea Food & Drink | Wilmington, DE | Bars, Italian | 0.90 |
| 2 | BellaBrava | St. Petersburg, FL | Pizza, Italian | 0.85 |
| 3 | Mike's Ice Cream | Nashville, TN | Ice Cream, Sandwiches | 0.80 |
| 4 | Aroma Mediterranean | King of Prussia, PA | Mediterranean | 0.75 |
| 5 | Nam Phuong | Philadelphia, PA | Seafood, Vietnamese | 0.70 |

Rationales reference objective business attributes (Yelp star rating, amenities, suitability
for groups) rather than fabricated user preferences — exactly the correct behaviour.

## Multi-Turn Conversational Refinement

Multi-turn affects two nodes. The profiler appends the full conversation history to the
prompt so the manifesto evolves with each turn — if the user says "actually I want something
with seafood", the manifesto shifts to reflect that. The candidate node uses the last user
message as the effective search query when no explicit `query` field is provided.

A typical three-turn session:

1. **Turn 1** — general request, no query. The profiler builds a manifesto from review history
   alone. Recommendations reflect the user's overall preferences.
2. **Turn 2** — user says "actually I want something with seafood." The manifesto incorporates
   the refinement; the candidate node searches for seafood-related businesses.
3. **Turn 3** — user adds "with outdoor seating if possible." The manifesto and search query
   narrow further.

The server remains stateless throughout — the client accumulates the `messages` array and
re-sends it with each request.

## Cross-Domain

The manifesto captures *general* preference dimensions — price sensitivity, ambiance
preferences, service expectations — not just category-specific tastes. When a user who
primarily reviews Italian restaurants asks for "something different", the manifesto's general
traits (e.g., preference for casual atmosphere and moderate prices) guide the candidate search
across categories, naturally surfacing cross-domain matches like casual Mexican or Thai
restaurants that share the same attribute profile.

# 5. Experiments and Results

We evaluated on 304 test users (n=304, k=10) using binary NDCG@10, Hit@10, and Liked
Hit@10.

| Metric | Overall | Warm-start | Cold-start |
|---|---|---|---|
| Hit@10 | 0.3% (304 users) | 0.3% (304 users) | n/a |
| Liked Hit@10 | 0.3% (304 users) | 0.3% (304 users) | n/a |
| NDCG@10 | 0.0044 (225 users) | 0.0044 (225 users) | n/a |

79 of 304 users were excluded from NDCG because their held-out review was below 4 stars (no
"liked" ground truth). All test users are warm-start due to the per-user holdout split.

## Why NDCG@10 Is Structurally Low

**The measurement problem.** Our NDCG evaluation asks: *did the system include the exact
business this user visited next, out of 4,350 candidates, in its top 10?* This is a
next-item prediction task. Content-based semantic ranking is designed for preference
alignment, not next-item prediction — these are different problems.

**What a good content-based recommender actually does.** A user who loves quiet Italian
restaurants gets 10 well-matched Italian restaurants. If their held-out review happened to be
for one specific Italian restaurant they visited that month, all 10 recommendations are
"wrong" by NDCG — even if they are objectively better matches to the user's preferences.

**Random baseline context.** A recommender that picks 10 businesses at random from 4,350
would achieve approximately 0.23% hit rate (10/4,350). Our system achieves 0.3% — above the
random baseline, confirming that semantic similarity provides a real signal even under this
framing.

**What would improve NDCG.** Collaborative filtering ("users who liked X also liked Y")
directly targets next-item prediction by learning co-occurrence patterns. This requires a
denser interaction matrix than our 10.7 average reviews per user and is out of scope for this
competition.

**Where the real signal lives.** The competition rubric awards 45 points for criteria that
directly assess recommendation quality: Cold-Start and Cross-Domain (25 pts) and Contextual
Relevance via human evaluation (20 pts). Both measure whether the system gives *good*
recommendations to *real* users — which is exactly what content-based semantic ranking
optimises for.

# 6. Ablation Studies

**`business_id` hallucination.** Early testing asked the ranker to return `business_id`
strings in its output. The LLM frequently invented plausible-looking but non-existent IDs.
Switching to a 1-based `candidate_number` index into the provided list eliminated this
entirely — the index is unambiguous and verifiable.

**Vectorstore coverage bug.** The initial `ingest_businesses.py` script only read
`train.csv`, leaving 83.3% of test businesses (353 of 424) absent from the index. It was
structurally impossible for the recommender to ever surface them. Fixed by including test-only
businesses with their review content stripped (text and stars\_review set to null) to prevent
data leakage. Business pool grew from 3,997 to 4,350.

**IDCG calculation bug.** An early implementation computed IDCG from the retrieved gains
rather than from the ground-truth relevant count. This inflated NDCG: an agent finding 1 of
10 relevant items scored 1.0 instead of approximately 0.43. Fixed to compute IDCG as the
ideal DCG assuming the top `min(|relevant|, k)` positions are all relevant.

**First-N-rows data slice.** The initial dataset used the first 30,000 rows of the raw
review JSON — biased toward a specific date cluster. Replaced with true random sampling.

**Random row split.** A random train/test split across all rows produced test users with zero
training reviews, creating universal cold-start and meaningless NDCG. Replaced with per-user
holdout split.

# 7. Nigerian Contextualization

An optional `nigerian_mode` flag is supported across the pipeline. When enabled:

- The **profiler** frames the preference manifesto using Nigerian cultural touchpoints — for
  example, affinity for "buka" spots, expectations for "correct" portions, or preference for
  "bougie" island places.
- The **ranker** writes recommendation rationales in Nigerian English.
- For **cold-start** users, the Nigerian context injection is guarded: it adds cultural
  framing to the manifesto without fabricating specific Nigerian cuisine or venue preferences
  that the data does not support.

The underlying Yelp dataset is US-centric, so the cultural voice applies to the generation
style and rationale framing rather than the business data itself. Full deployment in a
Nigerian context would require a Nigeria-specific business dataset.

# 8. Limitations and Future Work

**No collaborative filtering.** The candidate step is purely content-based (vector
similarity). Adding a collaborative signal — "users who liked X also liked Y" — could
significantly improve ranking quality, especially for cold-start users.

**Cold-start untested in automated evaluation.** The per-user holdout split requires at
least 7 reviews, so all test users are warm-start. Cold-start performance is validated only
through the demo scenario, not through automated metrics.

**Thin user history.** With an average of 10.7 training reviews per user, the manifesto has
limited signal. A richer dataset (more reviews per user) would improve profiler quality and
downstream recommendation accuracy.

**Model constraint.** We use Gemini 2.5 Flash Lite throughout. A stronger model for the
ranker specifically would improve the quality of rationales and the precision of fit scores.

**Nigerian context on US data.** While `nigerian_mode` provides a localized tone, the
business data remains US-centric. A Nigerian business dataset (e.g., local restaurant
directories) is needed for real-world deployment.
