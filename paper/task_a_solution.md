---
title: "Task A — User Modeling: Simulating Reviews with Persona-Driven LLM Agents"
subtitle: "DSN × BCT LLM Agent Challenge — Solution Paper"
author: "Oluwatoyin Faniyan, Alimi David"
---

# 1. Introduction

Online review platforms contain rich behavioural signals — every rating and review is a window
into a user's preferences, expectations, and decision-making patterns. Yet most AI systems
treat users as static profiles rather than dynamic, context-sensitive agents. Task A of this
challenge asks us to close that gap: given a user's review history and a target business they
have not yet visited, simulate both the star rating they would assign and the written review
they would leave — matching their tone, vocabulary, and rating behaviour.

Our system is a five-node LangGraph agent pipeline that models each user as a *persona* rather
than a feature vector. An analyst node reads the user's past reviews and compresses them into
a Persona Manifesto — a concise behavioural profile. A retriever fetches semantically similar
reviews from other users as grounding context. A reasoner predicts the star rating through
structured chain-of-thought. A drafter ghostwrites the review in the user's voice. Finally,
a critic evaluates the draft against four quality criteria and sends it back for revision if
needed.

**Dataset.** We use the Kaggle Yelp Academic Dataset (6,990,280 reviews). **Frameworks and
models.** LangGraph for agent orchestration, ChromaDB for vector storage, Gemini 2.5 Flash
Lite (`gemini-2.5-flash-lite`) as the LLM, and Gemini Embedding 2
(`models/gemini-embedding-2`) for embeddings via the Google GenAI API.

# 2. Dataset and Preprocessing

From the full Yelp dataset we randomly sampled 500,000 reviews, 300,000 users, and 100,000
businesses (`random.seed(42)` for reproducibility). After merging on `user_id` and
`business_id` (inner joins), we filtered for *prolific users* with at least 7 reviews in the
sample — below this threshold the persona manifesto has too little signal.

**Train/test split.** We use a per-user holdout: for each qualifying user, their most recent
review (by date) is held out as the test case and all earlier reviews go into the training
set. This guarantees every test user has indexed history — no accidental cold-start
contamination. A naive random row split was our baseline; it produced test users with zero
training reviews despite being prolific in the full dataset, making evaluation meaningless.

| Metric | Value |
|---|---|
| Train rows | 4,569 |
| Test rows (unique users) | 429 |
| Avg train reviews per user | 10.7 |
| Users with 0 train reviews | 0 |

Business attributes (a nested JSON dictionary in the raw data) are flattened to a
comma-separated string (`biz_attributes_clean`) for use as plain text in LLM prompts. The
`elite` field is converted from a comma-separated year list to a count
(`user_elite_count`).

# 3. Architecture

```
analyst → retriever → reasoner → drafter → critic ─┐
                                          ↑         │ approved or
                                          └─────────┘ max revisions → END
```

**Analyst.** Reads user metadata (review count, average stars, elite status, fan count) and
samples up to 15 past reviews from the `yelp_reviews` ChromaDB collection using a metadata
filter (`where={"user_id": user_id}`) — not a similarity search. Reviews are stratified: at
least one low-rated (≤3 stars) and one high-rated (>3 stars) to prevent the manifesto from
being skewed by a run of unusually good or bad experiences. The output is a 4–6 sentence
third-person Persona Manifesto describing the user's tone, rating bias, cuisine preferences,
ambiance preferences, and deal-breakers.

**Retriever.** Embeds the target business attributes and runs a similarity search against
`yelp_reviews` to find reviews of similar businesses by other users. Returns up to 5
reference reviews to give the reasoner grounding in what real customers say about this type
of business.

**Reasoner.** Takes the manifesto, reference reviews, and target business metadata. Produces
a `predicted_rating` (float, 1.0–5.0) and a `reasoning_log` (chain-of-thought) via Pydantic
structured output. The reasoning follows a four-step Preference Alignment Analysis:
(1) anchor on the user's historical average star rating as a prior, (2) identify positive
synergies between business attributes and user preferences, (3) identify negative friction
where attributes violate preferences, and (4) derive a final rating from the net effect.

**Drafter.** Receives the manifesto, predicted rating, reasoning log, reference reviews, and
the target business's full attribute dictionary. Generates the written review in the user's
voice — matching vocabulary, sentence length, and structural patterns. Every claim must be
grounded in the provided metadata rather than invented. On revision passes, the prompt
includes the previous draft and the critic's feedback, instructing the model to rewrite
rather than start from scratch.

**Critic.** Evaluates the draft against four criteria using structured output
(`is_approved: bool`, `feedback: str`):

1. *Behavioral fidelity* — does the review sound like this user?
2. *Hallucination* — does it invent specifics not in the reasoning trace or business metadata?
3. *Generic AI-isms* — does it end with hollow conclusions or use robotic language?
4. *Target match* — does it describe the correct business?

If the critic rejects, the graph routes back to the drafter with feedback. The loop exits
on approval or after `MAX_REVISIONS` (default 2) to bound latency.

## Key Design Decisions

**Persona Manifesto as intermediate representation.** Rather than passing raw review history
to downstream nodes, the analyst compresses it into a manifesto. This stays within context
limits regardless of review count, forces explicit preference extraction, and makes the
reasoning step more focused — the LLM reasons against a profile, not raw text.

**Separation of reasoning and drafting.** An earlier design produced rating and review text
in one LLM call. This frequently resulted in score-text mismatches — a 4-star review with
2-star language, or vice versa. Splitting into reasoner (commits to a number first) and
drafter (writes text that matches the number) eliminated this class of error.

**Structured output for rating prediction.** The reasoner uses
`with_structured_output(ReasonerOutput)` rather than parsing free text. This eliminates
rating extraction errors and forces the LLM to commit to a precise number before the drafter
begins.

# 4. Experiments and Results

We evaluated on 50 test users (n=50), measuring RMSE, MAE, and ROUGE-L against
held-out ground-truth reviews.

## Progression Across Fixes

| Metric | Before fixes | + Prompt rewrite | + Explicit avg\_stars |
|---|---|---|---|
| RMSE | 1.3964 | 1.3596 | **1.2215** |
| MAE | 1.1200 | 0.9592 | **0.8920** |
| ROUGE-L | 0.1300 | 0.1367 | **0.1327** |
| Within 0.5 stars | — | 35% | **38%** |
| Exact match | — | — | **30%** |

The *prompt rewrite* replaced the friction-only reasoner prompt with a balanced Preference
Alignment Analysis. The *explicit avg\_stars* fix passed the user's numerical average star
rating directly to the reasoner rather than expecting it to infer this from the manifesto.
This single change was the largest improvement (RMSE −0.14), confirming that baseline
anchoring works best with precise numerical input.

## Per-Star Error Analysis

| Actual Stars | n | Mean Predicted | Mean Signed Error | MAE |
|---|---|---|---|---|
| 1 star | 3 | 3.67 | +2.67 | 2.67 |
| 2 stars | 5 | 4.00 | +2.00 | 2.00 |
| 3 stars | 9 | 3.78 | +0.78 | 0.78 |
| 4 stars | 9 | 3.91 | −0.09 | 0.58 |
| 5 stars | 24 | 4.40 | −0.60 | 0.60 |

The 3–5 star range is well-calibrated (MAE ≤ 0.78), with 4-star predictions nearly perfect
(mean error −0.09). Excluding the 8 low-star outliers, the effective MAE on the remaining 42
cases is approximately 0.62.

**Experience-driven outlier blindness.** Low-star reviews (1–2 stars) are almost universally
driven by one-off bad *experiences* — food poisoning, insects in food, wrong orders — that
cannot be inferred from business attributes or user preferences. The model correctly
identifies that the business type matches the user's preferences but cannot anticipate service
failures. This is a fundamental limitation of attribute-based reasoning.

## Sample Outputs

**Example 1 — Mark (1,014 reviews, avg 3.41 stars) → Tee-Off Restaurant (Steakhouse)**

| | Rating | Review excerpt |
|---|---|---|
| **Actual** | 5.0 stars | *"This actually is the best steak in town! Tee Off is a place like none other in Santa Barbara..."* |
| **Predicted** | 3.2 stars | *"This spot offers a decent atmosphere for a night out, and I must admit the steakhouse aspect is a solid draw..."* |

The model captures Mark's casual, direct tone but underpredicts by 1.8 stars — it anchors
on his 3.41 average and finds no strong positive signal in the business attributes to push
higher. The actual 5-star experience reflects a personal discovery that the system cannot
anticipate from metadata alone.

**Example 2 — Richard (642 reviews, avg 3.3 stars) → Mimi Blue Meatballs (Italian)**

| | Rating | Review excerpt |
|---|---|---|
| **Actual** | 3.0 stars | *"TEN THOUGHTS ABOUT MEATBALLS: 1. The space is fantastic. Very bistro-ish..."* |
| **Predicted** | 3.8 stars | *"This Italian spot had a good menu. The meatballs don't look like much initially, but they were quite delicious..."* |

The prediction is within 0.8 stars. The system captures Richard's matter-of-fact style,
though it misses his distinctive list-based review format.

**Example 3 — J (100 reviews, avg 4.23 stars) → McDonald's (Fast Food)**

| | Rating | Review excerpt |
|---|---|---|
| **Actual** | 1.0 star | *"Think might have gotten food poisoning from here. Ordered something, came out wrong..."* |
| **Predicted** | 4.5 stars | *"A good place for a quick bite. I really enjoy it, great value for the price!"* |

This is the experience-driven outlier problem in action. The model sees a user who averages
4.23 stars visiting a fast-food restaurant — a reasonable attribute match — and predicts high
satisfaction. It cannot anticipate the food poisoning incident that drove the actual 1-star
rating.

# 5. Ablation Studies

**Combined rating and review in one call.** Our initial design produced both the star rating
and review text in a single LLM call. This led to frequent score-text mismatches: 4-star
ratings accompanied by clearly negative review text, or 2-star ratings with enthusiastic
prose. Splitting into a separate reasoner (rating) and drafter (text) eliminated this class
of error entirely.

**Friction-only reasoner prompt.** The original reasoner framed rating prediction as an
"Internal Friction Analysis" — a three-step process focused exclusively on finding attribute
violations. This caused the model to collapse predictions toward 3.5–4.0 stars regardless
of actual rating, producing a mean signed error of +3.0 on 1-star actuals and −1.0 on 5-star
actuals. Replacing with a balanced Preference Alignment Analysis that considers both positive
synergies and negative friction, anchored on the user's average stars, reduced MAE from 1.12
to 0.96.

**Drafter without business attributes.** The drafter prompt demanded specificity about
business features (WiFi, parking, price range) but was never passed `biz_attributes_clean` —
only the reasoner received it. This forced the LLM to hallucinate details to satisfy the
prompt's instructions. Passing the full attribute dictionary and rewriting the prompt rule to
"Ground every detail in provided metadata" reduced hallucinated specifics.

**Random row split for train/test.** A random split across all rows produced test users with
zero training reviews — making them appear as cold-start users despite being prolific in the
full dataset. Replaced with per-user holdout split.

**Critic loop.** Added to address generic draft quality and hallucinated details. The loop
improves qualitative output but its effect on RMSE/MAE/ROUGE-L has not yet been quantified
in a controlled A/B evaluation.

# 6. Nigerian Contextualization

An optional `nigerian_mode` flag is supported across the pipeline. When enabled, the analyst
frames the Persona Manifesto using Nigerian archetypes (e.g., "Lagos Foodie", "Abuja Big
Boy", "Mainland Hustler"), and the drafter injects Nigerian English phrasing and Pidgin
constructs (e.g., "abeg", "jara") into the generated review. This satisfies the bonus
criterion for culturally localized persona simulation. The underlying Yelp dataset remains
US-centric, so the cultural voice applies to the generation style rather than the business
data itself.

# 7. Limitations and Future Work

**Experience-driven outlier blindness** remains the largest error source. Low-star reviews
are driven by unpredictable service failures, not attribute mismatches. Addressing this would
require incorporating real-time signals such as recent review sentiment trends for the target
business.

**5-star residual conservatism.** After fixes, 13 of 24 actual 5-star reviews were predicted
at 4.0 stars. The model is still slightly reluctant to commit to perfect scores. Further
tuning or a stronger model for the reasoner node may help.

**ROUGE-L ceiling.** Lexical overlap with ground-truth reviews is inherently limited because
the model generates plausible text in the user's style, not the exact words they would write.
BERTScore (semantic similarity) would be a fairer measure of text quality but was not
computed in this evaluation.

**Model constraint.** We use Gemini 2.5 Flash Lite throughout. A stronger model for the
reasoner specifically would likely improve rating calibration.

**Critic loop latency.** Each revision adds one critic call plus one drafter call. With
`MAX_REVISIONS=2`, this means up to 4 extra LLM calls in the worst case. The quality benefit
has not been measured against the no-critic baseline in a controlled experiment.
