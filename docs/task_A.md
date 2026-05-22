# Task A — User Modeling: Design Decisions & Findings

> Living document. Update whenever an architectural decision, evaluation result, or ablation
> finding is made. Do not add bug fixes, CI changes, or refactors that don't change the approach.

---

## Problem Statement

Given a user's review history and a target business they have not reviewed, simulate:
1. The **star rating** they would give (1–5, half-star precision)
2. The **written review** they would leave, matching their tone and behavioural style

---

## Architecture

Five-node LangGraph pipeline with a reflection loop:

```
analyst → retriever → reasoner → drafter → critic ─┐
                                           ↑         │ approved or
                                           └─────────┘ max revisions reached → END
```

### analyst
Reads user metadata (review count, average stars, elite status, fan count) and samples up to 15
of their past reviews from the `yelp_reviews` ChromaDB collection. Builds a **Persona Manifesto**
— a 4–6 sentence third-person profile covering tone, rating bias, cuisine preferences, ambiance
preferences, and inferred deal-breakers. Uses a warm-start prompt when reviews exist and a
cold-start prompt (more cautious, general) when the user has no indexed history.

### retriever
Embeds the target business attributes (name, categories, attributes) and runs a similarity search
against `yelp_reviews` to find reviews of similar businesses by other users. Returns up to 5
reference reviews to give the reasoner grounding in what real customers say about this type of
business.

### reasoner
Takes the persona manifesto, retrieved reference reviews, and target business metadata. Uses
structured output (Pydantic model) to produce:
- `predicted_rating` (float, 1–5)
- `reasoning_log` (chain-of-thought explaining the rating decision)

The chain-of-thought explicitly checks for attribute collision (e.g. user hates noisy bars →
target is a noisy bar → predict low) and cross-domain generalisation (e.g. user loves Indian
food → target is Pakistani → likely positive).

### drafter
Takes the persona manifesto, predicted rating, reasoning log, and reference reviews. Generates
the written review in the user's voice — matching their vocabulary, sentence length, tone
(enthusiastic vs critical vs neutral), and typical structural patterns. On revision passes, the
prompt is extended with the previous draft and the critic's specific feedback, instructing the
model to rewrite rather than start from scratch.

### critic
Evaluates the draft review against four criteria using structured output (`is_approved: bool`,
`feedback: str`):
1. **Behavioral fidelity** — does the review sound like this user (tone, vocabulary, rating bias)?
2. **Hallucination** — does the review invent specific details (dishes, features) not grounded in
   the reasoning trace or business categories?
3. **Generic AI-isms** — does the review end with hollow conclusions ("Overall, a great
   experience!") or use robotic, impersonal language?
4. **Target match** — does the review accurately describe the target business, not some other
   business?

If `is_approved` is false, the graph routes back to the drafter with the feedback attached. If
`is_approved` is true, or `revision_count > MAX_REVISIONS` (default: 2), the graph exits.

---

## Key Design Decisions

### Structured output for rating prediction
The reasoner uses `with_structured_output(ReasonerOutput)` (Pydantic) rather than parsing free
text. This eliminates rating extraction errors and forces the LLM to commit to a number before
writing the review, preventing the drafter from contradicting the rating.

### Separation of reasoning and drafting
An earlier design had the LLM produce rating and review text in one call. Splitting them into
reasoner + drafter improved consistency: the drafter receives an explicit `predicted_rating` and
`reasoning_log`, so it writes a review that actually matches the star score. Combined calls
frequently produced 4★ reviews with 2★ text or vice versa.

### Persona Manifesto as intermediate representation
Rather than passing raw review history directly to the reasoner and drafter, the analyst
compresses it into a manifesto. This:
- Stays within context limits regardless of how many reviews the user has
- Forces explicit preference extraction (the LLM must articulate what the user values)
- Makes the reasoning step more focused — the LLM reasons against a profile, not raw text

### ChromaDB review sampling strategy
The analyst samples up to 15 reviews with stratified sampling: at least 1 low-rated (≤3★) and
at least 1 high-rated (>3★). This prevents the manifesto from being skewed by a run of
unusually good or bad experiences.

### Reflection/critic loop
Early drafts frequently exhibited two failure modes: (1) generic closing sentences that sounded
like boilerplate AI output regardless of the user's actual writing style, and (2) hallucinated
specifics (dish names, features) that were not grounded in the reasoning trace. Adding a critic
node that evaluates the draft before committing addresses both without requiring prompt
engineering on the drafter alone.

The loop is bounded by `MAX_REVISIONS` (default 2, configurable via env var) to cap latency. At
the limit, the best draft produced so far is returned regardless of critic verdict. The trade-off
is up to `MAX_REVISIONS` additional LLM calls per request; the quality benefit has not yet been
quantified against the baseline (see Known Weaknesses).

### Nigerian Contextualization (Bonus)
An optional `nigerian_mode` flag is supported across the pipeline. When enabled, the analyst frames the persona manifesto using Nigerian archetypes (e.g., 'Lagos Foodie', 'Mainland Hustler'), and the drafter injects Nigerian English phrasing and Pidgin constructs (e.g., 'abeg', 'jara') into the review. This satisfies the hackathon bonus criterion for culturally localized persona simulation.

---

## Data Pipeline

### Old approach (baseline)
- First 30,000 rows of `yelp_academic_dataset_review.json` (not random — biased toward a
  specific date cluster)
- Random row split across all users → many test users had 0–3 reviews in the index → cold-start
  even for prolific users
- Prolific user filter: ≥ 3 reviews

### New approach (current)
- Random sampling with `random.seed(42)`: 500K reviews, 300K users, 100K businesses
- **Per-user holdout split**: each user's most recent review held out as the test case; all
  earlier reviews go into train — guarantees every test user has indexed history
- Prolific user filter: ≥ 7 reviews in the sample

**Dataset stats (current):**
| Metric | Value |
|---|---|
| Train rows | 4,569 |
| Test rows (unique users) | 429 |
| Avg train reviews per user | 10.7 |
| Users with 0 train reviews | 0 |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| RMSE | Root mean squared error on predicted vs actual star rating |
| MAE | Mean absolute error on predicted vs actual star rating |
| ROUGE-L | F1 lexical overlap between generated and actual review text |

CLI: `uv run python -m src.main --n <N> --output results/output.csv`

---

## Results

### Baseline run (old dataset, n=20)
| Metric | Score |
|---|---|
| RMSE | 1.3115 |
| MAE | 1.0800 |
| ROUGE-L | 0.1159 |

**Key observation:** The agent systematically underpredicts high ratings. Multiple 5★ actual
reviews were predicted at 2.5–3.5★. The pattern suggests the persona manifesto is building
cautious profiles (not enough positive signal in the limited review history under the old dataset)
and the reasoner defaults to middling predictions when uncertain.

### Final run (new dataset)
| Metric | Score |
|---|---|
| RMSE | [TBD] |
| MAE | [TBD] |
| ROUGE-L | [TBD] |

---

## Ablations & Things That Did Not Work

- **Combined rating + review in one LLM call**: Frequent score-text mismatches. Abandoned in
  favour of the two-stage reasoner → drafter split.
- **Passing raw reviews directly to reasoner**: Context window issues for prolific users;
  manifesto compression solved this cleanly.
- **Critic loop quality gain (unquantified)**: The critic loop was added to address generic
  drafts and hallucinated details, but no controlled A/B eval (with vs without critic) has been
  run yet. Effect on RMSE/MAE/ROUGE-L is unknown.

---

## Known Weaknesses & Future Work

- **5★ underprediction bias**: The agent skews to 3–4★ even for enthusiastic users. Likely
  fixable by tuning the reasoner prompt to weight positive signals from the manifesto more
  aggressively.
- **ROUGE-L ceiling**: Lexical overlap with ground-truth reviews is inherently limited because
  the model generates plausible text, not the exact words the user would write. BERTScore
  (semantic similarity) would be a fairer measure of text quality.
- **Model**: Currently using Gemini 2.5 Flash Lite. Upgrading to a stronger model for the
  reasoner node specifically would likely improve rating calibration.
- **Nigerian context**: While the `nigerian_mode` flag forces the agent to adopt a Nigerian voice, the
  underlying Yelp dataset remains US-centric. Adapting the system fully to Nigerian businesses would
  require a Nigeria-specific dataset to prevent a mismatch between the cultural tone and the
  actual business attributes.
- **Critic loop latency**: Each revision cycle adds one critic call + one drafter call. With
  `MAX_REVISIONS=2` this means up to 4 extra LLM calls in the worst case. Latency impact has not
  been measured; may matter if the endpoint is used in a real-time context.
