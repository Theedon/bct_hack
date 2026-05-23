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

The chain-of-thought follows a four-step **Preference Alignment Analysis**: (1) anchor on the
user's historical average star rating as a prior, (2) identify positive synergies between the
target business attributes and the user's known preferences, (3) identify negative friction
where attributes violate preferences, and (4) derive a final rating from the net effect. The
prompt explicitly encourages use of the full 1.0–5.0 scale.

### drafter
Takes the persona manifesto, predicted rating, reasoning log, reference reviews, and **target
business attributes** (categories + full attribute dictionary). Generates the written review in
the user's voice — matching their vocabulary, sentence length, tone (enthusiastic vs critical vs
neutral), and typical structural patterns. The prompt instructs the drafter to ground every
detail in the provided metadata rather than inventing specifics. On revision passes, the prompt
is extended with the previous draft and the critic's specific feedback, instructing the model to
rewrite rather than start from scratch.

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

### Sample run (new dataset, n=50, after prompt fixes)

**Progression across fixes:**
| Metric | Before fixes | + Prompt rewrite | + Explicit avg_stars |
|---|---|---|---|
| RMSE | 1.3964 | 1.3596 | **1.2215** |
| MAE | 1.1200 | 0.9592 | **0.8920** |
| ROUGE-L | 0.1300 | 0.1367 | **0.1327** |
| Within 0.5★ | — | 35% | **38%** |
| Exact match | — | — | **30%** |

**Per-star error analysis (latest):**
| Actual Stars | n | Mean Predicted | Mean Signed Error | MAE |
|---|---|---|---|---|
| 1★ | 3 | 3.67 | +2.67 | 2.67 |
| 2★ | 5 | 4.00 | +2.00 | 2.00 |
| 3★ | 9 | 3.78 | +0.78 | 0.78 |
| 4★ | 9 | 3.91 | −0.09 | 0.58 |
| 5★ | 24 | 4.40 | −0.60 | 0.60 |

**Key findings:**
- 38% of predictions are within 0.5★ of actual, and 30% are exact matches.
- 3–5★ actuals are well-calibrated (MAE ≤ 0.78). 4★ predictions are nearly perfect
  (mean error −0.09).
- Passing the explicit `average_stars` value to the reasoner was the single biggest
  improvement (RMSE −0.14), confirming that baseline anchoring works best with
  precise numerical input rather than inferring from the manifesto.
- **Low-star reviews (1–2★) remain severely overpredicted.** All 8 low-star actuals were
  overpredicted, with mean error of +2.3★. Root cause: these reviews describe one-off bad
  *experiences* (food poisoning, insects in food, wrong orders) that cannot be inferred from
  business attributes or user preferences. The model correctly identifies that the business
  *type* matches the user's preferences but cannot anticipate service failures.
- Excluding the 8 unpredictable low-star outliers, the effective MAE on the remaining 42
  cases is ~0.62.
- **No cold-start users in this sample** (`new_experience=False` for all 50). The per-user
  holdout split guarantees every test user has indexed history, so warm/cold breakdown is not
  applicable with this dataset.

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
- **Friction-only reasoner prompt (replaced)**: The original reasoner prompt framed rating
  prediction as an "Internal Friction Analysis" — a three-step process focused on finding
  attribute violations. This caused the model to collapse all predictions toward 3.5–4.0★
  regardless of the actual rating, producing a mean signed error of +3.0 on 1★ actuals and
  −1.0 on 5★ actuals. Replacing with a balanced "Preference Alignment Analysis" that anchors
  on the user's average stars and considers both synergies and friction reduced MAE from 1.12
  to 0.96.
- **Drafter without business attributes (fixed)**: The drafter prompt demanded specificity
  about attributes ("WiFi, parking, price range") but was never passed `biz_attributes_clean`
  — only the reasoner received it. This forced the LLM to hallucinate details to satisfy the
  prompt. Passing the full attribute dictionary and rewriting the prompt to "Ground every
  detail" in provided metadata reduced hallucinated specifics.

---

## Known Weaknesses & Future Work

- **Experience-driven outlier blindness**: The biggest remaining error source. Low-star reviews
  (1–2★) are almost always driven by one-off bad experiences (food poisoning, insects, wrong
  orders) that cannot be inferred from business attributes or user preference profiles. The model
  correctly identifies attribute-preference alignment but cannot anticipate service failures.
  This is a fundamental limitation of attribute-based reasoning; addressing it would require
  incorporating real-time signals (e.g. recent review sentiment trends for the target business).
- **5★ residual conservatism**: After prompt fixes, 13/24 actual 5★ reviews were predicted at
  4.0★ (−1.0 error). The model is still slightly reluctant to commit to perfect scores. Further
  tuning of the reasoner prompt or using a stronger model for the reasoner node may help.
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
