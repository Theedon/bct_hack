# BCT Hack — Stateful Persona Agents

DSN × BCT LLM Agent Challenge submission. Two LangGraph agents wrapped as a single FastAPI service:

- **Task A — User Modeling**: given a user's Yelp profile and a target business, simulate the review they would write and predict their star rating. Includes a critic/reflection loop for quality assurance.
- **Task B — Recommendation**: given a user's Yelp profile and an optional freetext query, return a ranked list of personalised business recommendations. Supports cold-start users, multi-turn conversational refinement, and Nigerian contextualization.

## Architecture

```
POST /generate-review                  POST /recommend
        │                                      │
   ┌────▼────┐                           ┌─────▼──────┐
   │ analyst │  persona manifesto        │  profiler  │  preference manifesto
   └────┬────┘                           └─────┬──────┘
   ┌────▼──────┐                         ┌─────▼──────┐
   │ retriever │  reference reviews      │ candidate  │  vector search
   └────┬──────┘                         └─────┬──────┘
   ┌────▼──────┐                         ┌─────▼──────┐
   │ reasoner  │  predicted rating       │   ranker   │  LLM re-ranking
   └────┬──────┘                         └─────┬──────┘
   ┌────▼──────┐                              END
   │  drafter  │  ghostwritten review
   └────┬──────┘
   ┌────▼──────┐
   │  critic   │──→ approved? → END
   └────┬──────┘
        │ revise (up to MAX_REVISIONS)
        └──→ drafter
```

Both pipelines share:
- **LLM**: Gemini `gemini-2.5-flash-lite` via `langchain-google-genai`
- **Embeddings**: Gemini `models/gemini-embedding-2` via REST
- **Vector store**: ChromaDB with two collections — `yelp_reviews` (per-review) and `yelp_businesses` (per-business)

## Requirements

- Python 3.14+
- [uv](https://github.com/astral-sh/uv)
- Google Gemini API key

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env.local
# Edit .env.local and set GOOGLE_API_KEY

# 3. Activate the pre-commit formatting hook (one-time per clone)
uv run pre-commit install

# 4. Build Chroma indexes (one-time, ~2 min each)
uv run python -m scripts.ingest_reviews
uv run python -m scripts.ingest_businesses
```

Data files expected at `data/yelp_review/train.csv` and `data/yelp_review/test.csv`.

## Running

```bash
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000
```

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## Docker

### Quick start (recommended for judges)

Pull and run — no Python, no `uv`, no ingestion required:

```bash
docker run -e GOOGLE_API_KEY=your-key -p 8000:8000 theedon/bct-hack:latest
```

### Build from source

```bash
# Make sure data/chroma_db/ exists (run ingest scripts first if needed)
GOOGLE_API_KEY=your-key docker compose up --build
```

### Verify

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## Endpoints

### `POST /generate-review` — Task A

Simulates the review and star rating a specific user would give a target business.

**Request**
```json
{
  "user_id": "3MpDvy5gEdsbZh9-p92dHg",
  "user_name": "Cassie",
  "user_review_count": 621,
  "average_stars": 3.67,
  "user_elite_count": 14,
  "user_fans": 38,
  "business_id": "abc123",
  "biz_name": "The Rusty Anchor",
  "categories": "Seafood, Restaurants",
  "biz_attributes_clean": "WiFi: free, Parking: street, Noise: loud",
  "nigerian_mode": false
}
```

`nigerian_mode` is optional (default `false`). When `true`, the agent adopts Nigerian English phrasing and cultural archetypes.

**Response**
```json
{
  "predicted_rating": 3.5,
  "draft_review": "I wanted to love this place...",
  "user_manifesto": "Cassie is a seasoned reviewer who...",
  "reasoning_log": "Step 1 — Preference Alignment: ...",
  "new_experience": false
}
```

---

### `POST /recommend` — Task B

Returns a personalised ranked list of businesses. Supports cold-start users (no review history) and cross-domain discovery. Already-visited businesses are automatically excluded.

**Request**
```json
{
  "user_id": "3MpDvy5gEdsbZh9-p92dHg",
  "user_name": "Cassie",
  "user_review_count": 621,
  "average_stars": 3.67,
  "user_elite_count": 14,
  "user_fans": 38,
  "query": "somewhere I can read for an hour",
  "k": 10,
  "messages": [],
  "nigerian_mode": false
}
```

| Field | Required | Default | Description |
|---|---|---|---|
| `query` | No | `null` | Natural-language search query |
| `k` | No | `10` | Number of recommendations to return |
| `messages` | No | `[]` | Conversation history for multi-turn refinement (`[{"role": "user", "content": "..."}]`) |
| `nigerian_mode` | No | `false` | Use Nigerian English and cultural touchpoints |

**Response**
```json
{
  "recommendations": [
    {
      "business_id": "WZPCfTRiN4ipajP7gFfziA",
      "biz_name": "The Centennial Cafe",
      "categories": "Restaurants, Coffee & Tea, Food",
      "biz_city": "Philadelphia",
      "biz_state": "PA",
      "score": 0.9,
      "rationale": "This place fits because you prefer quiet spots and it has a quiet noise level with free WiFi."
    }
  ],
  "user_manifesto": "Cassie gravitates toward casual, food-focused...",
  "reasoning_log": "Compared candidates against manifesto...",
  "cold_start": false
}
```

`cold_start: true` is returned when the user has no review history — the agent builds a demographic-only preference manifesto without fabricating preferences.

## Project Structure

```
src/
├── api.py                        FastAPI app — /generate-review and /recommend
├── main.py                       CLI evaluator for Task A (RMSE, MAE, ROUGE-L)
├── main_recommend.py             CLI evaluator for Task B (NDCG@10, Hit@10)
├── core/
│   ├── settings.py               Pydantic env config
│   ├── llm.py                    get_llm("gemini")
│   ├── embeddings.py             GeminiEmbeddings (REST)
│   └── vectorstore.py            get_vectorstore(), get_business_vectorstore()
└── agent/
    ├── reviewer/                 Task A
    │   ├── graph.py              analyst → retriever → reasoner → drafter → critic
    │   ├── state.py              AgentState
    │   └── nodes/
    │       ├── analyst.py        Persona manifesto from past reviews
    │       ├── retriever.py      Reference reviews via semantic search
    │       ├── reasoner.py       Star rating via preference alignment analysis
    │       ├── drafter.py        Ghostwrites review in user's voice
    │       └── critic.py         Quality gate — fidelity, hallucination, AI-isms
    └── recommender/              Task B
        ├── graph.py              profiler → candidate → ranker
        ├── state.py              RecommenderState
        └── nodes/
            ├── profiler.py       Preference manifesto; handles cold-start
            ├── candidate.py      Vector search, filters visited businesses
            └── ranker.py         LLM re-ranking with structured output

scripts/
├── ingest_reviews.py             Build yelp_reviews Chroma collection
└── ingest_businesses.py          Build yelp_businesses Chroma collection

docs/                             Living design documents (task_A.md, task_B.md, data.md)
paper/                            Solution paper sources + PDF converter
notebooks/                        Data exploration (Yelp_EDA.ipynb)
tests/                            Pytest suite
demos/                            Recommendation demo scenarios
```

## Evaluation

```bash
# Task A — RMSE, MAE, ROUGE-L over held-out reviews
uv run python -m src.main --n 50 --output results/output.csv

# Task B — NDCG@10, Hit@10 over held-out recommendations
uv run python -m src.main_recommend --n 50 --k 10 --output results/output_recommend.csv
```

Both evaluators use the per-user holdout test set (`data/yelp_review/test.csv`).

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | Yes | — | Gemini API key |
| `GEMINI_MODEL` | No | `gemini-2.5-flash-lite` | Chat model |
| `EMBEDDING_MODEL` | No | `models/gemini-embedding-2` | Embedding model |
| `CHROMA_PATH` | No | `data/chroma_db` | Chroma persistence directory |
| `MAX_REVISIONS` | No | `2` | Max critic→drafter revision cycles (Task A) |
