# BCT Hack — Stateful Persona Agents

DSN × BCT LLM Agent Challenge submission. Two LangGraph agents wrapped as a single FastAPI service:

- **Task A — User Modeling**: given a user's Yelp profile and a target business, simulate the review they would write and predict their star rating.
- **Task B — Recommendation**: given a user's Yelp profile and an optional freetext query, return a ranked list of personalised business recommendations with cold-start support.

## Architecture

```
POST /generate-review                  POST /recommend
        │                                      │
   ┌────▼────┐                           ┌─────▼──────┐
   │ analyst │  writing manifesto        │  profiler  │  preference manifesto
   └────┬────┘                           └─────┬──────┘
   ┌────▼──────┐                         ┌─────▼──────┐
   │ retriever │  semantic memories      │ candidate  │  vector search → yelp_businesses
   └────┬──────┘                         └─────┬──────┘
   ┌────▼──────┐                         ┌─────▼──────┐
   │ reasoner  │  friction analysis      │   ranker   │  structured LLM ranking
   └────┬──────┘                         └─────┬──────┘
   ┌────▼──────┐                              END
   │  drafter  │  ghostwritten review
   └─────┬─────┘
        END
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

# 3. Build Chroma indexes (one-time, ~2 min each)
uv run python -m scripts.ingest_reviews        # yelp_reviews  — 1,421 review docs
uv run python -m scripts.ingest_businesses     # yelp_businesses — 1,039 business docs
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
  "biz_attributes_clean": "WiFi: free, Parking: street, Noise: loud"
}
```

**Response**
```json
{
  "predicted_rating": 3.5,
  "draft_review": "I wanted to love this place...",
  "user_manifesto": "Cassie is a seasoned reviewer who...",
  "reasoning_log": "Step 1 — Attribute Collision: ...",
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
  "k": 5
}
```

`query` and `k` are optional (defaults: no query, top 5).

**Response**
```json
{
  "recommendations": [
    {
      "business_id": "WZPCfTRiN4ipajP7gFfziA",
      "biz_name": "The Centennial Cafe",
      "categories": "Restaurants, Coffee & Tea, Food",
      "score": 0.9,
      "rationale": "This place fits because you prefer quiet spots and it has a quiet noise level with free WiFi."
    }
  ],
  "user_manifesto": "Cassie gravitates toward casual, food-focused...",
  "reasoning_log": "Compared candidates against manifesto...",
  "cold_start": false
}
```

`cold_start: true` is returned when the user has no review history — the agent infers preferences from demographic stats only.

## Project Structure

```
src/
├── api.py                        FastAPI app — /generate-review and /recommend
├── core/
│   ├── settings.py               Pydantic env config
│   ├── llm.py                    get_llm("gemini")
│   ├── embeddings.py             GeminiEmbeddings (REST)
│   └── vectorstore.py            get_vectorstore(), get_business_vectorstore()
└── agent/
    ├── reviewer/                 Task A
    │   ├── graph.py              analyst → retriever → reasoner → drafter
    │   ├── state.py              AgentState
    │   └── nodes/
    │       ├── analyst.py        Builds writing-style manifesto from past reviews
    │       ├── retriever.py      Fetches semantically similar past reviews
    │       ├── reasoner.py       Predicts star rating via friction analysis
    │       └── drafter.py        Ghostwrites review in user's voice
    └── recommender/              Task B
        ├── graph.py              profiler → candidate → ranker
        ├── state.py              RecommenderState
        └── nodes/
            ├── profiler.py       Builds preference manifesto; handles cold-start
            ├── candidate.py      Vector search over businesses, filters visited
            └── ranker.py         LLM ranking with structured output

scripts/
├── ingest_reviews.py             Build yelp_reviews Chroma collection
└── ingest_businesses.py          Build yelp_businesses Chroma collection
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | Yes | — | Gemini API key |
| `GEMINI_MODEL` | No | `gemini-2.5-flash-lite` | Chat model |
| `EMBEDDING_MODEL` | No | `models/gemini-embedding-2` | Embedding model |
| `CHROMA_PATH` | No | `data/chroma_db` | Chroma persistence directory |
