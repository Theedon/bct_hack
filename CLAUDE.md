# Claude Guidelines

## Commits

- Use conventional commits: `feat`, `fix`, `chore`, `refactor`, `test`, `docs`, `ci`, `perf`
- Include a scope in parens when relevant: `feat(recommend): ...`, `feat(reviewer): ...`, `feat(api): ...`
- Split commits by concern — never lump unrelated changes into one commit
- Commit after each discrete task is complete and verified working
- Never commit broken or unverified code
- Never auto-push — pushes require explicit instruction

## Commands

- Always run Python commands via `uv run` — never `python3`, `python`, or direct script invocation
- Examples: `uv run pytest`, `uv run python -c "..."`, `uv run mypy src/`
- Run the CLI evaluator: `uv run python -m src.main --n 5 --output results/output.csv`
- Run ingest scripts: `uv run python -m scripts.ingest_reviews` / `uv run python -m scripts.ingest_businesses`
- Start the API: `uv run uvicorn src.api:app --host 0.0.0.0 --port 8000`

## Tests

- Write tests for business logic, data transformations, utilities, and error paths
- Skip tests for pure wiring/orchestration code unless it encodes meaningful logic
- Always update existing tests before committing a change to covered code
- Prefer testing behaviour and contracts over implementation details
- For LLM-facing components, mock the LLM and test the surrounding logic

## Project Layout

```
src/agent/
    reviewer/          Task A — graph, state, nodes (analyst, retriever, reasoner, drafter)
    recommender/       Task B — graph, state, nodes (profiler, candidate, ranker)
src/core/              Shared: llm.py, embeddings.py, vectorstore.py, settings.py
src/api.py             FastAPI service — POST /generate-review and POST /recommend
src/main.py            CLI evaluator for Task A
scripts/
    ingest_reviews.py      Build yelp_reviews Chroma collection
    ingest_businesses.py   Build yelp_businesses Chroma collection
data/
    yelp_review/train.csv
    yelp_review/test.csv
    chroma_db/             Persisted Chroma index (gitignored)
```

## Key Patterns

- Both agents use `async def` routes with `graph.ainvoke()` — never sync routes with `graph.invoke()` inside FastAPI (causes event-loop deadlock with the Gemini client)
- LLM is always `get_llm("gemini")` from `src/core/llm.py` — Anthropic key is not required
- Embeddings use `GeminiEmbeddings` from `src/core/embeddings.py` (direct REST, not langchain wrapper)
- Ranker structured output uses `candidate_number: int` (1-based index) not `business_id` to avoid LLM hallucinating IDs
- ChromaDB has two collections: `yelp_reviews` (review-indexed) and `yelp_businesses` (business-indexed)
