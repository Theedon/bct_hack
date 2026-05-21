# Claude Guidelines

## Commits

- Use conventional commits: `feat`, `fix`, `chore`, `refactor`, `test`, `docs`, `ci`, `perf`
- Include a scope in parens when relevant: `feat(recommend): ...`, `feat(reviewer): ...`, `feat(api): ...`
- Split commits by concern — never lump unrelated changes into one commit
- Commit after each discrete task is complete and verified working
- Never commit broken or unverified code
- Never auto-push — pushes require explicit instruction

## Commands

Always use `uv run` — never `python3`, `python`, or direct invocation.

| Purpose | Command |
|---|---|
| Start API | `uv run uvicorn src.api:app --host 0.0.0.0 --port 8000` |
| Task A evaluator | `uv run python -m src.main --n 5 --output results/output.csv` |
| Task B evaluator | `uv run python -m src.main_recommend --n 20 --k 5 --output results/output_recommend.csv` |
| Build review index | `uv run python -m scripts.ingest_reviews` |
| Build business index | `uv run python -m scripts.ingest_businesses` |
| Run tests | `uv run pytest` |
| Format code | `uv run black .` |

**Task A evaluator flags:** `--n` rows (default 5), `--output` path, `--delay` secs between calls (default 1.0)

**Task B evaluator flags:** `--n` users (default all), `--k` recs per user (default 10), `--output` path, `--delay` secs between users (default 1.0)

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

## Solution Paper

`docs/task_A.md` and `docs/task_B.md` are living documents that feed into the hackathon
solution paper. Update the relevant file whenever a task produces:

- A new architectural decision (with the rationale behind it)
- An evaluation result (metric + dataset + conditions)
- An ablation finding (what was tried, what changed, and why)
- A known weakness or future work item that emerged from testing

Do **not** update these files for: bug fixes, CI changes, test additions, formatting, or
refactors that do not change the system's approach or behaviour.

Rule of thumb: if the finding would belong in a conference paper's Methods or Results section,
add it. If it would only appear in a commit message, skip it.

## Key Patterns

- Both agents use `async def` routes with `graph.ainvoke()` — never sync routes with `graph.invoke()` inside FastAPI (causes event-loop deadlock with the Gemini client)
- LLM is always `get_llm("gemini")` from `src/core/llm.py` — Anthropic key is not required
- Embeddings use `GeminiEmbeddings` from `src/core/embeddings.py` (direct REST, not langchain wrapper)
- Ranker structured output uses `candidate_number: int` (1-based index) not `business_id` to avoid LLM hallucinating IDs
- ChromaDB has two collections: `yelp_reviews` (review-indexed) and `yelp_businesses` (business-indexed)
