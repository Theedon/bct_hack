# Claude Guidelines

## Commits

- Use conventional commits: `feat`, `fix`, `chore`, `refactor`, `test`, `docs`, `ci`, `perf`
- Include a scope in parens when relevant: `feat(agent): ...`
- Split commits by concern — never lump unrelated changes into one commit
- Commit after each discrete task is complete and verified working
- Never commit broken or unverified code
- Never auto-push — pushes require explicit instruction

## Commands

- Always run Python commands via `uv run` — never `python3`, `python`, or direct script invocation
- Examples: `uv run pytest`, `uv run python -c "..."`, `uv run mypy src/`
- Run `src/main.py` as a module: `uv run python -m src.main --n 5 --output results/output.csv`

## Tests

- Write tests for business logic, data transformations, utilities, and error paths
- Skip tests for pure wiring/orchestration code unless it encodes meaningful logic
- Always update existing tests before committing a change to covered code
- Prefer testing behaviour and contracts over implementation details
- For LLM-facing components, mock the LLM and test the surrounding logic
