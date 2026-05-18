from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.recommender.state import RecommenderState
from src.core.llm import get_llm


class RankedItem(BaseModel):
    candidate_number: int = Field(
        description="The [N] number of the candidate exactly as shown in the list (1-based)"
    )
    score: float = Field(description="Fit score from 0.0 to 1.0", ge=0.0, le=1.0)
    rationale: str = Field(
        description="1-2 sentences in second person explaining why this fits the user"
    )


class RankerOutput(BaseModel):
    reasoning: str = Field(
        description="Internal chain of thought across the full candidate list"
    )
    ranked: list[RankedItem] = Field(description="Top-k businesses ranked by fit")


_llm = get_llm("gemini").with_structured_output(RankerOutput)

_SYSTEM_PROMPT = """\
You are the Ranker for a personalised recommendation agent.

You will receive:
- The user's Preference Manifesto.
- An optional explicit Query (what the user is looking for right now).
- A numbered list of Candidate businesses with their attributes.

Your Task:
1. In `reasoning`, briefly walk through how you compared candidates against \
   the manifesto and the query. Reference concrete attributes — do not be \
   vague.
2. Return the top-k candidates in the `ranked` field, ordered best-first. \
   Each entry must include:
   - `candidate_number`: the exact [N] number shown in the candidate list
   - `score`: a fit confidence in [0.0, 1.0]
   - `rationale`: 1–2 sentences in second person ("This place fits because \
     you...") tied to concrete attributes

Rules:
- Use only candidate_numbers that appear in the list provided.
- If the explicit query is present, it should heavily shape the ranking.
- Diversity is a tiebreaker — avoid filling the top with near-duplicate \
  categories unless the user's preferences clearly justify it.
- Do not fabricate attributes that are not in the candidate metadata."""


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates, start=1):
        lines.append(
            f"[{i}] Name: {c['biz_name']}\n"
            f"    Categories: {c['categories']}\n"
            f"    Attributes: {c['biz_attributes_clean']}\n"
            f"    Yelp stars: {c['biz_stars']} | "
            f"avg reviewer stars: {c['avg_user_stars']} | "
            f"reviews: {c['review_count']}"
        )
    return "\n\n".join(lines)


def ranker(state: RecommenderState) -> dict:
    candidates = state.get("candidates", [])
    k = state.get("k", 5)
    if not candidates:
        return {"recommendations": [], "reasoning_log": "No candidates available."}

    query_text = (state.get("query") or "").strip() or "(no explicit query)"
    content = (
        f"## Preference Manifesto\n{state['user_manifesto']}\n\n"
        f"## Explicit Query\n{query_text}\n\n"
        f"## Candidates ({len(candidates)} total — return top {k})\n"
        f"{_format_candidates(candidates)}"
    )

    output: RankerOutput = _llm.invoke(  # type: ignore[assignment]
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=content)]
    )

    recommendations: list[dict] = []
    for item in output.ranked[:k]:
        idx = item.candidate_number - 1
        if not (0 <= idx < len(candidates)):
            continue
        biz = candidates[idx]
        recommendations.append(
            {
                "business_id": biz["business_id"],
                "biz_name": biz["biz_name"],
                "categories": biz["categories"],
                "score": round(item.score, 3),
                "rationale": item.rationale,
            }
        )

    return {"recommendations": recommendations, "reasoning_log": output.reasoning}
