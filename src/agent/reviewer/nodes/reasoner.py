from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.reviewer.state import AgentState
from src.core.llm import get_llm


class ReasonerOutput(BaseModel):
    reasoning: str = Field(description="Step-by-step internal friction analysis")
    predicted_rating: float = Field(description="Predicted star rating from 1.0 to 5.0")


_llm = get_llm("gemini").with_structured_output(ReasonerOutput)

_SYSTEM_PROMPT = """\
You are the Reasoning Core of a Stateful Persona Agent.

You have been given:
- A Persona Manifesto describing who this user is and how they rate businesses.
- Semantic Memories of their past reviews at similar venues (or a note that none exist).
- Target Business Metadata describing the unseen business to be rated.

Your Task: Perform a Preference Alignment Analysis in four steps.

Step 1 — Baseline Anchoring:
Start from the user's historical average star rating as your prior. A user who \
averages 4.5★ is generous by nature; a user who averages 2.5★ is harsh. Your \
final prediction should deviate from this baseline only when specific evidence \
pushes it up or down.

Step 2 — Positive Synergies:
Identify specific attributes from the Target Business that align with the user's \
known preferences (inferred from the Manifesto and Memories). Strong alignment \
should push the rating above the baseline — e.g. a user who loves craft cocktails \
visiting a cocktail-focused bar should trend toward 5.0.

Step 3 — Negative Friction:
Identify specific attributes from the Target Business that violate the user's \
known preferences. Friction transfers across business types — if the user docked \
stars for poor parking at a park, the same friction applies to a restaurant with \
no valet. Strong friction should push the rating below the baseline.

Step 4 — Rating Calculation:
Starting from the baseline, apply the net effect of synergies and friction to \
derive a precise star rating (1.0 to 5.0). Use the full scale boldly: if every \
attribute aligns and the user is generous, predict 5.0. If every attribute \
clashes and the user is harsh, predict 1.0. Do not default to the middle of \
the scale.

Output your full reasoning trace in the `reasoning` field and the final rating \
in `predicted_rating`."""


def _format_memories(state: AgentState) -> str:
    if state["new_experience"] or not state["retrieved_reviews"]:
        return "No prior experiences found — extrapolate entirely from the Persona Manifesto."
    lines = []
    for r in state["retrieved_reviews"]:
        lines.append(f"[{r['stars']}/5 — {r['biz_name']} ({r['categories']})]")
        lines.append(r["text"][:300])
        lines.append("")
    return "\n".join(lines).strip()


def reasoner(state: AgentState) -> dict:
    memories = _format_memories(state)
    content = (
        f"## Persona Manifesto\n{state['user_manifesto']}\n\n"
        f"## User Baseline\nHistorical average star rating: {state['average_stars']}\n\n"
        f"## Semantic Memories\n{memories}\n\n"
        f"## Target Business\n"
        f"Name: {state['biz_name']}\n"
        f"Categories: {state['categories']}\n"
        f"Attributes: {state['biz_attributes_clean']}"
    )
    output: ReasonerOutput = _llm.invoke(  # type: ignore[assignment]
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=content)]
    )
    return {
        "reasoning_log": output.reasoning,
        "predicted_rating": round(output.predicted_rating, 1),
    }
