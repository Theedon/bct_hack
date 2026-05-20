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
- Target Business Metadata and Context describing the unseen business to be rated, including location, factual attributes, and a summary of vibes/complaints from other users.

Your Task: Perform an Internal Friction Analysis in three steps.

Step 1 — Attribute Collision:
Identify specific attributes from the Target Business that match or violate the \
user's known preferences (inferred from the Manifesto and Memories). Friction \
transfers across business types — if the user docked stars for poor parking at a \
park, the same friction applies to a restaurant with no valet.

Step 2 — Experience Extrapolation:
If this is a new business type the user has never encountered, find the closest \
friction analogy from their memories. Describe the analogy explicitly.

Step 3 — Rating Calculation:
Derive a precise star rating (1.0 to 5.0) from the collision analysis. \
Justify the number.

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
        f"## Semantic Memories\n{memories}\n\n"
        f"## Target Business\n"
        f"Name: {state['biz_name']}\n"
        f"Categories: {state['categories']}\n"
        f"Attributes: {state['biz_attributes_clean']}\n"
        f"Context & Facts: {state.get('business_context', 'No extra context available')}"
    )
    output: ReasonerOutput = _llm.invoke(  # type: ignore[assignment]
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=content)]
    )
    return {
        "reasoning_log": output.reasoning,
        "predicted_rating": round(output.predicted_rating, 1),
    }
