from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.reviewer.state import AgentState
from src.core.llm import get_llm


class CriticOutput(BaseModel):
    is_approved: bool = Field(
        description="True if the review meets all quality criteria, False otherwise."
    )
    feedback: str = Field(
        description="Specific feedback on what needs to be fixed if the review is not approved."
    )


_llm = get_llm("gemini").with_structured_output(CriticOutput)

_SYSTEM_PROMPT = """\
You are a Quality Assurance Reviewer for a Yelp agent. Your job is to ensure the generated draft review meets strict quality criteria.

Criteria:
1. Behavioral Fidelity: Does the review sound like the user based on their 'User Manifesto'? Does it match their tone, vocabulary, and preferences?
2. Hallucination Check: Does the review incorrectly mention specific details (dishes, features) not supported by the 'Reasoning Trace' or 'Categories'?
3. Generic AI-isms: Does the review end with generic conclusions like "Overall, it was a great experience!" or use sterile, robotic language?
4. Target Matching: Does the review accurately reflect the provided 'Target Business' and 'Reasoning Trace'?

If the review passes all criteria, set is_approved to true and provide an empty string for feedback.
If it fails any criteria, set is_approved to false and provide specific, actionable feedback on what needs to be fixed without breaking character.
"""


def critic(state: AgentState) -> dict:
    content = (
        f"## User Manifesto\n{state['user_manifesto']}\n\n"
        f"## Target Business Categories\n{state['categories']}\n\n"
        f"## Reasoning Trace\n{state['reasoning_log']}\n\n"
        f"## Draft Review\n{state['draft_review']}\n"
    )

    output: CriticOutput = _llm.invoke(  # type: ignore[assignment]
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=content)]
    )

    current_count = state.get("revision_count", 0)

    return {
        "is_approved": output.is_approved,
        "critic_feedback": output.feedback,
        "revision_count": current_count + 1,
    }
