from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.core.llm import get_llm

_llm = get_llm("gemini")

_SYSTEM_PROMPT = """\
You are a Behavioral Analyst. Analyze the following user metrics to create a \
'Writing & Rating Manifesto.'

Your Task:
Define the user's persona. Are they a 'Polite Amateur,' a \
'Hard-to-Please Professional,' or a 'Vibe-Focused Influencer'? How do they \
handle 3-star vs 5-star ratings? What is their vocabulary level?

Output: A concise, 3-sentence persona description that will guide all future \
writing."""


def analyst(state: AgentState) -> dict:
    elite_status = (
        f"{state['user_elite_count']} years of Elite status"
        if state["user_elite_count"] > 0
        else "Non-Elite member"
    )
    metrics = (
        f"Name: {state['user_name']}\n"
        f"Status: {elite_status}\n"
        f"Influence: {state['user_fans']} fans, {state['user_review_count']} reviews\n"
        f"Rating Bias: {state['average_stars']} average stars"
    )
    response = _llm.invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=metrics)]
    )
    return {"user_manifesto": response.content}
