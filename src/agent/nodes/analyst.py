from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.core.llm import get_llm
from src.core.vectorstore import get_vectorstore

_llm = get_llm("gemini")

_SYSTEM_PROMPT = """\
You are a Behavioral Analyst. Analyze the following user metrics — and their \
actual review samples where available — to create a 'Writing & Rating Manifesto.'

Your Task:
Define the user's persona. Are they a 'Polite Amateur,' a \
'Hard-to-Please Professional,' or a 'Vibe-Focused Influencer'? How do they \
handle 3-star vs 5-star ratings? What is their vocabulary level?

Output: A concise, 3-sentence persona description that will guide all future \
writing."""


def _build_metrics(state: AgentState) -> str:
    elite_status = (
        f"{state['user_elite_count']} years of Elite status"
        if state["user_elite_count"] > 0
        else "Non-Elite member"
    )
    return (
        f"Name: {state['user_name']}\n"
        f"Status: {elite_status}\n"
        f"Influence: {state['user_fans']} fans, {state['user_review_count']} reviews\n"
        f"Rating Bias: {state['average_stars']} average stars"
    )


def _fetch_user_reviews(user_id: str) -> list[dict]:
    try:
        vs = get_vectorstore()
        results = vs.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"],
        )
        return [
            {"text": doc, "stars": meta["stars_review"], "biz_name": meta["biz_name"]}
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
    except Exception:
        return []


def analyst(state: AgentState) -> dict:
    metrics = _build_metrics(state)
    reviews = _fetch_user_reviews(state["user_id"])

    if reviews:
        samples = "\n\n".join(
            f"[{r['stars']}/5 — {r['biz_name']}]\n{r['text']}" for r in reviews
        )
        content = f"{metrics}\n\nWriting Samples ({len(reviews)} reviews found):\n{samples}"
    else:
        content = metrics

    response = _llm.invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=content)]
    )
    return {"user_manifesto": response.content}
