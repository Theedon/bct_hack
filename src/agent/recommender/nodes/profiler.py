import math
import random

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.recommender.state import RecommenderState
from src.core.llm import clean_content, get_llm
from src.core.vectorstore import get_vectorstore

_MAX_REVIEWS = 15

_llm = get_llm("gemini")

_SYSTEM_PROMPT_WARM = """\
You are a Preference Analyst building a Recommendation Manifesto for a Yelp user.

Read the user's metrics and review samples and infer a concise profile of \
what they value in a business. Cover: cuisine / business types they gravitate \
toward, ambiance preferences (quiet vs lively, casual vs upscale), price \
sensitivity, service expectations, and any deal-breakers you can infer from \
their lower-rated reviews.

Output: A 4–6 sentence preference manifesto written in third person ("This \
user prefers..."). Be specific and grounded in the samples — do not invent \
preferences the data does not support."""

_SYSTEM_PROMPT_COLD = """\
You are a Preference Analyst working with very limited information about a \
Yelp user — no review history is available, only demographic stats.

Infer cautiously. Mention what the stats suggest (e.g. a high review count \
with Elite status implies an active, discerning reviewer; zero reviews \
implies a true newcomer) but do not fabricate specific cuisine or ambiance \
preferences.

Output: A 3–4 sentence preference manifesto in third person that \
acknowledges the limited signal and stays general."""


def _build_metrics(state: RecommenderState) -> str:
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


def _sample_reviews(reviews: list[dict], n: int) -> list[dict]:
    low = [r for r in reviews if r["stars"] <= 3]
    high = [r for r in reviews if r["stars"] > 3]

    if not low or not high:
        return random.sample(reviews, min(n, len(reviews)))

    low_slots = max(1, math.floor(n * len(low) / len(reviews)))
    high_slots = max(1, n - low_slots)

    sampled_low = random.sample(low, min(low_slots, len(low)))
    sampled_high = random.sample(high, min(high_slots, len(high)))
    return sampled_low + sampled_high


def _fetch_user_reviews(user_id: str) -> tuple[list[dict], list[str]]:
    """Return (sampled reviews, full list of visited business_ids)."""
    try:
        vs = get_vectorstore()
        results = vs.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"],
        )
        reviews = [
            {
                "text": doc,
                "stars": meta["stars_review"],
                "biz_name": meta["biz_name"],
                "categories": meta.get("categories", ""),
                "business_id": meta.get("business_id", ""),
            }
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
    except Exception:
        return [], []

    visited = sorted({r["business_id"] for r in reviews if r["business_id"]})
    sampled = (
        _sample_reviews(reviews, _MAX_REVIEWS)
        if len(reviews) > _MAX_REVIEWS
        else reviews
    )
    return sampled, visited


_HISTORY_SUFFIX = """\

## Conversation History
{history}

Given the above conversation history, incorporate the user's latest follow-up \
request into the preference manifesto. If the user asked for something specific \
(e.g. "cheaper options", "outdoor seating"), reflect that constraint in the manifesto."""


def _format_history(messages: list[dict[str, str]]) -> str:
    lines = []
    for m in messages:
        role = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


def profiler(state: RecommenderState) -> dict:
    metrics = _build_metrics(state)
    reviews, visited = _fetch_user_reviews(state["user_id"])

    if reviews:
        samples = "\n\n".join(
            f"[{r['stars']}/5 — {r['biz_name']} ({r['categories']})]\n{r['text']}"
            for r in reviews
        )
        content = f"{metrics}\n\nReview Samples ({len(reviews)} reviews):\n{samples}"
        system_prompt = _SYSTEM_PROMPT_WARM
        cold_start = False
    else:
        content = metrics
        system_prompt = _SYSTEM_PROMPT_COLD
        cold_start = True

    messages = state.get("messages") or []
    if messages:
        content += _HISTORY_SUFFIX.format(history=_format_history(messages))

    if state.get("nigerian_mode"):
        if cold_start:
            system_prompt += (
                "\n\nFrame the manifesto in a Nigerian context using Nigerian English "
                "or archetypes, but do not fabricate specific Nigerian cuisine or "
                "venue preferences without data."
            )
        else:
            system_prompt += (
                "\n\nFrame the manifesto in a Nigerian context. Describe their preferences "
                "using Nigerian cultural touchpoints (e.g., affinity for 'buka' spots, "
                "expectations for 'correct' portions, or preference for 'bougie' island places)."
            )

    response = _llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=content)]
    )
    return {
        "user_manifesto": clean_content(response.content),
        "cold_start": cold_start,
        "visited_business_ids": visited,
    }
