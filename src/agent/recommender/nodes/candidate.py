from src.agent.recommender.state import RecommenderState
from src.core.vectorstore import get_business_vectorstore

_OVERFETCH_K = 25
_MAX_CANDIDATES = 20
_MANIFESTO_CHARS = 600


def _build_query(manifesto: str, user_query: str) -> str:
    manifesto_excerpt = manifesto.strip().replace("\n", " ")[:_MANIFESTO_CHARS]
    if user_query.strip():
        return f"{user_query.strip()} | Preferences: {manifesto_excerpt}"
    return f"Preferences: {manifesto_excerpt}"


def candidate(state: RecommenderState) -> dict:
    query = _build_query(state["user_manifesto"], state.get("query", "") or "")
    visited = set(state.get("visited_business_ids", []) or [])

    hits = get_business_vectorstore().similarity_search(query, k=_OVERFETCH_K)

    candidates: list[dict] = []
    for h in hits:
        business_id = h.metadata.get("business_id", "")
        if business_id in visited:
            continue
        candidates.append(
            {
                "business_id": business_id,
                "biz_name": h.metadata.get("biz_name", ""),
                "categories": h.metadata.get("categories", ""),
                "biz_attributes_clean": h.metadata.get("biz_attributes_clean", ""),
                "biz_city": h.metadata.get("biz_city", ""),
                "biz_state": h.metadata.get("biz_state", ""),
                "biz_stars": h.metadata.get("biz_stars", 0.0),
                "avg_user_stars": h.metadata.get("avg_user_stars", 0.0),
                "review_count": h.metadata.get("review_count", 0),
            }
        )
        if len(candidates) >= _MAX_CANDIDATES:
            break

    return {"candidates": candidates}
