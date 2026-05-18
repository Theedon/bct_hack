from src.agent.reviewer.state import AgentState
from src.core.vectorstore import get_vectorstore


def retriever(state: AgentState) -> dict:
    query = (
        f"Category: {state['categories']} | "
        f"Attributes: {state['biz_attributes_clean'][:150]}"
    )
    results = get_vectorstore().similarity_search(
        query, k=5, filter={"user_id": state["user_id"]}
    )

    if not results:
        return {"retrieved_reviews": [], "new_experience": True}

    reviews = [
        {
            "text": r.metadata.get("review_text", ""),
            "biz_name": r.metadata["biz_name"],
            "stars": r.metadata["stars_review"],
            "categories": r.metadata["categories"],
        }
        for r in results
    ]
    return {"retrieved_reviews": reviews, "new_experience": False}
