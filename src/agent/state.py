from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    # User metadata (populated at graph entry)
    user_id: str
    user_name: str
    user_review_count: int
    user_elite_count: int
    user_fans: int
    average_stars: float

    # Analyst node output
    user_manifesto: str

    # Target business (populated at graph entry)
    business_id: str
    biz_name: str
    categories: str
    biz_attributes_clean: str

    # Retriever node outputs
    retrieved_reviews: list[dict]
    new_experience: bool
