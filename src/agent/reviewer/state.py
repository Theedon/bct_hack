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
    nigerian_mode: bool

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

    # Reasoner node outputs
    reasoning_log: str
    predicted_rating: float

    # Drafter node output
    draft_review: str

    # Critic node output
    critic_feedback: str
    revision_count: int
    is_approved: bool
