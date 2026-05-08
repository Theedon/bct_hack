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
