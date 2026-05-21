from typing import TypedDict


class RecommenderState(TypedDict):
    # Inputs
    user_id: str
    user_name: str
    user_review_count: int
    average_stars: float
    user_elite_count: int
    user_fans: int
    query: str
    k: int
    messages: list[dict[str, str]]  # conversation history, newest last

    # Profiler outputs
    user_manifesto: str
    cold_start: bool
    visited_business_ids: list[str]

    # Candidate outputs
    candidates: list[dict]

    # Ranker outputs
    recommendations: list[dict]
    reasoning_log: str
