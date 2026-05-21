from fastapi import FastAPI
from pydantic import BaseModel

from src.agent.reviewer.graph import graph
from src.agent.recommender.graph import recommend_graph

app = FastAPI(title="BCT Hack — Stateful Persona Agents")


class ReviewRequest(BaseModel):
    user_id: str
    user_name: str
    user_review_count: int

    average_stars: float
    user_elite_count: int
    user_fans: int
    business_id: str
    biz_name: str
    categories: str
    biz_attributes_clean: str


class ReviewResponse(BaseModel):
    predicted_rating: float
    draft_review: str
    user_manifesto: str
    reasoning_log: str
    new_experience: bool


class RecommendRequest(BaseModel):
    user_id: str
    user_name: str
    user_review_count: int
    average_stars: float
    user_elite_count: int
    user_fans: int
    query: str | None = None
    k: int = 5


class Recommendation(BaseModel):
    business_id: str
    biz_name: str
    categories: str
    biz_city: str
    biz_state: str
    score: float
    rationale: str


class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]
    user_manifesto: str
    reasoning_log: str
    cold_start: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-review", response_model=ReviewResponse)
async def generate_review(request: ReviewRequest):
    state = await graph.ainvoke(  # type: ignore[arg-type]
        {
            "messages": [],
            "user_id": request.user_id,
            "user_name": request.user_name,
            "user_review_count": request.user_review_count,
            "average_stars": request.average_stars,
            "user_elite_count": request.user_elite_count,
            "user_fans": request.user_fans,
            "business_id": request.business_id,
            "biz_name": request.biz_name,
            "categories": request.categories,
            "biz_attributes_clean": request.biz_attributes_clean,
        }  # type: ignore
    )
    return ReviewResponse(
        predicted_rating=state["predicted_rating"],
        draft_review=state["draft_review"],
        user_manifesto=state["user_manifesto"],
        reasoning_log=state["reasoning_log"],
        new_experience=state["new_experience"],
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    state = await recommend_graph.ainvoke(  # type: ignore[arg-type]
        {
            "user_id": request.user_id,
            "user_name": request.user_name,
            "user_review_count": request.user_review_count,
            "average_stars": request.average_stars,
            "user_elite_count": request.user_elite_count,
            "user_fans": request.user_fans,
            "query": request.query or "",
            "k": request.k,
        }  # type: ignore
    )
    return RecommendResponse(
        recommendations=[Recommendation(**r) for r in state["recommendations"]],
        user_manifesto=state["user_manifesto"],
        reasoning_log=state["reasoning_log"],
        cold_start=state["cold_start"],
    )
