from langgraph.graph import END, StateGraph

from src.agent.recommender.nodes.candidate import candidate
from src.agent.recommender.nodes.profiler import profiler
from src.agent.recommender.nodes.ranker import ranker
from src.agent.recommender.state import RecommenderState


def build_graph():
    g = StateGraph(RecommenderState)
    g.add_node("profiler", profiler)
    g.add_node("candidate", candidate)
    g.add_node("ranker", ranker)
    g.set_entry_point("profiler")
    g.add_edge("profiler", "candidate")
    g.add_edge("candidate", "ranker")
    g.add_edge("ranker", END)
    return g.compile()


recommend_graph = build_graph()
