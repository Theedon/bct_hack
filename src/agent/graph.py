from langgraph.graph import END, StateGraph

from src.agent.nodes.analyst import analyst
from src.agent.state import AgentState


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("analyst", analyst)
    g.set_entry_point("analyst")
    g.add_edge("analyst", END)
    return g.compile()


graph = build_graph()
