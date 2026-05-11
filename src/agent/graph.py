from langgraph.graph import END, StateGraph

from src.agent.nodes.analyst import analyst
from src.agent.nodes.drafter import drafter
from src.agent.nodes.reasoner import reasoner
from src.agent.nodes.retriever import retriever
from src.agent.state import AgentState


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("analyst", analyst)
    g.add_node("retriever", retriever)
    g.add_node("reasoner", reasoner)
    g.add_node("drafter", drafter)
    g.set_entry_point("analyst")
    g.add_edge("analyst", "retriever")
    g.add_edge("retriever", "reasoner")
    g.add_edge("reasoner", "drafter")
    g.add_edge("drafter", END)
    return g.compile()


graph = build_graph()
