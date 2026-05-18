from langgraph.graph import END, StateGraph

from src.agent.reviewer.nodes.analyst import analyst
from src.agent.reviewer.nodes.drafter import drafter
from src.agent.reviewer.nodes.reasoner import reasoner
from src.agent.reviewer.nodes.retriever import retriever
from src.agent.reviewer.state import AgentState


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
