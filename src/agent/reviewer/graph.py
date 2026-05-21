from langgraph.graph import END, StateGraph

from src.agent.reviewer.nodes.analyst import analyst
from src.agent.reviewer.nodes.critic import critic
from src.agent.reviewer.nodes.drafter import drafter
from src.agent.reviewer.nodes.reasoner import reasoner
from src.agent.reviewer.nodes.retriever import retriever
from src.agent.reviewer.state import AgentState
from src.core.settings import settings


def route_critic(state: AgentState) -> str:
    if (
        state.get("is_approved")
        or state.get("revision_count", 0) >= settings.MAX_REVISIONS
    ):
        return END
    return "drafter"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("analyst", analyst)
    g.add_node("retriever", retriever)
    g.add_node("reasoner", reasoner)
    g.add_node("drafter", drafter)
    g.add_node("critic", critic)
    g.set_entry_point("analyst")
    g.add_edge("analyst", "retriever")
    g.add_edge("retriever", "reasoner")
    g.add_edge("reasoner", "drafter")
    g.add_edge("drafter", "critic")
    g.add_conditional_edges("critic", route_critic)
    return g.compile()


graph = build_graph()
