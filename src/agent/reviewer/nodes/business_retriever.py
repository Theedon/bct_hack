from src.agent.reviewer.state import AgentState
from src.core.vectorstore import get_business_vectorstore


def business_retriever(state: AgentState) -> dict:
    vectorstore = get_business_vectorstore()

    # We use a direct filter on business_id instead of a semantic search
    results = vectorstore.get(where={"business_id": state["business_id"]})

    # Check if the business exists in our store
    if results and results.get("documents") and len(results["documents"]) > 0:
        # Get the first match
        business_context = results["documents"][0]
    else:
        # Fallback if business isn't found
        business_context = ""

    return {"business_context": business_context}
