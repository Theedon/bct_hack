from langchain.agents import create_agent

from src.core.llm import get_llm

llm = get_llm("claude")
agent = create_agent(model=llm, tools=[])
