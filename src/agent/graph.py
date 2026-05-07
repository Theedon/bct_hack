from langchain.agents import create_agent

from src.core.llm import get_llm

llm = get_llm("gemini")
agent = create_agent(model=llm, tools=[])
