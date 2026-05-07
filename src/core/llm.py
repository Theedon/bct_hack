from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm(model_name: str):
    if model_name == "claude":
        return ChatAnthropic(model_name="claude-haiku-4-5", timeout=None, stop=None)
    if model_name == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    raise ValueError(
        f"Unsupported model_name: {model_name!r}. Use 'claude' or 'gemini'."
    )
