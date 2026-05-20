from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.settings import settings


def clean_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return "".join(text_parts)
    return str(content)


def get_llm(model_name: str):
    if model_name == "claude":
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set in .env.local")
        return ChatAnthropic(
            model_name=settings.CLAUDE_MODEL,
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=None,
            stop=None,
        )
    if model_name == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY.get_secret_value(),
        )
    raise ValueError(
        f"Unsupported model_name: {model_name!r}. Use 'claude' or 'gemini'."
    )
