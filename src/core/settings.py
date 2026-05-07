from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.local", env_file_encoding="utf-8")

    ANTHROPIC_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr

    CLAUDE_MODEL: str = "claude-haiku-4-5"
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    EMBEDDING_MODEL: str = "models/gemini-embedding-2"
    CHROMA_PATH: str = "data/chroma_db"


settings = Settings()  # type: ignore[call-arg]
