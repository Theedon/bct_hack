import requests
from langchain_core.embeddings import Embeddings

from src.core.settings import settings

_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiEmbeddings(Embeddings):
    """Calls the Gemini REST embedding endpoints directly."""

    def __init__(self) -> None:
        self._api_key = settings.GOOGLE_API_KEY.get_secret_value()
        self._model = settings.EMBEDDING_MODEL

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        url = f"{_BASE}/models/{self._model.split('/')[-1]}:batchEmbedContents?key={self._api_key}"
        payload = {
            "requests": [
                {"model": self._model, "content": {"parts": [{"text": t}]}}
                for t in texts
            ]
        }
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return [e["values"] for e in resp.json()["embeddings"]]

    def embed_query(self, text: str) -> list[float]:
        url = f"{_BASE}/{self._model}:embedContent?key={self._api_key}"
        resp = requests.post(
            url,
            json={"model": self._model, "content": {"parts": [{"text": text}]}},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]
