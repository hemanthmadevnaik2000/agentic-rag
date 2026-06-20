from __future__ import annotations

from app.kb.embeddings.base import Embeddings


class OpenAIEmbeddings(Embeddings):
    def __init__(self, model_name: str, api_key: str, dim: int) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
