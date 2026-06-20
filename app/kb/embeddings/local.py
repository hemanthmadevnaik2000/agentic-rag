from __future__ import annotations

from app.kb.embeddings.base import Embeddings

# bge-*-en-v1.5 retrieval works best with this query-side instruction prefix.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self._query_instruction = (
            _BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode(
            [self._query_instruction + text], normalize_embeddings=True
        )
        return vec[0].tolist()
