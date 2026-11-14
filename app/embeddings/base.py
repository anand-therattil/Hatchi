"""Embedding interface plus the real (Granite) and fake (tests) implementations."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GraniteEmbedder:
    """ibm-granite/granite-embedding-97m-multilingual-r2 via sentence-transformers (384-dim)."""

    _models: dict = {}  # loaded once per process and shared by every instance

    def __init__(self, model_name: str, query_prefix: str = "", doc_prefix: str = ""):
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix

    def _load(self):
        if self._model_name not in self._models:
            from sentence_transformers import SentenceTransformer  # lazy: heavy import

            self._models[self._model_name] = SentenceTransformer(self._model_name, device="cpu")
        return self._models[self._model_name]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self._doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_prefix + text])[0]


_STOPWORDS = frozenset(
    "a an the and or but if of to in on at for with about is are was were be been am i me my you your "
    "we our it its this that these those do does did can could should would will what why how when "
    "where who which not no so as by from up out me please".split()
)


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder.

    Texts that share meaningful words get a high cosine similarity and texts that share
    none get zero, which makes retrieval behaviour testable without downloading a model.
    """

    def __init__(self, dim: int = 512):
        self._dim = dim

    @staticmethod
    def _stem(token: str) -> str:
        for suffix in ("ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
        return token

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in re.findall(r"[a-z0-9ऀ-ॿ]+", text.lower()):
            if tok in _STOPWORDS:
                continue
            idx = int(hashlib.md5(self._stem(tok).encode()).hexdigest(), 16) % self._dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
