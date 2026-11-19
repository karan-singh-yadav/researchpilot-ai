"""Embedding providers. All return L2-normalised float32 matrices, so dot product == cosine.

* hash                 - deterministic, offline, no downloads. A signed feature-hashing bag of
                         words+bigrams: it captures LEXICAL overlap, NOT meaning. Use it for
                         demos/tests; use a real model for semantic retrieval.
* openai               - any OpenAI-compatible /embeddings endpoint (OpenAI, vLLM, Ollama, ...).
* sentence-transformers - local model (optional dependency; downloads weights on first use).
"""

import hashlib
import math
import re
import time
from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from typing import Protocol

import httpx
import numpy as np

from app.config import Settings


class EmbeddingError(Exception):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class Embedder(Protocol):
    name: str  # identifies the vector space; stored with every document

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be but by for from has have in is it its of on or that the this to was "
    "were will with".split()
)


@lru_cache(maxsize=200_000)
def _hash(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hash-{dim}-v1"

    def _vector(self, text: str) -> np.ndarray:
        toks = [t for t in _TOKEN.findall(text.lower()) if len(t) > 1]
        feats: Counter[str] = Counter(t for t in toks if t not in _STOP)
        for a, b in zip(toks, toks[1:], strict=False):
            if a not in _STOP or b not in _STOP:
                feats[f"{a}_{b}"] += 0.5  # bigrams carry a little word-order signal
        v = np.zeros(self.dim, dtype=np.float32)
        for feat, count in feats.items():
            h = _hash(feat)
            v[h % self.dim] += (1.0 if (h >> 63) & 1 else -1.0) * (1.0 + math.log(count + 0.5))
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack([self._vector(t) for t in texts]))


class OpenAIEmbedder:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        batch_size: int = 64,
        timeout: float = 60.0,
        retries: int = 2,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model:
            raise EmbeddingError("EMBEDDING_MODEL must be set for the openai embedding provider")
        self.name = f"openai:{model}"
        self.model, self._key = model, api_key
        self._url = base_url.rstrip("/") + "/embeddings"
        self.batch_size, self.retries, self._sleep = batch_size, retries, sleep
        self._client = client or httpx.Client(timeout=timeout)

    def _post(self, batch: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        for attempt in range(self.retries + 1):
            try:
                r = self._client.post(
                    self._url, json={"model": self.model, "input": batch}, headers=headers
                )
            except httpx.TimeoutException:
                err = EmbeddingError("embedding request timed out", retryable=True)
            except httpx.HTTPError:
                err = EmbeddingError("embedding request failed (network error)", retryable=True)
            else:
                if r.status_code == 429 or r.status_code >= 500:
                    err = EmbeddingError(
                        f"embedding provider returned HTTP {r.status_code}", retryable=True
                    )
                elif r.status_code >= 400:
                    raise EmbeddingError(
                        f"embedding provider rejected the request (HTTP {r.status_code})"
                    )
                else:
                    try:
                        rows = sorted(r.json()["data"], key=lambda d: d["index"])
                        vecs = [d["embedding"] for d in rows]
                    except (ValueError, KeyError, TypeError):
                        raise EmbeddingError(
                            "embedding provider returned a malformed response"
                        ) from None
                    if len(vecs) != len(batch):
                        raise EmbeddingError(
                            "embedding provider returned the wrong number of vectors"
                        )
                    return vecs
            if not err.retryable or attempt == self.retries:
                raise err
            self._sleep(min(0.5 * 2**attempt, 8.0))
        raise AssertionError("unreachable")  # pragma: no cover

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out += self._post(texts[i : i + self.batch_size])
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        if len({len(v) for v in out}) != 1:
            raise EmbeddingError("embedding provider returned vectors of different sizes")
        return _normalise(np.array(out))


class SentenceTransformerEmbedder:
    def __init__(self, model: str, batch_size: int = 32) -> None:
        if not model:
            raise EmbeddingError("EMBEDDING_MODEL must be set for sentence-transformers")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers is not installed (pip install sentence-transformers)"
            ) from None
        self.name = f"st:{model}"
        self._model = SentenceTransformer(model)
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalise(
            self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True)
        )


def build_embedder(s: Settings) -> Embedder:
    p = s.embedding_provider.lower()
    if p == "hash":
        return HashingEmbedder(s.embedding_dim)
    if p == "openai":
        if (
            not s.embedding_api_key
            and s.embedding_base_url.rstrip("/") == "https://api.openai.com/v1"
        ):
            raise EmbeddingError("EMBEDDING_API_KEY must be set for the openai embedding provider")
        return OpenAIEmbedder(
            model=s.embedding_model,
            api_key=s.embedding_api_key,
            base_url=s.embedding_base_url,
            batch_size=s.embedding_batch_size,
        )
    if p in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder(s.embedding_model)
    raise EmbeddingError(f"Unknown EMBEDDING_PROVIDER '{s.embedding_provider}'")
