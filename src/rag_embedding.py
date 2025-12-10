# rag_embedding.py

from typing import Dict, List, Optional

import openai as openai_pkg
from chromadb.api.types import EmbeddingFunction
from openai import OpenAI

DEFAULT_MODEL_PRIORITY = [
    "text-embedding-3-large",
    "text-embedding-3-small",
]


class EmbeddingModelUnavailable(RuntimeError):
    """Raised when none of the candidate embedding models can be used."""


def _dedupe_preserve_order(models: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered


class OpenAIEmbeddingFunction(EmbeddingFunction[List[str]]):
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    @staticmethod
    def name() -> str:
        return "openai"

    def __call__(self, input: List[str]):
        try:
            response = self.client.embeddings.create(model=self.model, input=input)
            return [item.embedding for item in response.data]
        except Exception as exc:
            raise EmbeddingModelUnavailable(
                f"Failed to generate embeddings with '{self.model}': {exc}"
            ) from exc


class AutoEmbeddingFunction(OpenAIEmbeddingFunction):
    def __init__(self, client: OpenAI, configured_model: Optional[str] = None):
        self.client = client
        initial_list = ([configured_model] if configured_model else []) + DEFAULT_MODEL_PRIORITY
        self._model_queue = _dedupe_preserve_order(initial_list)

        if not self._model_queue:
            raise EmbeddingModelUnavailable(
                "No embedding model candidates configured. Set OPENAI_EMBED_MODEL."
            )

        self._current_model = None
        self._embedding_fn = None
        self._use_next_model(initial=True)

    def _use_next_model(self, initial: bool = False):
        if not self._model_queue:
            raise EmbeddingModelUnavailable(
                "No embedding model is available. Tried all candidates."
            )

        next_model = self._model_queue.pop(0)
        self._embedding_fn = OpenAIEmbeddingFunction(self.client, next_model)
        self._current_model = next_model

        if not initial:
            print(f"Switching to embedding model '{next_model}'.")

    @staticmethod
    def name() -> str:
        return "openai"

    def __call__(self, input: List[str]):
        while True:
            try:
                return self._embedding_fn(input)
            except EmbeddingModelUnavailable:
                if not self._model_queue:
                    raise
                self._use_next_model()


def create_embedding_function(env: Optional[Dict[str, str]] = None) -> AutoEmbeddingFunction:
    embedding_model = None
    api_key = openai_pkg.api_key

    if isinstance(env, dict):
        embedding_model = env.get("OPENAI_EMBED_MODEL") or embedding_model
        api_key = env.get("OPENAI_API_KEY") or api_key

    if not api_key:
        raise EmbeddingModelUnavailable("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    return AutoEmbeddingFunction(client, configured_model=embedding_model)
