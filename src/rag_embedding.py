# rag_embedding.py

from typing import List
from openai import OpenAI, OpenAIError


class EmbeddingModelUnavailable(Exception):
    pass


class OpenAIEmbeddingFunction:
    """
    Wrapper that makes an OpenAI embedding model compatible
    with ChromaDB's EmbeddingFunction interface.
    """

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def __call__(self, input: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=input
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            raise EmbeddingModelUnavailable(
                f"Failed to generate embeddings with '{self.model}': {exc}"
            ) from exc


def create_embedding_function(env=None) -> OpenAIEmbeddingFunction:
    """
    Return a Chroma-compatible embedding function object.
    """

    # Load from env dict if provided
    embedding_model = None
    api_key = None

    if isinstance(env, dict):
        embedding_model = env.get("EMBEDDING_MODEL")
        api_key = env.get("OPENAI_API_KEY")

    # Default to the large embedding model
    if not embedding_model:
        embedding_model = "text-embedding-3-large"

    # Create client
    try:
        client = OpenAI(api_key=api_key)
        # test
        client.embeddings.create(
            model=embedding_model,
            input="test"
        )
    except Exception as exc:
        raise EmbeddingModelUnavailable(
            f"Embedding model '{embedding_model}' unavailable: {exc}"
        ) from exc

    # Return the wrapper object Chroma expects
    return OpenAIEmbeddingFunction(client, embedding_model)
