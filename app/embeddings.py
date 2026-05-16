import numpy as np

from app.clients import get_openai_embedding_client
from app.config import get_settings

_fastembed_model = None


def _get_fastembed_model():
    global _fastembed_model
    if _fastembed_model is None:
        from fastembed import TextEmbedding

        settings = get_settings()
        _fastembed_model = TextEmbedding(model_name=settings.fastembed_model)
    return _fastembed_model


def _embed_fastembed(texts: list[str]) -> np.ndarray:
    model = _get_fastembed_model()
    vectors = list(model.embed(texts))
    return np.array(vectors, dtype=np.float64)


def _embed_openai(texts: list[str]) -> np.ndarray:
    settings = get_settings()
    client = get_openai_embedding_client()
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float64)


def embed_texts(texts: list[str]) -> np.ndarray:
    settings = get_settings()
    if settings.embedding_provider.lower() == "openai":
        return _embed_openai(texts)
    return _embed_fastembed(texts)


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query / (np.linalg.norm(query) + 1e-12)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return matrix_norms @ query_norm
