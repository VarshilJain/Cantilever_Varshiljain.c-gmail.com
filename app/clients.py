from functools import lru_cache

from openai import OpenAI

from app.config import get_settings


@lru_cache
def get_llm_client() -> OpenAI:
    settings = get_settings()
    if not settings.is_llm_configured():
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Set it in .env locally or in Vercel → Settings → Environment Variables."
        )
    return OpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
    )


def get_openai_embedding_client() -> OpenAI:
    settings = get_settings()
    if not settings.openai_api_key.strip():
        raise RuntimeError(
            "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai. "
            "Or set EMBEDDING_PROVIDER=fastembed to use local embeddings (no extra key)."
        )
    return OpenAI(api_key=settings.openai_api_key)
