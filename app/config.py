from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Groq — LLM (chat, tool calling, agent)
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"

    # Embeddings — fastembed needs no key; openai needs OPENAI_API_KEY
    embedding_provider: str = "fastembed"  # fastembed | openai
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"

    top_n_results: int = 10
    top_n_for_agent: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"

    def is_llm_configured(self) -> bool:
        return bool(self.groq_api_key.strip())

    def is_embedding_configured(self) -> bool:
        if self.embedding_provider.lower() == "openai":
            return bool(self.openai_api_key.strip())
        return True

    def is_ready(self) -> bool:
        return self.is_llm_configured() and self.is_embedding_configured()


@lru_cache
def get_settings() -> Settings:
    return Settings()
