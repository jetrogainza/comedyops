import os

from app.llm import LLM


def get_llm() -> LLM:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider and provider != "openai":
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider}'. This project now supports only 'openai'."
        )

    from app.providers.openai import OpenAILLM
    return OpenAILLM()
