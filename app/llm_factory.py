import os

from app.llm import LLM


def get_llm() -> LLM:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        from app.providers.ollama import OllamaLLM
        return OllamaLLM()

    elif provider == "openai":
        from app.providers.openai import OpenAILLM
        return OpenAILLM()

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")