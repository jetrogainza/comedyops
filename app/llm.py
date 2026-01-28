from abc import ABC, abstractmethod


class LLM(ABC):
    """Abstract interface for any LLM provider."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError