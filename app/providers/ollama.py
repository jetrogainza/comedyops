import os
from typing import Any

import httpx
from fastapi import HTTPException

from app.llm import LLM


class OllamaLLM(LLM):
    def __init__(self):
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT_S", "60"))

    def generate(self, prompt: str) -> str:
        url = f"{self.host.rstrip('/')}/api/generate"

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        try:
            resp = httpx.post(url, json=payload, timeout=self.timeout)
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Could not reach Ollama at {self.host}. Error: {e}",
            )

        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Ollama returned HTTP {resp.status_code}: {resp.text}",
            )

        data = resp.json()
        text = (data.get("response") or "").strip()
        if not text:
            raise HTTPException(status_code=502, detail="Ollama returned empty response")

        return text