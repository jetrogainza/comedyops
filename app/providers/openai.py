import os

from fastapi import HTTPException
from openai import OpenAI

from app.llm import LLM


class OpenAILLM(LLM):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1")

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        text = (response.output_text or "").strip()
        if text:
            return text

        # Fallback: dig through output content to find text or refusal.
        outputs = getattr(response, "output", None) or []
        for output in outputs:
            contents = getattr(output, "content", None) or []
            for content in contents:
                ctype = getattr(content, "type", None)
                if ctype == "output_text":
                    ctext = (getattr(content, "text", None) or "").strip()
                    if ctext:
                        return ctext
                if ctype == "refusal":
                    refusal = (getattr(content, "refusal", None) or "").strip()
                    if refusal:
                        raise HTTPException(status_code=502, detail=f"Model refused: {refusal}")

        raise HTTPException(status_code=502, detail="OpenAI returned empty response")
