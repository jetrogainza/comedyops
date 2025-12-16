from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Env vars let us switch model/host without changing code.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
REQUEST_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "60"))

app = FastAPI(title="ComedyOps", version="0.1.0")


class RewritePremiseRequest(BaseModel):
    premise: str = Field(..., min_length=1, description="A stand-up premise or setup line")


class RewritePremiseResponse(BaseModel):
    rewritten: str
    model: str


def _ollama_generate(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Calls Ollama's HTTP API and returns only the generated text."""
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": DEFAULT_TEMPERATURE},
    }

    try:
        resp = httpx.post(url, json=payload, timeout=REQUEST_TIMEOUT_S)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach Ollama at {OLLAMA_HOST}. Is Ollama running? Error: {e}",
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Ollama returned HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    text = (data.get("response") or "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="Ollama returned an empty response")

    return text


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rewrite_premise", response_model=RewritePremiseResponse)
def rewrite_premise(req: RewritePremiseRequest) -> RewritePremiseResponse:
    """Rewrite a comedy premise into a tighter, stage-ready setup."""
    prompt = (
        "You are a stand-up comedy editor.\n"
        "Task: rewrite the premise into a tighter, funnier, stage-ready setup.\n"
        "Constraints:\n"
        "- Keep the original meaning\n"
        "- Keep it suitable for a general audience (no slurs)\n"
        "- 1 to 3 sentences\n"
        "- Australian/British English spelling\n"
        "Return only the rewritten text.\n\n"
        f"Premise: {req.premise}\n"
        "Rewrite:"
    )

    rewritten = _ollama_generate(prompt=prompt, model=OLLAMA_MODEL)
    return RewritePremiseResponse(rewritten=rewritten, model=OLLAMA_MODEL)