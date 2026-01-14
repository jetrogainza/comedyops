from __future__ import annotations

"""ComedyOps API (FastAPI)

This file is the entry point for your service.
- FastAPI turns normal Python functions into HTTP endpoints.
- We call Ollama (running locally) to generate text.
- We run a tiny "agent loop": generate several candidates, score them, pick the best.
"""

import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -----------------------------
# Configuration (via env vars)
# -----------------------------
# Environment variables let you change behaviour without editing code.
# This is handy for:
# - switching models
# - pointing to another Ollama host
# - tweaking generation params
#
# If the env var isn't set, we fall back to a default.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Temperature controls randomness:
# - lower = more deterministic / conservative
# - higher = more varied / creative
DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))

# HTTP timeout so requests don't hang forever if something is wrong.
REQUEST_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "60"))

# Create the FastAPI "app" object.
# Uvicorn runs this object to serve HTTP requests.
app = FastAPI(title="ComedyOps", version="0.1.0")


# ---------------------------------
# Request/Response schemas (Pydantic)
# ---------------------------------
# Pydantic models define the shape of JSON coming in (request) and going out (response).
# FastAPI uses these to:
# - validate inputs automatically
# - generate API docs automatically (Swagger/OpenAPI)

class RewritePremiseRequest(BaseModel):
    # "premise" must be a non-empty string (min_length=1)
    premise: str = Field(..., min_length=1, description="A stand-up premise/setup line")

    # Default prompt version is v1, but clients can request v2, v3, etc.
    prompt_version: str = Field("v1", description="Prompt version to use (e.g., v1, v2)")


class RewritePremiseResponse(BaseModel):
    # The rewritten result we want the client to use.
    rewritten: str

    # Provenance string (model + prompt version + agent version)
    model: str


# -----------------------------
# Prompt loading (versioned files)
# -----------------------------

def load_prompt(template_name: str, version: str) -> str:
    """Load a prompt template from disk.

    Example:
        load_prompt("rewrite_premise", "v1")

    Why this exists:
    - Prompts change often.
    - Keeping them in files makes changes easy to review, version, and roll back.
    """

    # Build a path like: prompts/rewrite_premise/v1.txt
    prompt_path = Path("prompts") / template_name / f"{version}.txt"

    if not prompt_path.exists():
        # HTTPException is a FastAPI-friendly way to return an error response.
        # status_code=500 means "server misconfigured" (missing file on disk).
        raise HTTPException(status_code=500, detail=f"Prompt not found: {prompt_path}")

    # Read the entire template as a string.
    return prompt_path.read_text(encoding="utf-8")


# -----------------------------
# Ollama client (HTTP call)
# -----------------------------

def _ollama_generate(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call Ollama's HTTP API and return the generated text.

    We keep this as a simple synchronous HTTP request for now.
    (Later we can make it async and/or stream tokens.)
    """

    # Ensure we don't get double slashes if OLLAMA_HOST ends with "/".
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"

    # Payload matches Ollama's API schema.
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # False means "return the full answer as one JSON response"
        "options": {
            "temperature": DEFAULT_TEMPERATURE,
        },
    }

    try:
        # Make the HTTP request to Ollama.
        resp = httpx.post(url, json=payload, timeout=REQUEST_TIMEOUT_S)
    except httpx.RequestError as e:
        # This happens if Ollama isn't running or the host is unreachable.
        raise HTTPException(
            status_code=503,  # 503 = "service unavailable"
            detail=f"Could not reach Ollama at {OLLAMA_HOST}. Is Ollama running? Error: {e}",
        )

    # If Ollama returns a non-200 response, we forward it as a "bad gateway".
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,  # 502 = "upstream service failed"
            detail=f"Ollama returned HTTP {resp.status_code}: {resp.text}",
        )

    data = resp.json()

    # Ollama returns the generated text in the "response" field.
    text = (data.get("response") or "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="Ollama returned an empty response")

    return text


# -----------------------------
# A tiny "tool": scoring function
# -----------------------------

def score_joke(text: str) -> int:
    """A tiny heuristic scorer.

    Why this is a "tool":
    - deterministic (same input => same score)
    - fast
    - testable

    It gives the agent loop a way to choose between candidates.
    """

    score = 0

    # Heuristic 1: shorter often feels punchier.
    if len(text) < 180:
        score += 1

    # Heuristic 2: contrast words can help create comedic turns.
    contrast_words = ["but", "however", "suddenly", "until"]
    if any(w in text.lower() for w in contrast_words):
        score += 1

    # Heuristic 3: quotes sometimes indicate a punch/observation.
    if "'" in text or '"' in text:
        score += 1

    return score


# ---------------------------------
# HTTP endpoints (FastAPI decorators)
# ---------------------------------
# The @app.get / @app.post lines are called "decorators".
# They "wrap" the function below and register it as an HTTP route.
#
# Example:
#   @app.get("/health")
# means:
#   "When a GET request hits /health, call the function named health()"


@app.get("/health")
def health():
    """Health endpoint.

    Used by humans, CI checks, Docker healthchecks, and monitoring.
    It's intentionally simple: if the server is up, return OK.
    """

    return {"status": "ok"}


@app.post("/rewrite_premise", response_model=RewritePremiseResponse)
def rewrite_premise(req: RewritePremiseRequest) -> RewritePremiseResponse:
    """Rewrite a premise using an agent loop.

    What happens here:
    1) Load a versioned prompt template from disk.
    2) Fill the {{premise}} placeholder.
    3) Generate multiple candidate rewrites using Ollama.
    4) Score candidates with a deterministic tool.
    5) Return the best candidate.

    FastAPI detail:
    - Because `req` is a Pydantic model, FastAPI will parse the incoming JSON into it.
    - `response_model=RewritePremiseResponse` means FastAPI validates/serialises output too.
    """

    # 1) Load the prompt template (v1/v2/...).
    prompt_template = load_prompt("rewrite_premise", req.prompt_version)

    # 2) Fill in the placeholder.
    prompt = prompt_template.replace("{{premise}}", req.premise)

    # 3) Generate candidates.
    candidates: list[str] = []
    for _ in range(5):
        text = _ollama_generate(prompt=prompt, model=OLLAMA_MODEL)
        candidates.append(text)

    # 4) Score and pick the best.
    scored = [(c, score_joke(c)) for c in candidates]
    best_text, best_score = max(scored, key=lambda x: x[1])

    # (best_score is currently unused in the response, but you might log it later.)
    _ = best_score

    # 5) Return a structured response.
    return RewritePremiseResponse(
        rewritten=best_text,
        model=f"{OLLAMA_MODEL}:rewrite_premise:{req.prompt_version}:agent_v1",
    )
