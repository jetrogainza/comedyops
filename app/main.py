from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.llm_factory import get_llm
from app.eval_logger import log_generation

llm = get_llm()

# Env vars let us switch model/host without changing code.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
REQUEST_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "60"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

app = FastAPI(title="ComedyOps", version="0.1.0")


def _reported_model_name() -> str:
    if LLM_PROVIDER == "openai":
        return OPENAI_MODEL
    return OLLAMA_MODEL


class RewritePremiseRequest(BaseModel):
    premise: str = Field(..., min_length=1)
    prompt_version: str = Field("v1", description="Prompt version to use")


class RewritePremiseResponse(BaseModel):
    rewritten: str
    model: str

class RewritePersonaRequest(BaseModel):
    premise: str = Field(..., min_length=1)
    persona: str = Field(..., min_length=2, description="Persona style to emulate (e.g. 'deadpan British insult comic').")
    prompt_version: str = Field("v1", description="Prompt version to use")


class RewritePersonaResponse(BaseModel):
    rewritten: str
    persona: str
    model: str

def load_prompt(template_name: str, version: str) -> str:
    """Load a prompt template from disk."""

    prompt_path = Path("prompts") / template_name / f"{version}.txt"
    if not prompt_path.exists():
        raise HTTPException(status_code=500, detail=f"Prompt not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8")


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
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned HTTP {resp.status_code}: {resp.text}",
        )

    data = resp.json()
    text = (data.get("response") or "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="Ollama returned an empty response")

    return text


# -----------------------------
# Tool schemas (JSON-based)
# -----------------------------
# The model is only allowed to request ONE of these actions.
# Our code validates and executes them.

ALLOWED_TOOLS = {"rewrite", "score", "final"}


def parse_action(json_text: str) -> dict:
    """Parse and validate a model-proposed action.

    Expected schema:
    {
      "action": "rewrite" | "score" | "final",
      "text": "..."
    }
    """

    try:
        data = json.loads(json_text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from model: {e}")

    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Model action must be a JSON object")

    action = data.get("action")
    text = data.get("text")

    if action not in ALLOWED_TOOLS:
        raise HTTPException(status_code=502, detail=f"Disallowed action: {action}")

    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=502, detail="Action 'text' must be a non-empty string")

    return {"action": action, "text": text}


def score_joke(text: str) -> int:
    """Very naive heuristic scorer.

    Higher score = better.
    This is intentionally simple and replaceable.
    """

    score = 0

    # Shorter tends to be punchier
    if len(text) < 180:
        score += 1

    # Presence of contrast words often helps comedy
    contrast_words = ["but", "however", "suddenly", "until"]
    if any(w in text.lower() for w in contrast_words):
        score += 1

    # Quotes often signal a punch or observation
    if "'" in text or '"' in text:
        score += 1

    return score


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rewrite_premise", response_model=RewritePremiseResponse)
def rewrite_premise(req: RewritePremiseRequest) -> RewritePremiseResponse:
    # Agent loop configuration
    MAX_STEPS = 5

    prompt_template = load_prompt("rewrite_premise", req.prompt_version)
    base_prompt = prompt_template.replace("{{premise}}", req.premise)

    system_instructions = (
        "You are an agent that must respond ONLY in JSON.\n"
        "Choose exactly one action per step from: rewrite, score, final.\n"
        "Schema: {\"action\": <action>, \"text\": <string>}\n"
        "Rules:\n"
        "- rewrite: propose a rewritten premise\n"
        "- score: request scoring of the provided text\n"
        "- final: provide the final chosen rewrite and stop\n"
        "Do not include any other text outside JSON."
    )

    state = {
        "best_text": "",
        "best_score": -1,
        "has_candidate": False,
    }

    for _ in range(MAX_STEPS):
        agent_prompt = (
            f"{system_instructions}\n\n"
            f"Premise prompt:\n{base_prompt}\n\n"
            f"Current best score: {state['best_score']}\n"
            f"Current best text: {state['best_text']}\n"
            "Next action:"
        )

        raw = llm.generate(agent_prompt)
        action = parse_action(raw)

        if action["action"] in {"rewrite", "score"}:
            candidate = action["text"]
            score = score_joke(candidate)
            state["has_candidate"] = True
            if score > state["best_score"]:
                state["best_score"] = score
                state["best_text"] = candidate

        elif action["action"] == "final":
            if not state["has_candidate"]:
                # Ignore premature final and continue the loop
                continue

            print(
                {
                    "provider": LLM_PROVIDER,
                    "prompt_version": req.prompt_version,
                    "best_score": state["best_score"],
                }
            )

            log_generation(
                {
                    "premise": req.premise,
                    "prompt_version": req.prompt_version,
                    "model_provider": LLM_PROVIDER,
                    "model_name": _reported_model_name(),
                    "chosen_text": action["text"],
                    "chosen_score": state["best_score"],
                }
            )
            
            return RewritePremiseResponse(
                rewritten=action["text"],
                model=f"{_reported_model_name()}:rewrite_premise:{req.prompt_version}:agent_v2",
            )

    # Fallback if agent never calls 'final'
    if not state["best_text"]:
        raise HTTPException(status_code=502, detail="Agent failed to produce a final result")
    
    print(
        {
            "provider": LLM_PROVIDER,
            "prompt_version": req.prompt_version,
            "best_score": state["best_score"],
        }
    )

    log_generation(
        {
            "premise": req.premise,
            "prompt_version": req.prompt_version,
            "model_provider": LLM_PROVIDER,
            "model_name": _reported_model_name(),
            "chosen_text": action["text"],
            "chosen_score": state["best_score"],
        }
    )

    return RewritePremiseResponse(
        rewritten=state["best_text"],
        model=f"{_reported_model_name()}:rewrite_premise:{req.prompt_version}:agent_v2",
    )

@app.post("/rewrite_persona", response_model=RewritePersonaResponse)
def rewrite_persona(req: RewritePersonaRequest) -> RewritePersonaResponse:
    MAX_STEPS = 5

    prompt_template = load_prompt("rewrite_persona", req.prompt_version)
    base_prompt = (
        prompt_template
        .replace("{{premise}}", req.premise)
        .replace("{{persona}}", req.persona)
    )

    # The agent must choose one action each step.
    # We add a 'critique' step so the model can explicitly check persona alignment.
    allowed_tools = {"rewrite", "critique", "final"}

    system_instructions = (
        "You are an agent that must respond ONLY in JSON.\n"
        "Choose exactly one action per step from: rewrite, critique, final.\n"
        "Schema: {\"action\": <action>, \"text\": <string>}\n"
        "Rules:\n"
        "- rewrite: produce a persona-styled rewrite of the premise\n"
        "- critique: critique the most recent rewrite for persona match + punchiness; suggest improvements\n"
        "- final: output the best rewrite and stop\n"
        "Do not include any other text outside JSON."
    )

    # Local state to keep track of the best candidate we have seen.
    state = {
        "best_text": "",
        "best_score": -1,
        "latest_text": "",
        "has_candidate": False,
        "latest_critique": "",
    }

    def score_persona(text: str, persona: str) -> int:
        """
        Simple heuristic score:
        - short/punchy
        - contains an observational/comedic 'turn'
        - uses cues that often correlate with 'voice' (not perfect, but useful for iteration)
        """
        score = 0
        if len(text) < 180:
            score += 1
        if any(w in text.lower() for w in ["but", "however", "until", "then"]):
            score += 1
        if "'" in text or '"' in text:
            score += 1
        # tiny nudge: if persona mentions 'deadpan'/'insult', look for sharper phrasing
        if any(w in persona.lower() for w in ["deadpan", "insult", "roast"]) and any(w in text.lower() for w in ["mate", "of course", "obviously", "right"]):
            score += 1
        return score

    for _ in range(MAX_STEPS):
        agent_prompt = (
            f"{system_instructions}\n\n"
            f"Task prompt:\n{base_prompt}\n\n"
            f"Persona: {req.persona}\n"
            f"Latest rewrite: {state['latest_text']}\n"
            f"Latest critique: {state['latest_critique']}\n"
            f"Best score so far: {state['best_score']}\n"
            f"Best text so far: {state['best_text']}\n"
            "Next action:"
        )

        raw = llm.generate(agent_prompt)

        # Parse JSON action, but with local tool set for this endpoint.
        try:
            data = json.loads(raw)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Invalid JSON from model: {e}")

        action = data.get("action")
        text = data.get("text")

        if action not in allowed_tools:
            raise HTTPException(status_code=502, detail=f"Disallowed action: {action}")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(status_code=502, detail="Action 'text' must be a non-empty string")

        if action == "rewrite":
            candidate = text.strip()
            state["latest_text"] = candidate
            state["has_candidate"] = True

            s = score_persona(candidate, req.persona)
            if s > state["best_score"]:
                state["best_score"] = s
                state["best_text"] = candidate

        elif action == "critique":
            # We store critique to feed back into the next loop iteration.
            state["latest_critique"] = text.strip()

        elif action == "final":
            # Guardrail: ignore premature final until we have at least one candidate.
            if not state["has_candidate"]:
                continue

            chosen = state["best_text"] or state["latest_text"] or text.strip()

            log_generation(
                {
                    "task": "rewrite_persona",
                    "premise": req.premise,
                    "persona": req.persona,
                    "prompt_version": req.prompt_version,
                    "model_provider": LLM_PROVIDER,
                    "model_name": _reported_model_name(),
                    "chosen_text": chosen,
                    "chosen_score": state["best_score"],
                }
            )

            return RewritePersonaResponse(
                rewritten=chosen,
                persona=req.persona,
                model=f"{_reported_model_name()}:rewrite_persona:{req.prompt_version}:agent_v1",
            )

    # Fallback: if no explicit final, return best we got.
    if not state["best_text"]:
        raise HTTPException(status_code=502, detail="Agent failed to produce a result")

    log_generation(
        {
            "task": "rewrite_persona",
            "premise": req.premise,
            "persona": req.persona,
            "prompt_version": req.prompt_version,
            "model_provider": LLM_PROVIDER,
            "model_name": _reported_model_name(),
            "chosen_text": state["best_text"],
            "chosen_score": state["best_score"],
        }
    )

    return RewritePersonaResponse(
        rewritten=state["best_text"],
        persona=req.persona,
        model=f"{_reported_model_name()}:rewrite_persona:{req.prompt_version}:agent_v1",
    )
