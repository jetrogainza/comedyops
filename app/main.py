from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
load_dotenv()

from app.llm_factory import get_llm
from app.eval_logger import log_generation, log_feedback


def _get_llm():
    """Lazily initialize the configured LLM so app startup doesn't fail on missing env."""
    try:
        return get_llm()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "LLM provider is not configured correctly. "
                "Set OPENAI_API_KEY and keep LLM_PROVIDER=openai."
            ),
        ) from e

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
ALLOWED_TASKS = {"rewrite_premise", "rewrite_persona"}


app = FastAPI(title="ComedyOps", version="0.1.0")

# Allow local frontends (Vite, simple static servers) to call the API during development.
# In production you should lock this down to your real domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend (static files) at /ui.
# We avoid mounting at / because your API endpoints live at /goal, /feedback, etc.
# Mounting at / would capture everything and break the API routes.
app.mount("/ui", StaticFiles(directory="frontend", html=True), name="frontend")


def _reported_model_name() -> str:
    return OPENAI_MODEL


class RewritePremiseRequest(BaseModel):
    premise: str = Field(..., min_length=1)
    prompt_version: str = Field("v1", description="Prompt version to use")


class RewritePremiseResponse(BaseModel):
    rewritten: str
    model: str
    run_id: str

class RewritePersonaRequest(BaseModel):
    premise: str = Field(..., min_length=1)
    persona: str = Field(..., min_length=2, description="Persona style to emulate (e.g. 'deadpan British insult comic').")
    prompt_version: str = Field("v1", description="Prompt version to use")


class RewritePersonaResponse(BaseModel):
    rewritten: str
    persona: str
    model: str
    run_id: str


class FeedbackRequest(BaseModel):
    run_id: str = Field(..., min_length=8)
    human_rating: int = Field(..., ge=1, le=5, description="1=bad, 5=great")
    would_use_on_stage: bool = Field(False)
    notes: str | None = Field(None, description="Optional notes (why it worked / didn’t)")


class FeedbackResponse(BaseModel):
    status: str


class RoutedGoal(BaseModel):
    task: str = Field(..., description="Target task to execute")
    premise: str
    persona: str | None = None
    prompt_version: str = "v1"


class GoalRequest(BaseModel):
    goal: str = Field(..., min_length=5)


class GoalResponse(BaseModel):
    rewritten: str
    task: str
    persona: str | None
    model: str
    run_id: str


def load_prompt(template_name: str, version: str) -> str:
    """Load a prompt template from disk."""

    prompt_path = Path("prompts") / template_name / f"{version}.txt"
    if not prompt_path.exists():
        raise HTTPException(status_code=500, detail=f"Prompt not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8")


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

def _extract_json_object(text: str) -> str:
    """Best-effort extraction of a JSON object from model output."""
    if not text or not text.strip():
        raise HTTPException(status_code=502, detail="Router returned empty response")

    stripped = text.strip()

    # Handle fenced code blocks like ```json ... ```
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()

    # If still not a bare JSON object, try to slice the first {...}
    if not (stripped.startswith("{") and stripped.endswith("}")):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise HTTPException(status_code=502, detail="Router did not return JSON")
        stripped = stripped[start : end + 1].strip()

    return stripped

def _parse_router_json(raw: str) -> dict:
    try:
        return json.loads(_extract_json_object(raw))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Invalid router JSON: {e}")

def route_goal(user_goal: str) -> RoutedGoal:
    prompt_template = load_prompt("goal_router", "v1")

    router_prompt = (
        f"{prompt_template}\n\n"
        f"User goal:\n{user_goal}\n\n"
        "Decision:"
    )

    raw = _get_llm().generate(router_prompt)
    try:
        data = _parse_router_json(raw)
    except HTTPException:
        # Retry once with a stricter instruction if the model responded with non-JSON.
        retry_prompt = (
            f"{router_prompt}\n\n"
            "Return ONLY the JSON object. No prose, no markdown, no code fences."
        )
        raw_retry = _get_llm().generate(retry_prompt)
        data = _parse_router_json(raw_retry)

    task = data.get("task")
    if task not in ALLOWED_TASKS:
        raise HTTPException(status_code=502, detail=f"Unsupported task: {task}")

    return RoutedGoal(**data)


# Root redirect so http://localhost:8000/ opens the UI.
@app.get("/")
def root_redirect():
    return RedirectResponse(url="/ui")


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

        raw = _get_llm().generate(agent_prompt)
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

            run_id = log_generation(
                {
                    "task": "rewrite_premise",
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
                run_id=run_id,
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

    run_id = log_generation(
        {
            "task": "rewrite_premise",
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
        run_id=run_id,
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

        raw = _get_llm().generate(agent_prompt)

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

            run_id = log_generation(
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
                run_id=run_id,
            )

    # Fallback: if no explicit final, return best we got.
    if not state["best_text"]:
        raise HTTPException(status_code=502, detail="Agent failed to produce a result")

    run_id = log_generation(
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
        run_id=run_id,
    )

@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    # Store feedback in a separate append-only file.
    log_feedback(
        {
            "run_id": req.run_id,
            "human_rating": req.human_rating,
            "would_use_on_stage": req.would_use_on_stage,
            "notes": req.notes,
        }
    )

    return FeedbackResponse(status="ok")

@app.post("/goal", response_model=GoalResponse)
def execute_goal(req: GoalRequest) -> GoalResponse:
    routed = route_goal(req.goal)

    if routed.task == "rewrite_premise":
        result = rewrite_premise(
            RewritePremiseRequest(
                premise=routed.premise,
                prompt_version=routed.prompt_version,
            )
        )

        return GoalResponse(
            rewritten=result.rewritten,
            task=routed.task,
            persona=None,
            model=result.model,
            run_id=result.run_id,
        )

    if routed.task == "rewrite_persona":
        result = rewrite_persona(
            RewritePersonaRequest(
                premise=routed.premise,
                persona=routed.persona or "generic stand-up comic",
                prompt_version=routed.prompt_version,
            )
        )

        return GoalResponse(
            rewritten=result.rewritten,
            task=routed.task,
            persona=routed.persona,
            model=result.model,
            run_id=result.run_id,
        )

    raise HTTPException(status_code=500, detail="Unhandled routed task")
