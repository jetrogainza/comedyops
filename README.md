# ComedyOps 🎭🤖

A hands-on ML engineering project that builds a local LLM-powered service for stand-up writing workflows.
Runs fully on your machine using Ollama (no paid cloud required).

## What it does (current)
- `POST /rewrite_premise`: rewrites a comedy premise into a tighter, stage-ready setup using a local model via Ollama.
- `GET /health`: basic service health check.

## Why this project exists
To learn ML Engineering end-to-end in a practical way:
- API deployment patterns (FastAPI)
- containerisation (Docker)
- CI (GitHub Actions)
- testing discipline (pytest)
- LLM integration + prompt/version management (coming next)
- monitoring + evaluation (coming next)

## Tech stack
- Python 3.11
- FastAPI + Uvicorn
- Ollama (local model runtime)
- httpx (HTTP calls to Ollama)
- pytest (tests)
- ruff (linting)

## Prerequisites
- Python 3.11
- Ollama installed and running
- A local model pulled, e.g.:
  - `ollama pull llama3.2:3b`

## Quickstart (local)
From the repo root:

1) Ensure you have Ollama running — the service needs it:
```bash
ollama serve
```

2) Create + activate venv, install deps:
```bash
source .venv/bin/activate
uv pip install -e ".[dev]"
```

3) Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

This will start the server on http://127.0.0.1:8000, then you can run your curl command. An example to run:

```bash
curl -X POST http://127.0.0.1:8000/rewrite_premise \
  -H "Content-Type: application/json" \
  -d '{"premise":"I moved to Australia and everyone calls me mate.","prompt_version":"v2"}'
  ```

  ## EDIT:
  Now that we are usng Docker, this is the command to spin up the container:
  ```bash
  docker run --rm \
  -p 8000:8000 \
  -e LLM_PROVIDER=ollama \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  comedyops:latest
  ```


