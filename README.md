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

1) Create + activate venv, install deps:
```bash
source .venv/bin/activate
uv pip install -e ".[dev]"
