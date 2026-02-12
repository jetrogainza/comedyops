# ComedyOps 🎭🤖

A hands-on ML engineering project that builds an **LLM-powered service** for stand-up writing workflows.

Uses the OpenAI API for model inference and is designed to practise **real ML engineering**, not just notebooks.

---

## What it does (current)
- `POST /rewrite_premise`  
  Rewrites a comedy premise into a tighter, stage-ready setup using an OpenAI model.
- `GET /health`  
  Basic service health check.

---

## Why this project exists

ComedyOps exists to practise **ML Engineering end-to-end**, not model theory:

- API deployment patterns (FastAPI)
- Containerisation (Docker)
- LLM provider integration (OpenAI API)
- Dependency & environment management (uv + pyproject.toml)
- CI (GitHub Actions)
- Testing discipline (pytest)
- Prompt/version management (coming next)
- Monitoring + evaluation (coming next)

This is intentionally **hands-on and slightly painful**, like real work.

---

## Tech stack

- Python 3.11
- FastAPI + Uvicorn
- OpenAI API
- Docker + Docker Compose
- uv (Python env + dependency manager)
- pytest (tests)
- ruff (linting)

---

## 🧠 Mental model (READ THIS EVERY TIME)

You are dealing with **layers**.  
Never mix responsibilities between them.

### 1️⃣ System tools (installed once)
- **Homebrew** → installs system-level tools
- **Docker Desktop** → runs containers
- **uv** → Python environments & dependencies

⬇️

### 2️⃣ Python environment (per project)
- `.venv/` created by `uv`
- Dependencies defined in **`pyproject.toml`**
- Installed via `uv pip install -e ".[dev]"`

⬇️

### 3️⃣ Application runtime
- FastAPI app
- Runs either:
  - directly via `uvicorn`
  - or inside Docker via `docker compose`

**Rules of thumb**
- ❌ Don’t invent `requirements.txt`
- ❌ Don’t mix `pip install` and `uv`
- ❌ Don’t install Python libs with `brew`
- ✅ `brew` installs tools
- ✅ `uv` installs Python deps
- ✅ `pyproject.toml` is the single source of truth

---

## Prerequisites (fresh machine friendly)

### 0️⃣ Git
```bash
git --version
```

---

### 1️⃣ Install Homebrew (macOS)

```bash
brew --version
```

If not installed:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

On Apple Silicon:
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Verify:
```bash
brew --version
```

---

### 2️⃣ Install uv

```bash
brew install uv
uv --version
```

---

### 3️⃣ Install Docker Desktop (REQUIRED)

Docker is needed to run the API in containers and simulate production.

Install:
https://www.docker.com/products/docker-desktop/

Verify:
```bash
docker --version
docker compose version
```

---

### 4️⃣ Configure OpenAI API key

Create a `.env` file and set:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1
```

---

## Project setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/comedyops.git
cd comedyops
```

---

### 2️⃣ Create the virtual environment

```bash
uv venv
```

---

### 3️⃣ Activate it

```bash
source .venv/bin/activate
```

---

### 4️⃣ Install dependencies

```bash
uv pip install -e ".[dev]"
```

---

### 5️⃣ Sanity checks

```bash
which python
python --version
```

---

## Run locally (no Docker)

```bash
uvicorn app.main:app --reload
```

Test:
```bash
curl -X POST http://127.0.0.1:8000/rewrite_premise \
  -H "Content-Type: application/json" \
  -d '{"premise":"I am a latino in Australia.","prompt_version":"v2"}'
```

---

## Run with Docker (recommended)

```bash
docker compose up --build
```

Or manually:
```bash
docker run --rm \
  -p 8000:8000 \
  -e LLM_PROVIDER=openai \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OPENAI_MODEL=gpt-4.1 \
  comedyops:latest
```

---

# How to run the front end.
In a terminal, activate the venv (`source .venv/bin/activate` in case you forgot), and then, either spin up FastAPI via uvicorn (for quick testing) or otherwise complie docker (for  production).

## Using FastAPI
in the terminal, do `uvicorn app.main:app --reload`. this:
- Loads app/main.py
- Creates the FastAPI() app
- Loads your .env (if you added load_dotenv())
- Mounts the /ui static frontend
- Starts the API server on `http://127.0.0.1:8000`

On your browser, go to `http://localhost:8000/` and you will see ComedyOps.

## Using Docker
If using Docker, first stop uvicorn (if using it) doing `CTRL` + `C`. Then execute `docker compose up --build` and once loaded, go to the same website `http://localhost:8000/`.


## Final reminder

This repo is a **learning system**, not a polished product.
Breakage is expected. Debugging is the skill.
