import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Logs directory is volume-mounted in docker-compose:
# ./eval_logs  ->  /app/eval_logs
EVAL_DIR = Path("eval_logs")
EVAL_DIR.mkdir(exist_ok=True)

EVAL_LOG_PATH = EVAL_DIR / "eval_logs.jsonl"
FEEDBACK_LOG_PATH = EVAL_DIR / "feedback.jsonl"


def log_generation(record: dict[str, Any]) -> str:
    """
    Append a generation event to eval_logs.jsonl and return a unique run_id.

    JSONL = one JSON object per line (easy to append, easy to analyse later).
    """
    run_id = str(uuid.uuid4())

    payload = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        **record,
    }

    with EVAL_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return run_id


def log_feedback(record: dict[str, Any]) -> None:
    """
    Append human feedback to feedback.jsonl.

    We keep feedback separate so generation logs remain immutable.
    """
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        **record,
    }

    with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")