import json
from datetime import datetime
from pathlib import Path
from typing import Any

# where we store the evaluation logs.
EVAL_LOG_PATH = Path("eval_logs") / "eval_logs.jsonl"


def log_generation(record: dict[str, Any]) -> None:
    """
    Append a single generation record as JSONL.

    JSONL = one JSON object per line.
    Easy to stream, parse, and analyse later.
    """
    record_with_time = {
        "timestamp": datetime.utcnow().isoformat(),
        **record,
    }

    with EVAL_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record_with_time) + "\n")