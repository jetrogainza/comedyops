import json
from pathlib import Path

import pandas as pd

LOG_PATH = Path("eval_logs/eval_logs.jsonl")

rows = []
with LOG_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

print(df.head())
print("\nColumns:")
print(df.columns)
print("\nRow count:", len(df))

print("\nAverage score by prompt version:")
print(df.groupby("prompt_version")["chosen_score"].mean())

print("\nAverage score by model:")
print(df.groupby("model_name")["chosen_score"].mean())