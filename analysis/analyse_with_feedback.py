import json
from pathlib import Path

import pandas as pd

EVAL_PATH = Path("eval_logs/eval_logs.jsonl")
FEEDBACK_PATH = Path("eval_logs/feedback.jsonl")

# Load generation logs
eval_rows = []
with EVAL_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        eval_rows.append(json.loads(line))

eval_df = pd.DataFrame(eval_rows)

# Load feedback logs
feedback_rows = []
with FEEDBACK_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        feedback_rows.append(json.loads(line))

feedback_df = pd.DataFrame(feedback_rows)

# Join on run_id
df = eval_df.merge(feedback_df, on="run_id", how="left")

print(df.head())
print("\nRow count:", len(df))

print("\nCorrelation: chosen_score vs human_rating")
print(df[["chosen_score", "human_rating"]].corr())

print("\nAverage human rating by prompt version:")
print(df.groupby("prompt_version")["human_rating"].mean())

print("\nWould use on stage (by persona):")
print(df.groupby("persona")["would_use_on_stage"].mean())