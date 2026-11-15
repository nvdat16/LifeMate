import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json
from collections import Counter

with open(DISTILLATION_DIR / "master_train_distilled.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip() and json.loads(line).get("state_match", False)]

mismatch_keys = Counter()
mismatch_tools = Counter()

for r in records:
    golds = r["gold_tool_calls"]
    preds = r["predicted_tool_calls"]
    
    for p in preds:
        p_name = p["name"]
        p_args = p.get("arguments", {})
        # Find matching gold tool by name
        for g in golds:
            if g["name"] == p_name:
                g_args = g.get("arguments", {})
                # Compare keys
                for k in p_args:
                    if k not in g_args:
                        mismatch_keys[f"{p_name} -> extra arg: {k}"] += 1
                for k in g_args:
                    if k not in p_args:
                        mismatch_keys[f"{p_name} -> missing arg: {k}"] += 1
                break

print("Top 20 Key Mismatches between Pred and Gold:")
for k, count in mismatch_keys.most_common(20):
    print(f"  {k:50s}: {count:3d} times")

