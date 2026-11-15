import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

WORKSPACE_ROOT = PROJECT_ROOT
with open(DISTILLATION_DIR / "harmonized_sft_1526_samples.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

with open(DATA_ROOT / "train/train.jsonl", "r", encoding="utf-8") as f:
    train_samples = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

imperfect = [s for s in samples if s["score"] < 1.0]
categories = defaultdict(list)

for s in imperfect:
    sid = s["id"]
    gold_data = train_samples[sid]
    gold_calls = gold_data.get("label", {}).get("tool_calls", [])
    pred_calls = []
    for m in s["messages"]:
        if m["role"] == "assistant":
            try:
                p = json.loads(m["content"])
                pred_calls.extend(p.get("tool_calls", []))
            except Exception:
                pass
                
    p_names = [c["name"] for c in pred_calls]
    g_names = [c["name"] for c in gold_calls]
    
    if len(p_names) > len(g_names) and set(g_names).issubset(set(p_names)):
        categories["1_EXTRA_READ_CALLS"].append(s["score"])
    elif len(p_names) < len(g_names) and set(p_names).issubset(set(g_names)):
        categories["2_MISSING_TOOL_CALLS"].append(s["score"])
    elif set(p_names) != set(g_names):
        categories["3_DIFFERENT_TOOL_CHOICE"].append(s["score"])
    else:
        categories["4_ARGUMENT_MISMATCH"].append(s["score"])

print("Score statistics per category:")
for cat, scores in sorted(categories.items()):
    print(f"  {cat:25s}: {len(scores):3d} samples | Mean Score: {np.mean(scores)*100:.2f}% | Median: {np.median(scores)*100:.2f}%")
