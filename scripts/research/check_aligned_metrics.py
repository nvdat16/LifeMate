import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json
import sys
from pathlib import Path

WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from lifemate.evaluation.metrics import compute_sample_score

with open(DISTILLATION_DIR / "master_train_distilled.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip() and json.loads(line).get("state_match", False)]

print(f"Total state match records: {len(records)}")

# If we align predicted tool calls to gold calls:
aligned_scores = []
aligned_f1s = []

for r in records:
    golds = r["gold_tool_calls"]
    pred_hash = r["predicted_state_hash"]
    gold_hash = r["gold_state_hash"]
    
    # Gold calls executed in sandbox produced gold_hash
    res = compute_sample_score(golds, golds, gold_hash, gold_hash)
    aligned_scores.append(res["score"])
    aligned_f1s.append(res["f1"])

print("=" * 80)
print("POTENTIAL DATASET METRICS AFTER ALIGNMENT TO GOLD STANDARD:")
print(f"Total Samples:        {len(records)}")
print(f"State Match Rate:     100.0%")
print(f"Average F1:           {sum(aligned_f1s)/len(aligned_f1s)*100:.2f}%")
print(f"Average Score:        {sum(aligned_scores)/len(aligned_scores)*100:.2f}%")
print(f"Perfect 1.00 Count:   {sum(1 for s in aligned_scores if s == 1.0)}/{len(records)} (100.0%)")
print("=" * 80)

