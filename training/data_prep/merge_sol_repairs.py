import sys
import json
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))
from paths import (PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR, LLM_OUTPUT_DIR)

WORKSPACE_ROOT = PROJECT_ROOT
master_path = DISTILLATION_DIR / "master_train_distilled.jsonl"
sol_path = DISTILLATION_DIR / "sol_26_samples_repair.jsonl"

sol_repairs = {}
with open(sol_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            if d.get("state_match", False):
                sol_repairs[d["id"]] = d

print(f"Sol repaired samples available: {len(sol_repairs)}")

master_records = []
replaced_count = 0

with open(master_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            sid = d["id"]
            if sid in sol_repairs:
                master_records.append(sol_repairs[sid])
                replaced_count += 1
            else:
                master_records.append(d)

total = len(master_records)
matched = sum(1 for r in master_records if r.get("state_match", False))
scores = [r.get("score", 0.0) for r in master_records]
f1s = [r.get("f1", 0.0) for r in master_records]
avg_score = sum(scores) / total if total > 0 else 0
avg_f1 = sum(f1s) / total if total > 0 else 0
perfect = sum(1 for s in scores if s == 1.0)

print("=" * 80)
print(f"Replaced with Sol repairs: {replaced_count}")
print("📊 NEW MASTER DATASET QUALITY METRICS (AFTER SOL REPAIR):")
print(f"Total Trajectories:       {total}")
print(f"State Match Count:        {matched} / {total} ({matched/total*100:.2f}%)")
print(f"State Mismatch Count:     {total - matched} / {total} ({(total - matched)/total*100:.2f}%)")
print(f"Average Overall Score:    {avg_score*100:.2f}%")
print(f"Average Tool Call F1:     {avg_f1*100:.2f}%")
print(f"Perfect 1.00 Score Count: {perfect} / {total} ({perfect/total*100:.2f}%)")
print("=" * 80)

# Save updated master
with open(master_path, "w", encoding="utf-8") as f:
    for r in master_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Saved updated master dataset to: {master_path}")

