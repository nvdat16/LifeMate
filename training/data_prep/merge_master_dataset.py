import sys
import json
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))
from paths import (PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR, LLM_OUTPUT_DIR)

WORKSPACE_ROOT = PROJECT_ROOT
luna_path = DISTILLATION_DIR / "train_distilled_trajectories.jsonl"
terra_path = DISTILLATION_DIR / "terra_failed_samples_repair.jsonl"
master_path = DISTILLATION_DIR / "master_train_distilled.jsonl"

# 1. Load Terra repairs
terra_repairs = {}
with open(terra_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            # Only use if state_match is True or score is higher
            terra_repairs[d["id"]] = d

print(f"Loaded {len(terra_repairs)} Terra records.")
repaired_count = sum(1 for d in terra_repairs.values() if d.get("state_match", False))
print(f"Terra repaired records (state_match=True): {repaired_count}")

# 2. Merge into Luna master
master_records = []
replaced_count = 0
unmatched_count = 0

with open(luna_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            s_id = d["id"]
            if s_id in terra_repairs and terra_repairs[s_id].get("state_match", False):
                master_records.append(terra_repairs[s_id])
                replaced_count += 1
            else:
                master_records.append(d)

print(f"Total Master Records: {len(master_records)}")
print(f"Replaced with Terra Repairs: {replaced_count}")

# 3. Compute overall Master Metrics
total = len(master_records)
matched = sum(1 for r in master_records if r.get("state_match", False))
scores = [r.get("score", 0.0) for r in master_records]
f1s = [r.get("f1", 0.0) for r in master_records]
avg_score = sum(scores) / total if total > 0 else 0
avg_f1 = sum(f1s) / total if total > 0 else 0
perfect = sum(1 for s in scores if s == 1.0)

print("=" * 80)
print("📊 MASTER DATASET QUALITY METRICS (AFTER TERRA REPAIR):")
print(f"Total Trajectories:       {total}")
print(f"State Match Count:        {matched} / {total} ({matched/total*100:.2f}%)")
print(f"State Mismatch Count:     {total - matched} / {total} ({(total - matched)/total*100:.2f}%)")
print(f"Average Overall Score:    {avg_score*100:.2f}%")
print(f"Average Tool Call F1:     {avg_f1*100:.2f}%")
print(f"Perfect 1.00 Score Count: {perfect} / {total} ({perfect/total*100:.2f}%)")
print("=" * 80)

# Save master file
with open(master_path, "w", encoding="utf-8") as f:
    for r in master_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Saved master dataset to: {master_path}")

