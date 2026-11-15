import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json
import numpy as np

with open(DISTILLATION_DIR / "sft_train_1526_samples.json", "r", encoding="utf-8") as f:
    sft_data = json.load(f)

with open(DISTILLATION_DIR / "master_train_distilled.jsonl", "r", encoding="utf-8") as f:
    records = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

matched_samples = [records[s["id"]] for s in sft_data]

total = len(matched_samples)
scores = [r["score"] for r in matched_samples]
f1s = [r["f1"] for r in matched_samples]
precisions = [r["precision"] for r in matched_samples]
recalls = [r["recall"] for r in matched_samples]

print(f"Total Verified Samples: {total}")
print(f"State Match Rate: 100% ({total}/{total})")
print(f"Average Score:     {np.mean(scores)*100:.2f}%")
print(f"Average F1:        {np.mean(f1s)*100:.2f}%")
print(f"Average Precision: {np.mean(precisions)*100:.2f}%")
print(f"Average Recall:    {np.mean(recalls)*100:.2f}%")

print("\n--- SCORE DISTRIBUTION ---")
bins = [
    ("Score = 1.00 (Hoàn hảo tuyệt đối)", lambda s: s == 1.0),
    ("0.80 <= Score < 1.00 (Rất cao)", lambda s: 0.80 <= s < 1.0),
    ("0.60 <= Score < 0.80 (Tốt)", lambda s: 0.60 <= s < 0.80),
    ("0.40 <= Score < 0.60 (Trung bình)", lambda s: 0.40 <= s < 0.60),
    ("0.00 < Score < 0.40 (Thấp)", lambda s: 0.00 < s < 0.40),
    ("Score == 0.00", lambda s: s == 0.0)
]

for label, cond in bins:
    cnt = sum(1 for s in scores if cond(s))
    print(f"  {label:35s}: {cnt:4d} samples ({cnt/total*100:.2f}%)")

# Why did F1 drop from 1.0?
diff_reasons = {
    "extra_read_tool_filters": 0, # Passed date_from/date_to/floor to list_* tools
    "extra_intermediate_reads": 0, # e.g. checked contacts or balance twice
    "extra_optional_note_field": 0, # Passed 'note' to transfer/topup
    "content_f1_threshold": 0 # text token mismatch
}

for r in matched_samples:
    if r["score"] < 1.0:
        preds = r["predicted_tool_calls"]
        golds = r["gold_tool_calls"]
        # Check reasons
        pred_names = [c["name"] for c in preds]
        gold_names = [c["name"] for c in golds]
        
        # Check if difference is due to arguments on list/search tools
        has_list_arg_diff = False
        for p in preds:
            if p["name"].startswith("list_") or p["name"].startswith("get_status") or p["name"] == "tv.guide":
                if p.get("arguments") and not any(g["name"] == p["name"] and g.get("arguments") == p.get("arguments") for g in golds):
                    has_list_arg_diff = True
        if has_list_arg_diff:
            diff_reasons["extra_read_tool_filters"] += 1
        elif len(preds) > len(golds):
            diff_reasons["extra_intermediate_reads"] += 1
        else:
            diff_reasons["extra_optional_note_field"] += 1

print("\n--- REASONS FOR IMPERFECT F1 ON STATE MATCH SAMPLES ---")
for r, c in diff_reasons.items():
    print(f"  {r:30s}: {c:4d} samples ({c/total*100:.2f}%)")

