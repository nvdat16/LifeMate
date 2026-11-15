import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json
from pathlib import Path

repaired_file = DISTILLATION_DIR / "terra_failed_samples_repair.jsonl"
records = {}
with open(repaired_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            records[d["id"]] = d

print("Total records:", len(records))
matches = [r for r in records.values() if r["state_match"]]
mismatches = [r for r in records.values() if not r["state_match"]]
print(f"Repaired: {len(matches)} / {len(records)} ({len(matches)/len(records)*100:.1f}%)")
print(f"Still mismatched: {len(mismatches)} / {len(records)}")

targets = ["train_0045", "train_0009", "train_0139", "train_0014"]
for target_id in targets:
    if target_id in records:
        r = records[target_id]
        print("=" * 70)
        print(f"ID: {target_id} | State Match: {r['state_match']} | Score: {r['score']}")
        print("User request:", r.get("audio_text"))
        print("Predicted calls:", json.dumps(r["predicted_tool_calls"], ensure_ascii=False))
        print("Gold calls:     ", json.dumps(r["gold_tool_calls"], ensure_ascii=False))
        for t in r["turns"]:
            print(f"  Turn {t['turn']} Thought: {t['thought']}")

