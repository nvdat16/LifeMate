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

mismatches = [r for r in records.values() if not r["state_match"]]
print(f"Total Still Mismatched: {len(mismatches)}")
print("=" * 80)

for r in mismatches:
    print(f"ID: {r['id']} | Score: {r['score']}")
    print(f"  Req:  {r.get('audio_text')}")
    print(f"  Pred: {[c['name'] for c in r['predicted_tool_calls']]}")
    print(f"  Gold: {[c['name'] for c in r['gold_tool_calls']]}")
    # print diff details
    pred_calls = json.dumps(r['predicted_tool_calls'], ensure_ascii=False)
    gold_calls = json.dumps(r['gold_tool_calls'], ensure_ascii=False)
    print(f"  Pred args: {pred_calls}")
    print(f"  Gold args: {gold_calls}")
    if r['turns']:
        print(f"  Thought: {r['turns'][-1]['thought'][:150]}")
    print("-" * 80)

