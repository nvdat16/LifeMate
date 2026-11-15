import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json

with open(DISTILLATION_DIR / "sol_26_samples_repair.jsonl", "r", encoding="utf-8") as f:
    records = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

for sid in ["train_0014", "train_0708", "train_1085", "train_0947", "train_0588"]:
    if sid in records:
        r = records[sid]
        print("=" * 70)
        print(f"ID: {sid} | State Match: {r['state_match']} | Score: {r['score']}")
        print("User request:", r.get("audio_text"))
        print("Pred calls:", json.dumps(r["predicted_tool_calls"], ensure_ascii=False))
        print("Gold calls:", json.dumps(r["gold_tool_calls"], ensure_ascii=False))
        for t in r["turns"]:
            print(f"  Turn {t['turn']} Thought: {t['thought']}")

