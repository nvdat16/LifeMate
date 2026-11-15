import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json

with open(DISTILLATION_DIR / "master_train_distilled.jsonl", "r", encoding="utf-8") as f:
    master_records = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

with open(DISTILLATION_DIR / "sft_train_1526_samples.json", "r", encoding="utf-8") as f:
    sft_data = json.load(f)

zero_score_samples = [master_records[s["id"]] for s in sft_data if master_records[s["id"]]["score"] == 0.0]
print(f"Total zero score samples with state_match == True: {len(zero_score_samples)}")

for r in zero_score_samples[:5]:
    print("=" * 70)
    print(f"ID: {r['id']} | Req: {r.get('audio_text')}")
    print(f"  Pred: {r['predicted_tool_calls']}")
    print(f"  Gold: {r['gold_tool_calls']}")
    print(f"  Pred Hash: {r['predicted_state_hash']}")
    print(f"  Gold Hash: {r['gold_state_hash']}")
    print(f"  Score: {r['score']} | F1: {r['f1']}")

