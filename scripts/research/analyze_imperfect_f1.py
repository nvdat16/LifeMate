import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json
from collections import Counter

with open(DISTILLATION_DIR / "sft_train_1526_samples.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

scores = [s["score"] for s in samples]
perfect = sum(1 for s in scores if s == 1.0)
imperfect = [s for s in samples if s["score"] < 1.0]

print(f"Total samples: {len(samples)}")
print(f"Perfect 1.00 (State Match + F1=1.0): {perfect} ({perfect/len(samples)*100:.2f}%)")
print(f"State Match True but F1 < 1.0: {len(imperfect)} ({len(imperfect)/len(samples)*100:.2f}%)")

avg_score = sum(scores) / len(scores)
print(f"Average Overall Score across 1,526 samples: {avg_score*100:.2f}%")

# Now let's inspect the reasons why F1 < 1.0
print("\nInspecting first 5 samples with State Match == True but F1 < 1.0:")
with open(DISTILLATION_DIR / "master_train_distilled.jsonl", "r", encoding="utf-8") as f:
    master_records = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

reasons = Counter()

for s in imperfect[:10]:
    sid = s["id"]
    rec = master_records[sid]
    preds = rec["predicted_tool_calls"]
    golds = rec["gold_tool_calls"]
    
    print("=" * 70)
    print(f"ID: {sid} | Score: {rec['score']} | Precision: {rec['precision']:.2f} | Recall: {rec['recall']:.2f}")
    print(f"  Req:  {rec.get('audio_text')}")
    print(f"  Pred ({len(preds)}): {preds}")
    print(f"  Gold ({len(golds)}): {golds}")

