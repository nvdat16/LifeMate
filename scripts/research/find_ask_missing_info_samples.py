import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json

with open(DISTILLATION_DIR / "sft_train_1526_samples.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

with open(DATA_ROOT / "train/train.jsonl", "r", encoding="utf-8") as f:
    train_samples = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

# Case A: Gold has sys.ask_missing_info, but Pred does NOT
case_a = []
# Case B: Both Gold and Pred have sys.ask_missing_info
case_b = []
# Case C: Pred has sys.ask_missing_info, but Gold does NOT
case_c = []

for s in samples:
    sid = s["id"]
    gold = train_samples[sid].get("label", {}).get("tool_calls", [])
    pred_calls = []
    for m in s["messages"]:
        if m["role"] == "assistant":
            try:
                p = json.loads(m["content"])
                pred_calls.extend(p.get("tool_calls", []))
            except Exception:
                pass
                
    g_has_ask = any(c["name"] == "sys.ask_missing_info" for c in gold)
    p_has_ask = any(c["name"] == "sys.ask_missing_info" for c in pred_calls)
    
    if g_has_ask and not p_has_ask:
        case_a.append((sid, s["score"], s["audio_text"], pred_calls, gold))
    elif g_has_ask and p_has_ask:
        case_b.append((sid, s["score"], s["audio_text"], pred_calls, gold))
    elif not g_has_ask and p_has_ask:
        case_c.append((sid, s["score"], s["audio_text"], pred_calls, gold))

print(f"Case A (Gold has sys.ask_missing_info, Pred does NOT): {len(case_a)} samples")
print(f"Case B (Both Gold and Pred have sys.ask_missing_info): {len(case_b)} samples")
print(f"Case C (Pred has sys.ask_missing_info, Gold does NOT): {len(case_c)} samples")

print("\n--- SAMPLE CASE A (Gold has sys.ask_missing_info, Pred does NOT) ---")
for sid, score, audio, preds, golds in case_a[:3]:
    print("=" * 70)
    print(f"ID: {sid} | Score: {score}")
    print(f"Req: {audio}")
    print(f"Pred: {[c['name'] for c in preds]}")
    print(f"Gold: {[c['name'] for c in golds]}")

print("\n--- SAMPLE CASE B (Both have sys.ask_missing_info) ---")
for sid, score, audio, preds, golds in case_b[:3]:
    print("=" * 70)
    print(f"ID: {sid} | Score: {score}")
    print(f"Req: {audio}")
    print(f"Pred: {[c['name'] for c in preds]}")
    print(f"Gold: {[c['name'] for c in golds]}")

print("\n--- SAMPLE CASE C (Pred has sys.ask_missing_info, Gold does NOT) ---")
for sid, score, audio, preds, golds in case_c[:3]:
    print("=" * 70)
    print(f"ID: {sid} | Score: {score}")
    print(f"Req: {audio}")
    print(f"Pred: {[c['name'] for c in preds]}")
    print(f"Gold: {[c['name'] for c in golds]}")

