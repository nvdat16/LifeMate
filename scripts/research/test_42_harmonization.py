import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import sys
from pathlib import Path
import json

WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory
from lifemate.evaluation.metrics import compute_sample_score

with open(DISTILLATION_DIR / "harmonized_sft_1526_samples.json", "r", encoding="utf-8") as f:
    harm_samples = {s["id"]: s for s in json.load(f)}

with open(DATA_ROOT / "train/train.jsonl", "r", encoding="utf-8") as f:
    train_samples = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

missing_ids = ['train_0042', 'train_0068', 'train_0065', 'train_0086', 'train_0605', 'train_0652', 'train_0775', 'train_0857', 'train_0974', 'train_1064', 'train_1103', 'train_1109', 'train_1147', 'train_1223', 'train_1237', 'train_1339', 'train_1427', 'train_1543', 'train_1570', 'train_1579', 'train_1620', 'train_1630', 'train_1679', 'train_1687', 'train_1688', 'train_1700', 'train_1723', 'train_1735', 'train_1743', 'train_0228', 'train_0243', 'train_0245', 'train_0273', 'train_0285', 'train_0331', 'train_0334', 'train_0341', 'train_0365', 'train_0366', 'train_0414', 'train_0420', 'train_0444']

factory = ZaicSandboxEnvFactory(DATA_ROOT)

def get_current_turn_calls(messages):
    prompt_idx = None
    for i, m in enumerate(messages):
        if m["role"] == "user" and "User Voice Request:" in m["content"]:
            prompt_idx = i
            break
    if prompt_idx is None:
        prompt_idx = [i for i, m in enumerate(messages) if m["role"] == "user"][-1]
    
    calls = []
    for m in messages[prompt_idx:]:
        if m["role"] == "assistant":
            try:
                data = json.loads(m["content"])
                tcs = data.get("tool_calls", [])
                calls.extend(tcs)
            except:
                pass
    return calls

results = []

for sid in missing_ids:
    s_harm = harm_samples[sid]
    s_gold = train_samples[sid]
    gold_calls = s_gold.get("label", {}).get("tool_calls", [])
    
    # current calls before
    curr_calls = get_current_turn_calls(s_harm["messages"])
    
    # Run before
    sess_before = factory.create_session("train", sid)
    for c in curr_calls:
        sess_before.call(c["name"], c.get("arguments", {}))
    pred_hash_before = sess_before.get_state_hash()
    
    sess_gold = factory.create_session("train", sid)
    for c in gold_calls:
        sess_gold.call(c["name"], c.get("arguments", {}))
    gold_hash = sess_gold.get_state_hash()
    
    score_before = compute_sample_score(curr_calls, gold_calls, pred_hash_before, gold_hash)
    
    # Now simulate harmonized calls matching gold
    # In gold, sys.ask_missing_info is present.
    # What if we ensure curr_calls includes the gold sys.ask_missing_info (and missing gold read calls if any)?
    harmonized_calls = list(curr_calls)
    for gc in gold_calls:
        if not any(c["name"] == gc["name"] for c in harmonized_calls):
            harmonized_calls.append(gc)
            
    sess_after = factory.create_session("train", sid)
    for c in harmonized_calls:
        sess_after.call(c["name"], c.get("arguments", {}))
    pred_hash_after = sess_after.get_state_hash()
    
    score_after = compute_sample_score(harmonized_calls, gold_calls, pred_hash_after, gold_hash)
    
    results.append({
        "id": sid,
        "score_before": score_before["score"],
        "match_before": score_before["state_match"],
        "score_after": score_after["score"],
        "match_after": score_after["state_match"],
        "calls_before": [c["name"] for c in curr_calls],
        "calls_after": [c["name"] for c in harmonized_calls],
        "gold_calls": [c["name"] for c in gold_calls]
    })

avg_before = sum(r["score_before"] for r in results) / len(results)
avg_after = sum(r["score_after"] for r in results) / len(results)
all_match = all(r["match_after"] for r in results)

print(f"Results across all 42 samples:")
print(f"  Avg Score Before: {avg_before:.4f}")
print(f"  Avg Score After:  {avg_after:.4f}")
print(f"  All State Match:  {all_match}")
print(f"  Perfect 1.00 count before: {sum(1 for r in results if r['score_before'] == 1.0)}")
print(f"  Perfect 1.00 count after:  {sum(1 for r in results if r['score_after'] == 1.0)}")

print("\n--- SAMPLES WITH SCORE < 1.0 AFTER ---")
for r in results:
    if r['score_after'] < 1.0:
        print(f"{r['id']}: Score={r['score_after']:.2f} (Before: {r['score_before']:.2f})")
        print(f"   After calls: {r['calls_after']}")
        print(f"   Gold calls:  {r['gold_calls']}")
