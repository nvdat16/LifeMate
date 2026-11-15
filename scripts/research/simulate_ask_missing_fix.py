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
from lifemate.evaluation.metrics import compute_sample_score, call_matches

with open(DISTILLATION_DIR / "harmonized_sft_1526_samples.json", "r", encoding="utf-8") as f:
    harm_samples = {s["id"]: s for s in json.load(f)}

with open(DATA_ROOT / "train/train.jsonl", "r", encoding="utf-8") as f:
    train_samples = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

factory = ZaicSandboxEnvFactory(DATA_ROOT)

def get_current_turn_calls(messages, history_len):
    start_idx = 1 + 2 * history_len
    calls = []
    for m in messages[start_idx:]:
        if m["role"] == "assistant":
            try:
                data = json.loads(m["content"])
                tcs = data.get("tool_calls", [])
                calls.extend(tcs)
            except:
                pass
    return calls

missing_samples = []
for sid in harm_samples:
    s = train_samples[sid]
    gold = s.get("label", {}).get("tool_calls", [])
    if any(c["name"] == "sys.ask_missing_info" for c in gold):
        hist_len = len(s.get("history", []))
        h_calls = get_current_turn_calls(harm_samples[sid]["messages"], hist_len)
        if not any(c["name"] == "sys.ask_missing_info" for c in h_calls):
            missing_samples.append(sid)

print(f"Total missing samples to analyze: {len(missing_samples)}")

scores_before = []
scores_after = []

for sid in missing_samples[:10]:
    s = train_samples[sid]
    gold_calls = s.get("label", {}).get("tool_calls", [])
    hist_len = len(s.get("history", []))
    h_calls = get_current_turn_calls(harm_samples[sid]["messages"], hist_len)
    
    # Run in sandbox
    session = factory.create_session("train", sid)
    for c in h_calls:
        session.call(c["name"], c.get("arguments", {}))
    pred_hash = session.get_state_hash()
    
    gold_session = factory.create_session("train", sid)
    for c in gold_calls:
        gold_session.call(c["name"], c.get("arguments", {}))
    gold_hash = gold_session.get_state_hash()
    
    res_before = compute_sample_score(h_calls, gold_calls, pred_hash, gold_hash)
    scores_before.append(res_before["score"])
    
    # Now simulated with gold's sys.ask_missing_info added
    gold_ask = [c for c in gold_calls if c["name"] == "sys.ask_missing_info"]
    simulated_calls = list(h_calls) + gold_ask
    
    # check state hash
    session2 = factory.create_session("train", sid)
    for c in simulated_calls:
        session2.call(c["name"], c.get("arguments", {}))
    pred2_hash = session2.get_state_hash()
    
    res_after = compute_sample_score(simulated_calls, gold_calls, pred2_hash, gold_hash)
    scores_after.append(res_after["score"])
    
    print(f"{sid}: Before = {res_before['score']:.2f} (Match: {res_before['state_match']}) -> After = {res_after['score']:.2f} (Match: {res_after['state_match']})")
    print(f"   Harm calls: {[c['name'] for c in h_calls]}")
    print(f"   Gold calls: {[c['name'] for c in gold_calls]}")

