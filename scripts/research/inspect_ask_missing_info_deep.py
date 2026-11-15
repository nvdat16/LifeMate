import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import sys
from pathlib import Path

WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

import json
from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory
from lifemate.evaluation.metrics import compute_sample_score

with open(DISTILLATION_DIR / "harmonized_sft_1526_samples.json", "r", encoding="utf-8") as f:
    harm_samples = {s["id"]: s for s in json.load(f)}

with open(DATA_ROOT / "train/train.jsonl", "r", encoding="utf-8") as f:
    train_samples = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

factory = ZaicSandboxEnvFactory(DATA_ROOT)

gold_ask_ids = [sid for sid, s in train_samples.items() if any(c["name"] == "sys.ask_missing_info" for c in s.get("label", {}).get("tool_calls", []))]
harm_ask_ids = []
for sid, s in harm_samples.items():
    for m in s["messages"]:
        if m["role"] == "assistant":
            try:
                c = json.loads(m["content"])
                if any(tc.get("name") == "sys.ask_missing_info" for tc in c.get("tool_calls", [])):
                    harm_ask_ids.append(sid)
                    break
            except:
                pass

gold_set = set(gold_ask_ids)
harm_set = set(harm_ask_ids)
in_gold_not_harm = sorted(list(gold_set - harm_set))
in_harm_not_gold = sorted(list(harm_set - gold_set))

print(f"=== IN GOLD BUT NOT HARMONIZED ({len(in_gold_not_harm)} samples) ===")
for sid in in_gold_not_harm[:10]:
    s_harm = harm_samples[sid]
    s_gold = train_samples[sid]
    g_calls = s_gold.get("label", {}).get("tool_calls", [])
    # get harm calls
    h_calls = []
    for m in s_harm["messages"]:
        if m["role"] == "assistant":
            try:
                h_calls.extend(json.loads(m["content"]).get("tool_calls", []))
            except:
                pass
    print(f"ID: {sid} | Score: {s_harm.get('score', 0):.2f}")
    print(f"  Req:  {s_harm.get('audio_text')}")
    print(f"  Harm: {[c['name'] for c in h_calls]}")
    print(f"  Gold: {[c['name'] for c in g_calls]}")
    for gc in g_calls:
        if gc["name"] == "sys.ask_missing_info":
            print(f"  Gold ask_missing_info args: {gc.get('arguments')}")

print("\n=== IN HARMONIZED BUT NOT GOLD (23 samples) ===")
for sid in in_harm_not_gold[:10]:
    s_harm = harm_samples[sid]
    s_gold = train_samples[sid]
    g_calls = s_gold.get("label", {}).get("tool_calls", [])
    h_calls = []
    for m in s_harm["messages"]:
        if m["role"] == "assistant":
            try:
                h_calls.extend(json.loads(m["content"]).get("tool_calls", []))
            except:
                pass
    print(f"ID: {sid} | Score: {s_harm.get('score', 0):.2f}")
    print(f"  Req:  {s_harm.get('audio_text')}")
    print(f"  Harm: {[c['name'] for c in h_calls]}")
    print(f"  Gold: {[c['name'] for c in g_calls]}")
