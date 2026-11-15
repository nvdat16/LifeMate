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

with open(DISTILLATION_DIR / "harmonized_sft_1526_samples.json", "r", encoding="utf-8") as f:
    harm_samples = {s["id"]: s for s in json.load(f)}

with open(DATA_ROOT / "train/train.jsonl", "r", encoding="utf-8") as f:
    train_samples = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

def get_all_assistant_calls(messages):
    calls = []
    for m in messages:
        if m["role"] == "assistant":
            try:
                data = json.loads(m["content"])
                tcs = data.get("tool_calls", [])
                calls.extend(tcs)
            except:
                pass
    return calls

def get_current_turn_calls(messages, history_len):
    # History turns: 2 * history_len messages (user + assistant pairs)
    # Plus message 0 is system
    # Current turn starts at index 1 + 2 * history_len
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

print(f"Total harmonized samples: {len(harm_samples)}")

ask_in_gold = []
for sid in harm_samples:
    s = train_samples[sid]
    gold = s.get("label", {}).get("tool_calls", [])
    if any(c["name"] == "sys.ask_missing_info" for c in gold):
        hist_len = len(s.get("history", []))
        h_calls = get_current_turn_calls(harm_samples[sid]["messages"], hist_len)
        has_ask_h = any(c["name"] == "sys.ask_missing_info" for c in h_calls)
        ask_in_gold.append((sid, has_ask_h, h_calls, gold))

print(f"Total samples with sys.ask_missing_info in Gold (current turn): {len(ask_in_gold)}")
missing_ask = [x for x in ask_in_gold if not x[1]]
has_ask = [x for x in ask_in_gold if x[1]]

print(f"Samples where Harmonized ALREADY has sys.ask_missing_info: {len(has_ask)}")
print(f"Samples where Harmonized is MISSING sys.ask_missing_info: {len(missing_ask)}")

print("\nSample missing cases:")
for sid, _, h_c, g_c in missing_ask[:5]:
    print(f"ID: {sid}")
    print(f"  Harm: {[c['name'] for c in h_c]}")
    print(f"  Gold: {[c['name'] for c in g_c]}")
    print(f"  Gold args: {[c.get('arguments') for c in g_c if c['name'] == 'sys.ask_missing_info']}")

# Check samples where Gold does NOT have sys.ask_missing_info, but Harmonized DOES in current turn
extra_in_harm = []
for sid in harm_samples:
    s = train_samples[sid]
    gold = s.get("label", {}).get("tool_calls", [])
    if not any(c["name"] == "sys.ask_missing_info" for c in gold):
        hist_len = len(s.get("history", []))
        h_calls = get_current_turn_calls(harm_samples[sid]["messages"], hist_len)
        if any(c["name"] == "sys.ask_missing_info" for c in h_calls):
            extra_in_harm.append((sid, h_calls, gold))

print(f"Samples where Harmonized calls sys.ask_missing_info in current turn BUT Gold does NOT: {len(extra_in_harm)}")
for sid, h_c, g_c in extra_in_harm[:10]:
    print(f"ID: {sid} | Harm: {[c['name'] for c in h_c]} | Gold: {[c['name'] for c in g_c]}")
