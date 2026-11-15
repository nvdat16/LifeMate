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

with open(DISTILLATION_DIR / "harmonized_sft_1526_samples.json", "r", encoding="utf-8") as f:
    harm_samples = {s["id"]: s for s in json.load(f)}

with open(DATA_ROOT / "train/train.jsonl", "r", encoding="utf-8") as f:
    train_samples = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

def get_current_turn_msgs(messages):
    prompt_idx = None
    for i, m in enumerate(messages):
        if m["role"] == "user" and "User Voice Request:" in m["content"]:
            prompt_idx = i
            break
    if prompt_idx is None:
        prompt_idx = [i for i, m in enumerate(messages) if m["role"] == "user"][-1]
    return messages[prompt_idx:]

missing_ids = ['train_0042', 'train_0068', 'train_0065', 'train_0086', 'train_0605', 'train_0652', 'train_0775', 'train_0857', 'train_0974', 'train_1064', 'train_1103', 'train_1109', 'train_1147', 'train_1223', 'train_1237', 'train_1339', 'train_1427', 'train_1543', 'train_1570', 'train_1579', 'train_1620', 'train_1630', 'train_1679', 'train_1687', 'train_1688', 'train_1700', 'train_1723', 'train_1735', 'train_1743', 'train_0228', 'train_0243', 'train_0245', 'train_0273', 'train_0285', 'train_0331', 'train_0334', 'train_0341', 'train_0365', 'train_0366', 'train_0414', 'train_0420', 'train_0444']

print(f"Total missing: {len(missing_ids)}")

for sid in missing_ids:
    s_harm = harm_samples[sid]
    s_gold = train_samples[sid]
    g_calls = s_gold.get("label", {}).get("tool_calls", [])
    c_msgs = get_current_turn_msgs(s_harm["messages"])
    
    # assistant turns in current turn
    asst_turns = [m for m in c_msgs if m["role"] == "assistant"]
    print(f"\n--- {sid} ---")
    print(f"Req: {s_harm.get('audio_text')}")
    print(f"Gold calls: {[c['name'] for c in g_calls]}")
    gold_ask = [c for c in g_calls if c["name"] == "sys.ask_missing_info"][0]
    print(f"Gold ask args: {gold_ask.get('arguments')}")
    print(f"Assistant current turns count: {len(asst_turns)}")
    for j, at in enumerate(asst_turns):
        try:
            d = json.loads(at["content"])
            print(f"  Turn {j}: calls={[c['name'] for c in d.get('tool_calls', [])]}, finished={d.get('is_finished')}")
        except:
            print(f"  Turn {j}: raw content={at['content'][:100]}")

