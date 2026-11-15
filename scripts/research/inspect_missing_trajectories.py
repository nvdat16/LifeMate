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

missing_ids = ['train_0042', 'train_0068', 'train_0065', 'train_0086', 'train_0605', 'train_0652', 'train_0775', 'train_0857', 'train_0974', 'train_1064', 'train_1103', 'train_1109', 'train_1147', 'train_1223', 'train_1237', 'train_1339', 'train_1427', 'train_1543', 'train_1570', 'train_1579', 'train_1620', 'train_1630', 'train_1679', 'train_1687', 'train_1688', 'train_1700', 'train_1723', 'train_1735', 'train_1743', 'train_0228', 'train_0243', 'train_0245', 'train_0273', 'train_0285', 'train_0331', 'train_0334', 'train_0341', 'train_0365', 'train_0366', 'train_0414', 'train_0420', 'train_0444']

for sid in missing_ids:
    s = harm_samples[sid]
    g = train_samples[sid]
    g_calls = g.get("label", {}).get("tool_calls", [])
    gold_ask = [c for c in g_calls if c["name"] == "sys.ask_missing_info"][0]
    
    # Let's inspect the messages
    msgs = s["messages"]
    prompt_idx = None
    for i, m in enumerate(msgs):
        if m["role"] == "user" and "User Voice Request:" in m["content"]:
            prompt_idx = i
            break
    if prompt_idx is None:
        prompt_idx = [i for i, m in enumerate(msgs) if m["role"] == "user"][-1]
    
    curr_msgs = msgs[prompt_idx:]
    print(f"Sample: {sid} | Prompt idx: {prompt_idx} | Curr msgs count: {len(curr_msgs)}")
    print(f"  Gold ask: {gold_ask}")
    # print role and summary of curr msgs
    for j, m in enumerate(curr_msgs):
        content = m["content"]
        if m["role"] == "assistant":
            try:
                data = json.loads(content)
                print(f"    Msg {j} ({m['role']}): is_finished={data.get('is_finished')}, calls={[c['name'] for c in data.get('tool_calls', [])]}")
            except:
                print(f"    Msg {j} ({m['role']}): {content[:60]}...")
        else:
            print(f"    Msg {j} ({m['role']}): {content[:60]}...")

