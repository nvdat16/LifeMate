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

target_ids = []
for sid in harm_samples:
    s = train_samples[sid]
    gold = s.get("label", {}).get("tool_calls", [])
    if any(c["name"] == "sys.ask_missing_info" for c in gold):
        target_ids.append(sid)

print(f"Total Gold ask_missing_info samples in 1526: {len(target_ids)}")

# Let's see the distribution of gold calls
from collections import Counter
patterns = Counter()
for sid in target_ids:
    g_calls = [c["name"] for c in train_samples[sid].get("label", {}).get("tool_calls", [])]
    patterns[tuple(g_calls)] += 1

print("\nTop 10 Gold call patterns for ask_missing_info cases:")
for pat, cnt in patterns.most_common(10):
    print(f"  {cnt:3d} samples: {pat}")

