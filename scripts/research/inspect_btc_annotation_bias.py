import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json
from pathlib import Path

with open(DATA_ROOT / "train/train.jsonl", "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]

catalog = json.load(open(DATA_ROOT / "train/train_tools_catalog.json", encoding="utf-8"))
tool_kinds = {}
for app in catalog.get("apps", []):
    for tool in app.get("tools", []):
        tool_kinds[tool["name"]] = tool.get("kind", "write")

empty_labels = [s for s in samples if not s.get("label", {}).get("tool_calls")]
print(f"Total samples in train: {len(samples)}")
print(f"Samples with empty gold tool_calls []: {len(empty_labels)} ({len(empty_labels)/len(samples)*100:.1f}%)")

print("\n--- SAMPLE CONDITIONAL CASES WITH EMPTY GOLD ---")
for s in empty_labels[:8]:
    audio = s.get("audio_text", "")
    print(f"[{s['id']}] {audio}")

# Check: How many gold tool_calls across the entire dataset contain READ tools?
samples_with_read_in_gold = 0
samples_with_only_write_in_gold = 0
total_read_calls_in_gold = 0
total_write_calls_in_gold = 0

for s in samples:
    calls = s.get("label", {}).get("tool_calls", [])
    if not calls:
        continue
    kinds = [tool_kinds.get(c["name"], "write") for c in calls]
    has_read = "read" in kinds
    if has_read:
        samples_with_read_in_gold += 1
    else:
        samples_with_only_write_in_gold += 1
    for k in kinds:
        if k == "read":
            total_read_calls_in_gold += 1
        else:
            total_write_calls_in_gold += 1

print("\n--- BTC GOLD CALLS DISTRIBUTION (READ VS WRITE) ---")
print(f"Samples with at least 1 READ in Gold:   {samples_with_read_in_gold}/{len(samples)}")
print(f"Samples with ONLY WRITE in Gold:        {samples_with_only_write_in_gold}/{len(samples)}")
print(f"Total READ calls in Gold:               {total_read_calls_in_gold}")
print(f"Total WRITE calls in Gold:              {total_write_calls_in_gold}")

