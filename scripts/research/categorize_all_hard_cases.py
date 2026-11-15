import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json
from collections import defaultdict

with open(DISTILLATION_DIR / "terra_failed_samples_repair.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

# Load original luna records to see what Luna did
with open(DISTILLATION_DIR / "train_distilled_trajectories.jsonl", "r", encoding="utf-8") as f:
    luna_records = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

print(f"Total analyzed: {len(records)}")

repaired = [r for r in records if r["state_match"]]
unrepaired = [r for r in records if not r["state_match"]]

def get_apps(tool_calls):
    apps = set()
    for c in tool_calls:
        name = c["name"]
        if "." in name:
            apps.add(name.split(".")[0])
        else:
            apps.add(name)
    return apps

print("\n--- REPAIRED CASES (61) APP DISTRIBUTION ---")
repaired_apps = defaultdict(int)
for r in repaired:
    apps = get_apps(r["gold_tool_calls"])
    for app in apps:
        repaired_apps[app] += 1
for app, count in sorted(repaired_apps.items(), key=lambda x: -x[1]):
    print(f"  {app:20s}: {count} samples")

print("\n--- UNREPAIRED CASES (26) APP DISTRIBUTION ---")
unrepaired_apps = defaultdict(int)
for r in unrepaired:
    apps = get_apps(r["gold_tool_calls"])
    for app in apps:
        unrepaired_apps[app] += 1
for app, count in sorted(unrepaired_apps.items(), key=lambda x: -x[1]):
    print(f"  {app:20s}: {count} samples")

