import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json

lines = open(DISTILLATION_DIR / 'train_distilled_trajectories.jsonl', encoding='utf-8').readlines()
data = [json.loads(l) for l in lines]
mismatches = [d for d in data if not d['state_match']]

print(f"Total Mismatches: {len(mismatches)}")

# Pick diverse representative cases
cases_to_inspect = [
    'train_0009', # phone digits
    'train_0014', # smart home conditional
    'train_0045', # ask missing info
    'train_0078', # contacts update
    'train_0098', # multi-action
    'train_0139', # conditional
]

for s_id in cases_to_inspect:
    m = next((d for d in mismatches if d['id'] == s_id), None)
    if m:
        print("=" * 80)
        print(f"ID: {m['id']}")
        print(f"User Request: \"{m['audio_text']}\"")
        print("Pred Tool Calls:")
        for c in m['predicted_tool_calls']:
            print(f"   -> {c['name']}: {c['arguments']}")
        print("Gold Tool Calls:")
        for c in m['gold_tool_calls']:
            print(f"   -> {c['name']}: {c['arguments']}")
        print("Thought trace snippet:")
        for t in m['turns']:
            print(f"   [Turn {t['turn']}] {t['thought'][:150]}...")

