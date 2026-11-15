import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json

lines = open(DISTILLATION_DIR / 'train_distilled_trajectories.jsonl', encoding='utf-8').readlines()
data = [json.loads(l) for l in lines]

count = 0
for d in data:
    if len(d['gold_tool_calls']) > 0 and len(d['predicted_tool_calls']) == 0:
        print("=" * 80)
        print(f"ID: {d['id']}")
        print(f"User Request: {d['audio_text']}")
        print(f"Gold calls: {[c['name'] for c in d['gold_tool_calls']]}")
        for t in d['turns']:
            print(f"  Turn {t['turn']} Thought: {t['thought']}")
            print(f"  Turn {t['turn']} Calls: {t['tool_calls']}")
            print(f"  Turn {t['turn']} Finished: {t['is_finished']}")
        count += 1
        if count >= 3:
            break

