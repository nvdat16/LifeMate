import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json

data = json.load(open("experiments/luna_inference_10_samples.json", encoding="utf-8"))

for idx, d in enumerate(data, 1):
    print("=" * 80)
    print(f"[{idx}/10] ID: {d['id']} | Score: {d['score']:.2f} | Hash Match: {d['state_match']}")
    print(f"User Request: \"{d['audio_text']}\"")
    print(f"Number of Turns: {len(d['turns'])}")
    for t in d['turns']:
        print(f"  - Turn {t['turn']} ({t['elapsed_seconds']}s):")
        print(f"    Thought: {t['thought']}")
        print(f"    Dispatched: {[c['name'] for c in t['tool_calls']]}")
        if t['observations']:
            obs_summary = [(o['tool_name'], o['ok']) for o in t['observations']]
            print(f"    Obs: {obs_summary}")
        if t['final_response']:
            print(f"    Final: {t['final_response']}")
    print(f"PRED TOOL CALLS ({len(d['predicted_tool_calls'])}):")
    for c in d['predicted_tool_calls']:
        print(f"    {c['name']} -> {c['arguments']}")
    print(f"GOLD TOOL CALLS ({len(d['gold_tool_calls'])}):")
    for c in d['gold_tool_calls']:
        print(f"    {c['name']} -> {c['arguments']}")

