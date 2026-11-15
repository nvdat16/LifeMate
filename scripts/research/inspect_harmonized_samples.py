import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json

with open(DISTILLATION_DIR / "harmonized_sft_1526_samples.json", "r", encoding="utf-8") as f:
    data = json.load(f)

targets = ["train_0001", "train_0028", "train_0082"]
for item in data:
    if item["id"] in targets:
        print("=" * 80)
        print(f"SAMPLE ID: {item['id']} | Score: {item['score']} | State Match: {item['state_match']}")
        print(f"Audio: {item['audio_text']}")
        for m in item["messages"]:
            if m["role"] == "assistant":
                try:
                    p = json.loads(m["content"])
                    print(f"  [ASSISTANT]")
                    print(f"    thought:    {p.get('thought')}")
                    print(f"    tool_calls: {p.get('tool_calls')}")
                    print(f"    finished:   {p.get('is_finished')}")
                except Exception as e:
                    print("  [ASSISTANT - raw]", m["content"][:100])

