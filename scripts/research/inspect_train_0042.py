import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json

with open(DISTILLATION_DIR / "master_train_distilled.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if d["id"] == "train_0042":
            print("ID: train_0042")
            print("Pred calls:", d["predicted_tool_calls"])
            print("Gold calls:", d["gold_tool_calls"])
            for t in d["turns"]:
                print(f"Turn {t['turn']}:")
                print("  thought:", t.get("thought"))
                print("  tools:  ", t.get("tool_calls"))
                print("  obs:    ", t.get("observations"))
                print("  resp:   ", t.get("final_response"))
            break

