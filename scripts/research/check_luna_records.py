import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json

with open(DISTILLATION_DIR / "train_distilled_trajectories.jsonl", "r", encoding="utf-8") as f:
    for i in range(3):
        line = f.readline()
        if line.strip():
            d = json.loads(line)
            print(f"ID: {d['id']} | Score: {d['score']}")
            print("Pred:", d["predicted_tool_calls"])
            print("Gold:", d["gold_tool_calls"])

