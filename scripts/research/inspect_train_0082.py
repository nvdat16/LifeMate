import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json

with open(DATA_ROOT / "train/train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if d["id"] == "train_0082":
            print(json.dumps(d, ensure_ascii=False, indent=2))
            break

