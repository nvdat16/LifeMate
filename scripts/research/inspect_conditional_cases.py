import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json

with open(DATA_ROOT / "train/train.jsonl", "r", encoding="utf-8") as f:
    samples = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

# Let's search for all samples with "Nếu mai mưa" or "Mai mà mưa"
conditional_ids = []
for sid, s in samples.items():
    audio = s.get("audio_text", "").lower()
    if ("mưa" in audio and "thì" in audio) or ("nếu" in audio and "thì" in audio):
        conditional_ids.append(sid)

print(f"Total conditional (nếu/thì/mưa) samples: {len(conditional_ids)}")

for sid in conditional_ids[:15]:
    s = samples[sid]
    audio = s.get("audio_text", "")
    gold_calls = [c["name"] for c in s.get("label", {}).get("tool_calls", [])]
    print(f"[{sid}] Req: {audio[:65]}...")
    print(f"       Gold calls: {gold_calls}")

