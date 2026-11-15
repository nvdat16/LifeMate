import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json
import sys
from pathlib import Path

WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from lifemate.adapters.storage.jsonl_dataset_reader import JsonlDatasetReader
from run_terra_on_failed_samples import extract_full_installed_schemas, build_system_prompt

reader = JsonlDatasetReader(DATA_ROOT)
all_samples = {s.id: s for s in reader.read_samples("train")}
catalog = json.load(open(DATA_ROOT / "train/train_tools_catalog.json", encoding="utf-8"))

sample = all_samples["train_0082"]
context = sample.context
tool_schemas = extract_full_installed_schemas(catalog, context.installed_apps)
system_prompt = build_system_prompt(context, tool_schemas)

messages = [{"role": "system", "content": "SYSTEM_PROMPT_TRUNCATED"}]
for turn in sample.history:
    if turn.role == "user":
        messages.append({"role": "user", "content": turn.audio_text or turn.text or ""})
    elif turn.role == "assistant":
        if turn.tool_calls:
            tc_dicts = [tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in turn.tool_calls]
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "thought": f"Assisting request: {turn.text or ''}",
                    "tool_calls": tc_dicts,
                    "is_finished": False
                }, ensure_ascii=False)
            })
            if turn.tool_results:
                messages.append({
                    "role": "user",
                    "content": f"OBSERVATIONS FROM TOOL EXECUTION:\n{json.dumps(turn.tool_results, ensure_ascii=False)}"
                })
        if turn.text:
            messages.append({"role": "assistant", "content": turn.text})

messages.append({"role": "user", "content": f"User Voice Request: \"{sample.audio_text or ''}\""})

print(f"Total messages built for train_0082: {len(messages)}")
for i, m in enumerate(messages):
    print(f"\n--- Message {i} [{m['role']}] ---")
    print(m["content"][:200])
