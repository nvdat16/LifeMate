import json
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))
from paths import (PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR, LLM_OUTPUT_DIR)

WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from lifemate.adapters.storage.jsonl_dataset_reader import JsonlDatasetReader
from run_terra_on_failed_samples import extract_full_installed_schemas, build_system_prompt

def export_dataset():
    master_file = DISTILLATION_DIR / "master_train_distilled.jsonl"
    catalog_path = DATA_ROOT / "train" / "train_tools_catalog.json"
    catalog = json.load(open(catalog_path, encoding="utf-8"))

    reader = JsonlDatasetReader(DATA_ROOT)
    all_samples = {s.id: s for s in reader.read_samples("train")}

    # Also load terra and sol files to identify teacher source
    terra_ids = set()
    terra_file = DISTILLATION_DIR / "terra_failed_samples_repair.jsonl"
    if terra_file.is_file():
        with open(terra_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if d.get("state_match", False):
                        terra_ids.add(d["id"])

    sol_ids = set()
    sol_file = DISTILLATION_DIR / "sol_26_samples_repair.jsonl"
    if sol_file.is_file():
        with open(sol_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if d.get("state_match", False):
                        sol_ids.add(d["id"])

    sft_data = []

    with open(master_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("state_match", False):
                continue

            sid = rec["id"]
            sample = all_samples[sid]
            context = sample.context

            # 1. Build System Prompt
            tool_schemas = extract_full_installed_schemas(catalog, context.installed_apps)
            system_prompt = build_system_prompt(context, tool_schemas)

            messages = [{"role": "system", "content": system_prompt}]

            # 2. Add sample history if any (strictly formatted as LifeMate JSON protocol)
            for turn in sample.history:
                if turn.role == "user":
                    messages.append({"role": "user", "content": turn.audio_text or turn.text or ""})
                elif turn.role == "assistant":
                    if turn.tool_calls:
                        tc_dicts = [tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in turn.tool_calls]
                        messages.append({
                            "role": "assistant",
                            "content": json.dumps({
                                "thought": f"Executing tool request from history",
                                "tool_calls": tc_dicts,
                                "is_finished": False,
                                "final_response": None
                            }, ensure_ascii=False)
                        })
                        if turn.tool_results:
                            messages.append({
                                "role": "user",
                                "content": f"OBSERVATIONS FROM TOOL EXECUTION:\n{json.dumps(turn.tool_results, ensure_ascii=False)}"
                            })
                        if turn.text:
                            messages.append({
                                "role": "assistant",
                                "content": json.dumps({
                                    "thought": "Completed history request",
                                    "tool_calls": [],
                                    "is_finished": True,
                                    "final_response": turn.text
                                }, ensure_ascii=False)
                            })
                    elif turn.text:
                        messages.append({
                            "role": "assistant",
                            "content": json.dumps({
                                "thought": "Direct conversational response from history",
                                "tool_calls": [],
                                "is_finished": True,
                                "final_response": turn.text
                            }, ensure_ascii=False)
                        })

            # 3. Add user voice request
            messages.append({"role": "user", "content": f"User Voice Request: \"{sample.audio_text or ''}\""})

            # 4. Multi-turn trajectory
            turns = rec.get("turns", [])
            for i, t in enumerate(turns):
                resp_payload = {
                    "thought": t.get("thought", ""),
                    "tool_calls": t.get("tool_calls", []),
                    "is_finished": t.get("is_finished", False),
                    "final_response": t.get("final_response")
                }
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(resp_payload, ensure_ascii=False)
                })

                obs = t.get("observations", [])
                if obs and not t.get("is_finished", False) and i < len(turns) - 1:
                    messages.append({
                        "role": "user",
                        "content": f"OBSERVATIONS FROM TOOL EXECUTION:\n{json.dumps(obs, ensure_ascii=False, indent=2)}\n\nPlease provide your next action or final response."
                    })

            # Teacher identification
            if sid in sol_ids:
                teacher = "gpt-6.1-sol"
            elif sid in terra_ids:
                teacher = "gpt-6.1-terra"
            else:
                teacher = "gpt-luna"

            item = {
                "id": sid,
                "teacher": teacher,
                "score": rec.get("score", 1.0),
                "state_match": True,
                "audio_text": sample.audio_text or "",
                "total_turns": len(turns),
                "messages": messages
            }
            sft_data.append(item)

    print(f"Total verified SFT samples compiled: {len(sft_data)}")

    # Save as JSON array
    json_path = DISTILLATION_DIR / "sft_train_1526_samples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON dataset to: {json_path}")

    # Also save as JSONL
    jsonl_path = DISTILLATION_DIR / "sft_train_1526_samples.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved JSONL dataset to: {jsonl_path}")

    # Print teacher summary
    counts = {}
    for item in sft_data:
        t = item["teacher"]
        counts[t] = counts.get(t, 0) + 1
    print("\nTeacher Distribution:")
    for t, c in counts.items():
        print(f"  {t:15s}: {c:4d} samples ({c/len(sft_data)*100:.2f}%)")

if __name__ == "__main__":
    export_dataset()

