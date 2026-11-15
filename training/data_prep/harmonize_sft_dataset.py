"""
Harmonization Engine for LifeMate SFT Training Trajectories.

Features:
- Implements 3-tier synchronization: System Prompt -> Thought -> Action.
- Non-destructive: outputs to new files without overwriting originals.
- Minimal Arguments & Implicit Enum[0] cleanup.
- Unfiltered Read Tools cleanup (resets to {} and explains in thought).
- Memory-First & History-Reuse handling (aligns with gold [] when history already has context).
- Evaluates against the live Zaic Sandbox to verify 100% State Match.
- Exports to:
    1. training/llm/data/distillation/harmonized_sft_1526_samples.json
    2. training/llm/data/distillation/harmonized_sft_1526_samples.jsonl
    3. training/llm/data/sft/harmonized_train_chatml.jsonl (1,450 train)
    4. training/llm/data/sft/harmonized_val_chatml.jsonl (76 val)
    5. training/llm/data/sft/harmonized_dataset_info.json
"""

import os
import sys
import json
import re
import random
from pathlib import Path
import tiktoken
import numpy as np

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))
from paths import (PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR, LLM_OUTPUT_DIR)

WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from lifemate.domain.entities import SampleContext, Sample
from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory
from lifemate.adapters.storage.jsonl_dataset_reader import JsonlDatasetReader
from lifemate.evaluation.metrics import compute_sample_score, call_matches

enc = tiktoken.get_encoding("cl100k_base")

def extract_full_installed_schemas(catalog: dict, installed_apps: dict) -> list[dict]:
    allowed_tool_names = set()
    for app_id, app_info in installed_apps.items():
        if isinstance(app_info, dict) and "tools" in app_info:
            allowed_tool_names.update(app_info["tools"])
        elif isinstance(app_info, list):
            allowed_tool_names.update(app_info)

    allowed_tool_names.add("sys.ask_missing_info")

    full_schemas = []
    for app in catalog.get("apps", []):
        app_id = app.get("id", "")
        for tool in app.get("tools", []):
            name = tool.get("name", "")
            if name in allowed_tool_names:
                full_schemas.append({
                    "name": name,
                    "app": app_id,
                    "kind": tool.get("kind", "write"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                    "returns": tool.get("returns", "")
                })
    return full_schemas

def build_harmonized_system_prompt(context: SampleContext, tool_schemas: list[dict]) -> str:
    compact_schemas = json.dumps(tool_schemas, ensure_ascii=False, separators=(',', ':'))
    return f"""You are LifeMate, an autonomous voice assistant for Vietnamese users.
You translate natural Vietnamese voice requests into concrete, precise tool calls to manage user apps and devices.

### CURRENT CONTEXT:
- Current Time: {context.current_time}
- Timezone: {context.timezone}
- Location: {json.dumps(context.location, ensure_ascii=False)}
- User Profile: {json.dumps(context.user_profile, ensure_ascii=False)}

### AVAILABLE TOOLS (FULL SCHEMAS):
Below are the exact schemas of all installed tools authorized for this session. You MUST strictly adhere to parameter names, types, and formats.
{compact_schemas}

### REACT OPERATING PROTOCOL:
You operate in an iterative closed-loop:
1. Analyze user request, conditional logic (e.g. if weather is rainy, reschedule; otherwise do nothing), relative time references ("mai", "3 giờ chiều"), and context.
2. READ-BEFORE-WRITE: If a write action requires specific entity IDs (e.g. event_id, contact_id, device_id), search/list them first using [read] tools.
3. IN-SCOPE vs OUT-OF-SCOPE: If user request cannot be fulfilled by any installed tool, return empty tool_calls with is_finished=true.
4. AMBIGUITY & MISSING INFO PROTOCOL (STRICT MANDATORY TOOL CALL):
   - Whenever an entity search returns MULTIPLE candidates (e.g. 2 contacts named "Minh", 2 chats named "Lan"), or when a required parameter cannot be determined with certainty:
   - You MUST EXPLICITLY INVOKE `sys.ask_missing_info` with the intended write tool name and the ambiguous/missing parameter(s):
     {{"name": "sys.ask_missing_info", "arguments": {{"tool_name": "<intended_tool_name>", "params": ["<param_name>"]}}}}
   - NEVER ask the user only via natural language (final_response) without invoking `sys.ask_missing_info`! Calling `sys.ask_missing_info` is the obligatory system event to trigger user clarification.

### PARAMETER INVOCATION RULES (GENERIC MINIMAL ARGUMENTS CONVENTION):
1. MINIMAL ARGUMENT CALLS: Only pass parameters that are strictly required by the schema (`required`) or specifically demanded by the user. Never pass optional parameters with default values.
2. IMPLICIT ENUM[0] DEFAULT: For any optional parameter with an `enum` list, the first element (`enum[0]`) is the platform's implicit default value. NEVER pass an optional parameter if its value matches `enum[0]`. Only specify it when the user explicitly requests a non-default choice (`enum[1:]`).
3. UNFILTERED READ TOOLS: For data query and inspection tools (e.g., tools whose names begin with `list_` or `get_status`), invoke them with an empty argument object {{}} so the environment returns the complete collection; perform all necessary filtering in memory.
4. MEMORY-FIRST & HISTORY REUSE: If the requested information (e.g. weather forecast, wallet balance, contact info) has ALREADY been queried and observed in the conversation history, you MUST reuse that observed data directly from memory. NEVER invoke the read tool again to query already-known information. If a condition evaluates to no action required, return empty tool_calls with is_finished=true.
5. CONCISE FREE-TEXT: For free-text parameters (messages, notes, task titles), use concise text matching the user's spoken meaning without adding artificial conversational prefixes or polite padding.

### OUTPUT FORMAT:
You MUST respond STRICTLY in JSON format:
{{
  "thought": "<Detailed step-by-step reasoning explaining the logic, conditions checked, and tools selected>",
  "tool_calls": [
    {{
      "name": "<tool_name>",
      "arguments": {{
        "<parameter_name>": <value>
      }}
    }}
  ],
  "is_finished": false,
  "final_response": null
}}

When all actions are completed, or no actions are needed, return:
{{
  "thought": "<Explanation that all tasks are finished or out of scope>",
  "tool_calls": [],
  "is_finished": true,
  "final_response": "<Natural Vietnamese reply to the user>"
}}
"""

def clean_tool_call(call: dict, tool_schemas_map: dict, gold_calls: list[dict]) -> tuple[dict, list[str]]:
    """
    Cleans a single tool call according to the 5 Golden Rules and returns:
    (cleaned_call, list_of_reasoning_notes_for_thought)
    """
    name = call.get("name", "")
    args = dict(call.get("arguments", {}))
    schema = tool_schemas_map.get(name, {})
    params_schema = schema.get("parameters", {}).get("properties", {})
    required_keys = set(schema.get("parameters", {}).get("required", []))
    action_part = name.split(".")[-1]

    notes = []

    # Find if there is a matching gold tool call for reference
    matching_gold = None
    for g in gold_calls:
        if g.get("name") == name:
            matching_gold = g
            break

    # 1. Unfiltered Read Tools: list_*, get_status, guide, search_place
    if action_part.startswith("list_") or action_part.startswith("get_status") or action_part in ["guide", "list"]:
        if args != {}:
            args = {}
            notes.append(f"Theo quy tắc Unfiltered Read Tools, gọi `{name}({{}})` với tham số rỗng để lấy toàn bộ danh sách và lọc an toàn trong bộ nhớ.")
        return {"name": name, "arguments": args}, notes

    # 2. Write tools argument filtering
    for k, v in list(args.items()):
        if k not in required_keys:
            prop = params_schema.get(k, {})
            # Enum[0] check
            if "enum" in prop and prop["enum"] and v == prop["enum"][0]:
                del args[k]
                notes.append(f"Lược bỏ tham số `{k}` vì '{v}' là giá trị mặc định ngầm định (enum[0]) của hệ thống.")
                continue

            # Optional note, label, repeat on common tools if gold doesn't have it
            if k in ["label", "note", "description", "category", "repeat", "title", "reason"]:
                gold_has_it = matching_gold and k in matching_gold.get("arguments", {})
                if not gold_has_it:
                    del args[k]
                    notes.append(f"Theo quy tắc tối giản tham số (Minimal Arguments), không truyền trường `{k}` phụ.")

    return {"name": name, "arguments": args}, notes

def harmonize_sample_trajectory(
    sample: Sample,
    rec: dict,
    tool_schemas_map: dict,
    factory: ZaicSandboxEnvFactory
) -> dict:
    s_id = sample.id
    gold_calls = sample.label.get("tool_calls", []) if sample.label else []
    pred_calls = rec.get("predicted_tool_calls", [])
    turns = list(rec.get("turns", []))

    # Calculate sandbox hash for gold
    gold_session = factory.create_session("train", s_id)
    for gc in gold_calls:
        gold_session.call(gc["name"], gc.get("arguments", {}))
    gold_hash = gold_session.get_state_hash()

    # Special Case: Gold is empty [] (Condition false / Out-of-scope / Handled in history)
    # E.g. train_0082
    if len(gold_calls) == 0:
        # Check if history had relevant observations
        has_history_obs = any(t.tool_results for t in sample.history if t.role == "assistant")
        if has_history_obs:
            thought_text = (
                "Kiểm tra lịch sử hội thoại: thông tin cần thiết đã được truy vấn và xác nhận ở các lượt trước. "
                "Tuân thủ quy tắc Memory-First, tái sử dụng trực tiếp kết quả từ lịch sử mà không gọi lại API. "
                "Xét điều kiện yêu cầu, không cần thực hiện thêm hành động nào."
            )
        else:
            thought_text = (
                "Yêu cầu của người dùng không thỏa mãn điều kiện thực thi hoặc nằm ngoài phạm vi các công cụ được cài đặt. "
                "Không thực hiện hành động nào theo quy tắc Out-of-Scope / Condition Check."
            )

        new_turn = {
            "turn": 1,
            "elapsed_seconds": 0.5,
            "thought": thought_text,
            "tool_calls": [],
            "is_finished": True,
            "final_response": "Yêu cầu của bạn đã được kiểm tra từ thông tin trước đó và hiện không cần thay đổi nào.",
            "observations": []
        }

        eval_res = compute_sample_score([], gold_calls, gold_hash, gold_hash)

        return {
            "id": s_id,
            "teacher": rec.get("teacher", "gpt-luna"),
            "audio_text": sample.audio_text or "",
            "current_time": sample.context.current_time,
            "turns": [new_turn],
            "predicted_tool_calls": [],
            "gold_tool_calls": gold_calls,
            "predicted_state_hash": gold_hash,
            "gold_state_hash": gold_hash,
            "state_match": True,
            "score": eval_res["score"],
            "precision": eval_res["precision"],
            "recall": eval_res["recall"],
            "f1": eval_res["f1"],
            "notes": ["Memory-First / Zero-call condition aligned with Gold"]
        }

    # Standard Case: Harmonize existing turns
    all_harmonized_calls = []
    harmonized_turns = []
    session = factory.create_session("train", s_id)

    for turn_idx, t in enumerate(turns):
        t_calls = t.get("tool_calls", [])
        thought = t.get("thought", "")
        is_finished = t.get("is_finished", False)
        final_resp = t.get("final_response")

        new_t_calls = []
        turn_notes = []

        for call in t_calls:
            clean_call, notes = clean_tool_call(call, tool_schemas_map, gold_calls)
            new_t_calls.append(clean_call)
            turn_notes.extend(notes)
            all_harmonized_calls.append(clean_call)

        # Harmonize thought by appending the notes seamlessly if any rule was applied
        if turn_notes:
            unique_notes = list(dict.fromkeys(turn_notes))
            thought_addition = " " + " ".join(unique_notes)
            # Avoid repeating if already in thought
            if "quy tắc" not in thought.lower():
                thought = thought.rstrip(". ") + "." + thought_addition

        # Execute in sandbox to record exact observation
        observations = []
        if new_t_calls:
            for c in new_t_calls:
                obs = session.call(c["name"], c.get("arguments", {}))
                observations.append({
                    "tool_name": c["name"],
                    "ok": obs.ok,
                    "result": obs.result
                })

        harmonized_turns.append({
            "turn": turn_idx + 1,
            "elapsed_seconds": t.get("elapsed_seconds", 1.0),
            "thought": thought,
            "tool_calls": new_t_calls,
            "is_finished": is_finished,
            "final_response": final_resp,
            "observations": observations
        })

    pred_hash = session.get_state_hash()
    eval_res = compute_sample_score(all_harmonized_calls, gold_calls, pred_hash, gold_hash)

    # Sanity check: If for some reason state_hash diverged, keep original calls
    if not eval_res["state_match"]:
        # Fallback to original
        return rec

    return {
        "id": s_id,
        "teacher": rec.get("teacher", "gpt-luna"),
        "audio_text": sample.audio_text or "",
        "current_time": sample.context.current_time,
        "turns": harmonized_turns,
        "predicted_tool_calls": all_harmonized_calls,
        "gold_tool_calls": gold_calls,
        "predicted_state_hash": pred_hash,
        "gold_state_hash": gold_hash,
        "state_match": eval_res["state_match"],
        "score": eval_res["score"],
        "precision": eval_res["precision"],
        "recall": eval_res["recall"],
        "f1": eval_res["f1"]
    }

def build_chatml_messages(sample: Sample, catalog: dict, harmonized_rec: dict) -> list[dict]:
    context = sample.context
    tool_schemas = extract_full_installed_schemas(catalog, context.installed_apps)
    system_prompt = build_harmonized_system_prompt(context, tool_schemas)

    messages = [{"role": "system", "content": system_prompt}]

    # History turns
    for turn in sample.history:
        if turn.role == "user":
            messages.append({"role": "user", "content": turn.audio_text or turn.text or ""})
        elif turn.role == "assistant":
            if turn.tool_calls:
                tc_dicts = [tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in turn.tool_calls]
                messages.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "thought": "Executing tool request from history",
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

    # User current voice request
    messages.append({"role": "user", "content": f"User Voice Request: \"{sample.audio_text or ''}\""})

    # Harmonized multi-turn trajectory
    turns = harmonized_rec.get("turns", [])
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

    return messages

def main():
    print("=" * 80)
    print("🚀 LAUNCHING HARMONIZATION ENGINE (NON-DESTRUCTIVE)")
    print("=" * 80)

    # 1. Load Data
    reader = JsonlDatasetReader(DATA_ROOT)
    all_samples = {s.id: s for s in reader.read_samples("train")}
    catalog = json.load(open(DATA_ROOT / "train" / "train_tools_catalog.json", encoding="utf-8"))
    factory = ZaicSandboxEnvFactory(DATA_ROOT)

    tool_schemas_map = {}
    for app in catalog.get("apps", []):
        for tool in app.get("tools", []):
            tool_schemas_map[tool["name"]] = tool

    master_path = DISTILLATION_DIR / "master_train_distilled.jsonl"
    if not master_path.is_file():
        print(f"Error: {master_path} not found!")
        return

    verified_records = []
    with open(master_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d.get("state_match", False):
                    verified_records.append(d)

    print(f"Loaded {len(verified_records)} verified samples (state_match == True).")

    # Baseline metrics
    orig_scores = [r.get("score", 0.0) for r in verified_records]
    orig_f1s = [r.get("f1", 0.0) for r in verified_records]
    orig_perfect = sum(1 for s in orig_scores if s == 1.0)

    print(f"Baseline Average Score: {np.mean(orig_scores)*100:.2f}%")
    print(f"Baseline Average F1:    {np.mean(orig_f1s)*100:.2f}%")
    print(f"Baseline Perfect 1.00:  {orig_perfect}/{len(verified_records)} ({orig_perfect/len(verified_records)*100:.2f}%)")

    # 2. Harmonize all samples
    print("\nStarting harmonization across all samples...")
    harmonized_records = []
    sft_items = []
    token_lengths = []

    for idx, rec in enumerate(verified_records, 1):
        sid = rec["id"]
        sample = all_samples[sid]
        h_rec = harmonize_sample_trajectory(sample, rec, tool_schemas_map, factory)
        harmonized_records.append(h_rec)

        # Build ChatML messages
        messages = build_chatml_messages(sample, catalog, h_rec)
        full_text = "".join(m["content"] for m in messages)
        n_tok = len(enc.encode(full_text))
        token_lengths.append(n_tok)

        sft_items.append({
            "id": sid,
            "teacher": rec.get("teacher", "gpt-luna"),
            "score": h_rec["score"],
            "f1": h_rec["f1"],
            "state_match": h_rec["state_match"],
            "audio_text": sample.audio_text or "",
            "total_turns": len(h_rec["turns"]),
            "token_count": n_tok,
            "messages": messages
        })

        if idx % 200 == 0 or idx == len(verified_records):
            curr_scores = [r["score"] for r in harmonized_records]
            curr_perf = sum(1 for s in curr_scores if s == 1.0)
            print(f"  Processed [{idx:4d}/{len(verified_records)}] | Running Avg Score: {np.mean(curr_scores)*100:.2f}% | Perfect: {curr_perf}")

    # 3. Post-Harmonization Benchmark Results
    new_scores = [r["score"] for r in harmonized_records]
    new_f1s = [r["f1"] for r in harmonized_records]
    new_perfect = sum(1 for s in new_scores if s == 1.0)
    all_matched = all(r["state_match"] for r in harmonized_records)

    print("\n" + "=" * 80)
    print("🏆 HARMONIZATION COMPLETE! BENCHMARK QUALITY METRICS:")
    print(f"Total Samples:            {len(harmonized_records)}")
    print(f"State Match 100% Retained:{all_matched} ({sum(1 for r in harmonized_records if r['state_match'])}/{len(harmonized_records)})")
    print(f"Average Overall Score:    {np.mean(new_scores)*100:.2f}% (was {np.mean(orig_scores)*100:.2f}%)")
    print(f"Average Tool Call F1:     {np.mean(new_f1s)*100:.2f}% (was {np.mean(orig_f1s)*100:.2f}%)")
    print(f"Perfect 1.00 Count:       {new_perfect}/{len(harmonized_records)} ({new_perfect/len(harmonized_records)*100:.2f}%) (was {orig_perfect})")
    print(f"Token Length -> Mean: {np.mean(token_lengths):.1f}, Max: {np.max(token_lengths)}, Min: {np.min(token_lengths)}")
    print("=" * 80)

    # 4. Save to NEW standalone files (NO OVERWRITE of originals)
    out_dir = DISTILLATION_DIR
    out_json = out_dir / "harmonized_sft_1526_samples.json"
    out_jsonl = out_dir / "harmonized_sft_1526_samples.jsonl"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sft_items, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved new harmonized JSON:  {out_json}")

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item in sft_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Saved new harmonized JSONL: {out_jsonl}")

    # 5. Export Stratified Train / Val Splits in training/llm/data/sft/
    sft_dir = SFT_DIR
    sft_dir.mkdir(parents=True, exist_ok=True)

    # Stratify by teacher
    teachers = {}
    for item in sft_items:
        t = item.get("teacher", "gpt-luna")
        teachers.setdefault(t, []).append(item)

    random.seed(42)
    train_items = []
    val_items = []
    for t_name, group in teachers.items():
        random.shuffle(group)
        n_val = max(1, int(len(group) * 0.05))
        val_items.extend(group[:n_val])
        train_items.extend(group[n_val:])

    train_path = sft_dir / "harmonized_train_chatml.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps({"id": item["id"], "messages": item["messages"]}, ensure_ascii=False) + "\n")

    val_path = sft_dir / "harmonized_val_chatml.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_items:
            f.write(json.dumps({"id": item["id"], "messages": item["messages"]}, ensure_ascii=False) + "\n")

    info_path = sft_dir / "harmonized_dataset_info.json"
    info = {
        "dataset_name": "LifeMate-Harmonized-SFT-1526",
        "description": "Harmonized 3-tier SFT dataset with 100% thought-action consistency and minimal arguments.",
        "total_samples": len(sft_items),
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "metrics": {
            "state_match_rate": 1.0,
            "average_score": float(np.mean(new_scores)),
            "average_f1": float(np.mean(new_f1s)),
            "perfect_score_count": int(new_perfect)
        },
        "token_statistics": {
            "mean": float(np.mean(token_lengths)),
            "max": int(np.max(token_lengths)),
            "min": int(np.min(token_lengths))
        },
        "files": {
            "master_json": str(out_json.name),
            "master_jsonl": str(out_jsonl.name),
            "train_chatml": str(train_path.name),
            "val_chatml": str(val_path.name)
        }
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved new train split ({len(train_items)} samples): {train_path}")
    print(f"✅ Saved new val split ({len(val_items)} samples):     {val_path}")
    print(f"✅ Saved metadata:                                    {info_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()

