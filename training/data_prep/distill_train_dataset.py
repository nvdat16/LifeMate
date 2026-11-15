"""
Production Distillation Runner: Distill All 1,545 Train Samples with gpt-5.6-luna.

Features:
- 100% Generic ReAct prompt leveraging BTC's minimal argument & enum[0] default convention.
- Multi-threaded parallel processing (configurable workers).
- Resumable checkpointing: writes line-by-line to training/llm/data/distillation/train_distilled_trajectories.jsonl.
- Real-time accuracy, state hash match rate, and progress logging.
"""

import os
import sys
import json
import re
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))
from paths import (PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR, LLM_OUTPUT_DIR)

WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(DATA_ROOT / "public_test"))

import openai
from lifemate.domain.entities import SampleContext, Sample
from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory
from lifemate.adapters.storage.jsonl_dataset_reader import JsonlDatasetReader
from lifemate.evaluation.metrics import compute_sample_score

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def require_openai_api_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")
    return OPENAI_API_KEY


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

def build_system_prompt(context: SampleContext, tool_schemas: list[dict]) -> str:
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
4. MISSING INFO: If mandatory information is missing and cannot be retrieved via read tools, call `sys.ask_missing_info`.

### PARAMETER INVOCATION RULES (GENERIC MINIMAL ARGUMENTS CONVENTION):
1. MINIMAL ARGUMENT CALLS: Only pass parameters that are strictly required by the schema (`required`) or specifically demanded by the user. Never pass optional parameters with default values.
2. IMPLICIT ENUM[0] DEFAULT: For any optional parameter with an `enum` list, the first element (`enum[0]`) is the platform's implicit default value. NEVER pass an optional parameter if its value matches `enum[0]`. Only specify it when the user explicitly requests a non-default choice (`enum[1:]`).
3. UNFILTERED READ TOOLS: For data query and inspection tools (e.g., tools whose names begin with `list_` or `get_status`), invoke them with an empty argument object {{}} so the environment returns the complete collection; perform all necessary filtering in memory.
4. CONCISE FREE-TEXT: For free-text parameters (messages, notes, task titles), use concise text matching the user's spoken meaning without adding artificial conversational prefixes or polite padding.

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

def extract_json_from_raw_text(text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except Exception:
            pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {"thought": cleaned, "tool_calls": [], "is_finished": True, "final_response": cleaned}

def process_single_sample(
    sample: Sample,
    catalog: dict,
    factory: ZaicSandboxEnvFactory,
    client: openai.OpenAI,
    model_name: str,
    max_turns: int = 4
) -> dict:
    s_id = sample.id
    audio_text = sample.audio_text or ""
    context = sample.context

    tool_schemas = extract_full_installed_schemas(catalog, context.installed_apps)
    system_prompt = build_system_prompt(context, tool_schemas)

    messages = [{"role": "system", "content": system_prompt}]
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

    messages.append({"role": "user", "content": f"User Voice Request: \"{audio_text}\""})

    session = factory.create_session("train", s_id)
    executed_calls = []
    turn_records = []
    cur_turn = 0

    while cur_turn < max_turns:
        cur_turn += 1
        t_start = time.time()
        
        # Retry with exponential backoff on transient network errors
        raw_out = ""
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    reasoning_effort="medium"
                )
                raw_out = resp.choices[0].message.content or ""
                break
            except Exception as e:
                if attempt == 4:
                    raw_out = json.dumps({"thought": f"Error: {e}", "tool_calls": [], "is_finished": True})
                else:
                    time.sleep(1.5 + attempt * 2)

        elapsed = time.time() - t_start
        parsed = extract_json_from_raw_text(raw_out)
        thought = parsed.get("thought", "")
        tool_calls = parsed.get("tool_calls", [])
        is_finished = parsed.get("is_finished", False)
        final_resp = parsed.get("final_response")

        observations = []
        if tool_calls:
            for c in tool_calls:
                c_name = c.get("name", "")
                c_args = c.get("arguments", {})
                obs = session.call(c_name, c_args)
                executed_calls.append({"name": c_name, "arguments": c_args})
                observations.append({
                    "tool_name": c_name,
                    "ok": obs.ok,
                    "result": obs.result
                })

        turn_records.append({
            "turn": cur_turn,
            "elapsed_seconds": round(elapsed, 2),
            "thought": thought,
            "tool_calls": tool_calls,
            "is_finished": is_finished,
            "final_response": final_resp,
            "observations": observations
        })

        if is_finished or not tool_calls:
            break

        messages.append({"role": "assistant", "content": raw_out})
        messages.append({
            "role": "user",
            "content": f"OBSERVATIONS FROM TOOL EXECUTION:\n{json.dumps(observations, ensure_ascii=False, indent=2)}\n\nPlease provide your next action or final response."
        })

    pred_hash = session.get_state_hash()

    # Score against gold label
    gold_calls = sample.label.get("tool_calls", []) if sample.label else []
    gold_session = factory.create_session("train", s_id)
    for gc in gold_calls:
        gold_session.call(gc["name"], gc.get("arguments", {}))
    gold_hash = gold_session.get_state_hash()

    eval_res = compute_sample_score(executed_calls, gold_calls, pred_hash, gold_hash)

    return {
        "id": s_id,
        "audio_text": audio_text,
        "current_time": context.current_time,
        "turns": turn_records,
        "predicted_tool_calls": executed_calls,
        "gold_tool_calls": gold_calls,
        "predicted_state_hash": pred_hash,
        "gold_state_hash": gold_hash,
        "state_match": eval_res["state_match"],
        "score": eval_res["score"],
        "precision": eval_res["precision"],
        "recall": eval_res["recall"],
        "f1": eval_res["f1"]
    }

def main():
    parser = argparse.ArgumentParser(description="Distill Entire Train Dataset with Frontier ReAct Agent")
    parser.add_argument("--model", type=str, default="gpt-5.6-luna", help="Frontier model name")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent worker threads")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for testing")
    parser.add_argument("--output", type=str, default=str(DISTILLATION_DIR / "train_distilled_trajectories.jsonl"), help="Output path")
    args = parser.parse_args()

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    catalog_path = DATA_ROOT / "train" / "train_tools_catalog.json"
    catalog = json.load(open(catalog_path, encoding="utf-8"))

    reader = JsonlDatasetReader(DATA_ROOT)
    all_samples = list(reader.read_samples("train", limit=args.limit))
    total_count = len(all_samples)

    # Load existing processed IDs to resume gracefully
    done_ids = set()
    if output_path.is_file():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        done_ids.add(record["id"])
                    except Exception:
                        pass

    pending_samples = [s for s in all_samples if s.id not in done_ids]
    print("=" * 80)
    print(f"🚀 PRODUCTION DISTILLATION RUNNER — MODEL: {args.model}")
    print(f"📁 Output JSONL:       {output_path}")
    print(f"📊 Total Samples:       {total_count}")
    print(f"✅ Already Processed:   {len(done_ids)}")
    print(f"⏳ Remaining To Run:    {len(pending_samples)}")
    print(f"🧵 Parallel Workers:    {args.workers}")
    print("=" * 80)

    if not pending_samples:
        print("🎉 All samples have already been distilled! Nothing left to run.")
        return

    factory = ZaicSandboxEnvFactory(DATA_ROOT)
    client = openai.OpenAI(api_key=require_openai_api_key())
    file_lock = threading.Lock()

    completed_count = len(done_ids)
    cum_score = 0.0
    cum_matches = 0
    start_time = time.time()

    def worker_task(sample: Sample):
        res = process_single_sample(sample, catalog, factory, client, args.model)
        with file_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        return res

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_task, s): s for s in pending_samples}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                result = future.result()
                completed_count += 1
                cum_score += result["score"]
                if result["state_match"]:
                    cum_matches += 1

                elapsed = time.time() - start_time
                avg_score = (cum_score / (completed_count - len(done_ids))) * 100
                match_pct = (cum_matches / (completed_count - len(done_ids))) * 100
                rate = (completed_count - len(done_ids)) / elapsed if elapsed > 0 else 0
                eta_minutes = ((total_count - completed_count) / rate / 60) if rate > 0 else 0

                print(
                    f"[{completed_count:4d}/{total_count}] Sample {result['id']} "
                    f"| Score: {result['score']:.2f} | Match: {result['state_match']} "
                    f"| Batch Avg: {avg_score:.1f}% | Hash Match: {match_pct:.1f}% "
                    f"| ETA: {eta_minutes:.1f}m"
                )
            except Exception as e:
                print(f"❌ Error on sample {sample.id}: {e}")

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("🏆 DISTILLATION BATCH COMPLETE!")
    print(f"Total time elapsed: {total_elapsed / 60:.1f} minutes")
    print(f"Output saved to: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()

