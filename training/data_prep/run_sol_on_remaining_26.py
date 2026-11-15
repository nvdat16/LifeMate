"""
Run gpt-6.1-sol (reasoning_effort="low") on the remaining 26 failed samples.

Goal:
- Use gpt-6.1-sol with 'low' reasoning effort to test whether a less pedantic,
  pragmatic reasoning level can overcome the overthinking traps of frontier medium models
  while retaining enough intelligence to resolve latent context and hidden intentions.
- Saves results to training/llm/data/distillation/sol_26_samples_repair.jsonl.
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

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))
from paths import (PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR, LLM_OUTPUT_DIR)

WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(DATA_ROOT / "public_test"))

import openai
from lifemate.domain.entities import SampleContext, Sample
from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory
from lifemate.adapters.storage.jsonl_dataset_reader import JsonlDatasetReader
from lifemate.evaluation.metrics import compute_sample_score
from run_terra_on_failed_samples import extract_full_installed_schemas, build_system_prompt, extract_json_from_raw_text

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def require_openai_api_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")
    return OPENAI_API_KEY

MODEL_NAME = "gpt-6.1-sol"
REASONING_EFFORT = "low"

def process_single_sample(
    sample: Sample,
    catalog: dict,
    factory: ZaicSandboxEnvFactory,
    client: openai.OpenAI,
    model_name: str = MODEL_NAME,
    reasoning_effort: str = REASONING_EFFORT,
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
        
        raw_out = ""
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    reasoning_effort=reasoning_effort
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
    parser = argparse.ArgumentParser(description="Run Sol on Remaining 26 Samples")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent worker threads")
    parser.add_argument("--effort", type=str, default=REASONING_EFFORT, help="Reasoning effort (low, medium)")
    parser.add_argument("--output", type=str, default=str(DISTILLATION_DIR / "sol_26_samples_repair.jsonl"), help="Output path")
    args = parser.parse_args()

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Identify the 26 unrepaired sample IDs from Terra run
    terra_file = DISTILLATION_DIR / "terra_failed_samples_repair.jsonl"
    if not terra_file.is_file():
        print(f"Error: {terra_file} not found!")
        return

    failed_ids = []
    with open(terra_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if not d.get("state_match", False):
                    failed_ids.append(d["id"])

    print("=" * 80)
    print(f"🚀 RUNNING {MODEL_NAME} (effort='{args.effort}') ON {len(failed_ids)} REMAINING SAMPLES")
    print(f"📁 Output JSONL:       {output_path}")
    print(f"🧵 Parallel Workers:    {args.workers}")
    print("=" * 80)

    # Load samples
    reader = JsonlDatasetReader(DATA_ROOT)
    all_samples = {s.id: s for s in reader.read_samples("train")}
    target_samples = [all_samples[sid] for sid in failed_ids if sid in all_samples]

    # Check already processed
    done_ids = set()
    repaired_count = 0
    if output_path.is_file():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        done_ids.add(record["id"])
                        if record.get("state_match", False):
                            repaired_count += 1
                    except Exception:
                        pass

    pending_samples = [s for s in target_samples if s.id not in done_ids]
    print(f"Pending to run: {len(pending_samples)} (Already done: {len(done_ids)})")

    if not pending_samples:
        print("All target samples already processed!")
        return

    catalog_path = DATA_ROOT / "train" / "train_tools_catalog.json"
    catalog = json.load(open(catalog_path, encoding="utf-8"))
    factory = ZaicSandboxEnvFactory(DATA_ROOT)
    client = openai.OpenAI(api_key=require_openai_api_key())
    file_lock = threading.Lock()

    completed = len(done_ids)
    start_t = time.time()

    def worker_fn(sample: Sample):
        res = process_single_sample(sample, catalog, factory, client, MODEL_NAME, args.effort)
        with file_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        return res

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_fn, s): s for s in pending_samples}
        for future in as_completed(futures):
            s = futures[future]
            try:
                res = future.result()
                completed += 1
                is_match = res["state_match"]
                if is_match:
                    repaired_count += 1
                status_icon = "✅ REPAIRED" if is_match else "❌ STILL MISMATCHED"
                print(
                    f"[{completed:2d}/{len(target_samples)}] {res['id']} -> {status_icon} "
                    f"| Score: {res['score']:.2f} | Total Sol Repaired so far: {repaired_count} "
                    f"({repaired_count / completed * 100:.1f}%)"
                )
            except Exception as e:
                print(f"Error on {s.id}: {e}")

    total_elapsed = time.time() - start_t
    print("\n" + "=" * 80)
    print("🏆 SOL EVALUATION ON REMAINING 26 SAMPLES COMPLETE!")
    print(f"Total time elapsed: {total_elapsed / 60:.1f} minutes")
    print(f"Total Repaired by Sol: {repaired_count}/{len(target_samples)} ({repaired_count / len(target_samples) * 100:.1f}%)")
    print(f"Output saved to: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()

