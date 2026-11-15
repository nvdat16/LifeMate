"""
Evaluate a Trained SFT Checkpoint on the 76 Validation Samples using the ZAIC Sandbox Environment.

Features:
- Dual Inference Mode:
    1. vLLM / OpenAI Server Mode (Fastest, via --api_base)
    2. Direct Python In-Process Mode (via --checkpoint_dir using Unsloth/HuggingFace)
- Runs iterative ReAct loops on harmonized_val_chatml.jsonl.
- Executes tool calls against the real ZaicSandboxEnvFactory to compute actual state_hash matches.
- Prints exact benchmark metrics (State Match Rate, F1, Precision, Recall).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAINING_ROOT))
from paths import (PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR, LLM_OUTPUT_DIR)

WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(TRAINING_ROOT / "data_prep"))
sys.path.insert(0, str(DATA_ROOT / "public_test"))

from lifemate.domain.entities import Sample
from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory
from lifemate.adapters.storage.jsonl_dataset_reader import JsonlDatasetReader
from lifemate.evaluation.metrics import compute_sample_score
from run_terra_on_failed_samples import extract_json_from_raw_text

def evaluate_checkpoint():
    parser = argparse.ArgumentParser(description="Evaluate SFT Checkpoint on Validation Set")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to local checkpoint directory (e.g. training/llm/outputs/.../checkpoint-360)")
    parser.add_argument("--val_file", type=str, default=str(SFT_DIR / "harmonized_val_chatml.jsonl"), help="Validation file path")
    parser.add_argument("--api_base", type=str, default=None, help="vLLM / local server URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key")
    parser.add_argument("--model_name", type=str, default="default", help="Model name on server")
    parser.add_argument("--chat_template", choices=["auto", "qwen-2.5", "qwen3-instruct"], default="auto", help="Chat template for direct checkpoint evaluation")
    parser.add_argument("--max_seq_length", type=int, default=16384, help="Inference context length, capped at 16K for this competition")
    args = parser.parse_args()

    if args.max_seq_length <= 0 or args.max_seq_length > 16384:
        raise ValueError("--max_seq_length must be between 1 and 16384 tokens for this competition.")

    val_path = Path(args.val_file).resolve()
    if not val_path.is_file():
        # Fallback relative to workspace root
        val_path = WORKSPACE_ROOT / args.val_file
    if not val_path.is_file():
        print(f"Error: {val_path} not found!")
        return

    # Determine execution mode: API Server vs Direct Python In-process
    use_api = args.api_base is not None
    client = None
    model = None
    tokenizer = None

    if use_api:
        import openai
        client = openai.OpenAI(base_url=args.api_base, api_key=args.api_key)
        print(f"🌐 Running via vLLM / API Server: {args.api_base}")
    elif args.checkpoint_dir is not None:
        ckpt_path = Path(args.checkpoint_dir).resolve()
        if not ckpt_path.exists():
            ckpt_path = WORKSPACE_ROOT / args.checkpoint_dir
        if not ckpt_path.exists():
            print(f"Error: Checkpoint path not found: {ckpt_path}")
            return

        print(f"📦 Loading Checkpoint directly in Python: {ckpt_path}")
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        chat_template = args.chat_template
        if chat_template == "auto":
            chat_template = "qwen3-instruct" if "qwen3" in str(ckpt_path).lower() else "qwen-2.5"

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(ckpt_path),
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    else:
        # Default fallback to localhost:8000 if neither is explicitly passed
        import openai
        args.api_base = "http://localhost:8000/v1"
        client = openai.OpenAI(base_url=args.api_base, api_key=args.api_key)
        use_api = True
        print(f"🌐 Running via default vLLM URL: {args.api_base}")

    reader = JsonlDatasetReader(DATA_ROOT)
    all_samples = {s.id: s for s in reader.read_samples("train")}
    factory = ZaicSandboxEnvFactory(DATA_ROOT)

    val_records = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_records.append(json.loads(line))

    print("=" * 80)
    print(f"🧪 EVALUATING CHECKPOINT ON {len(val_records)} VALIDATION SAMPLES")
    print(f"Validation File: {val_path}")
    print("=" * 80)

    matched_count = 0
    scores = []
    f1s = []

    for idx, item in enumerate(val_records, 1):
        sid = item["id"]
        sample = all_samples.get(sid)
        if not sample:
            continue

        session = factory.create_session("train", sid)
        executed_calls = []

        # Construct initial messages from prompt
        messages = [m for m in item["messages"] if m["role"] == "system" or (m["role"] == "user" and "User Voice Request:" in m["content"])]

        # Run ReAct loop
        for turn in range(4):
            raw_out = ""
            if use_api:
                try:
                    resp = client.chat.completions.create(
                        model=args.model_name,
                        messages=messages,
                        temperature=0.0
                    )
                    raw_out = resp.choices[0].message.content or ""
                except Exception as e:
                    print(f"Server error on {sid}: {e}")
                    break
            else:
                # Direct Python Generation via Unsloth
                import torch
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        use_cache=True,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                raw_out = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            parsed = extract_json_from_raw_text(raw_out)
            tool_calls = parsed.get("tool_calls", [])
            is_finished = parsed.get("is_finished", False)

            observations = []
            if tool_calls:
                for c in tool_calls:
                    c_name = c.get("name", "")
                    c_args = c.get("arguments", {})
                    obs = session.call(c_name, c_args)
                    executed_calls.append({"name": c_name, "arguments": c_args})
                    observations.append({"tool_name": c_name, "ok": obs.ok, "result": obs.result})

            if is_finished or not tool_calls:
                break

            messages.append({"role": "assistant", "content": raw_out})
            messages.append({
                "role": "user",
                "content": f"OBSERVATIONS FROM TOOL EXECUTION:\n{json.dumps(observations, ensure_ascii=False, indent=2)}\n\nPlease provide your next action or final response."
            })

        pred_hash = session.get_state_hash()
        gold_calls = sample.label.get("tool_calls", []) if sample.label else []
        gold_session = factory.create_session("train", sid)
        for gc in gold_calls:
            gold_session.call(gc["name"], gc.get("arguments", {}))
        gold_hash = gold_session.get_state_hash()

        eval_res = compute_sample_score(executed_calls, gold_calls, pred_hash, gold_hash)
        is_match = eval_res["state_match"]
        if is_match:
            matched_count += 1
        scores.append(eval_res["score"])
        f1s.append(eval_res["f1"])

        status_icon = "✅ MATCH" if is_match else "❌ MISMATCH"
        print(f"[{idx:2d}/{len(val_records)}] {sid} -> {status_icon} | Score: {eval_res['score']:.2f} | Running Match Rate: {matched_count/idx*100:.1f}%")

    print("\n" + "=" * 80)
    print("🏆 VALIDATION BENCHMARK RESULTS:")
    print(f"Total Samples Evaluated: {len(val_records)}")
    print(f"State Match Rate:        {matched_count} / {len(val_records)} ({matched_count/len(val_records)*100:.2f}%)")
    print(f"Average Score:           {sum(scores)/len(scores)*100:.2f}%")
    print(f"Average Tool Call F1:    {sum(f1s)/len(f1s)*100:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    evaluate_checkpoint()
