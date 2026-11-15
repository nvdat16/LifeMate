"""
Apply Targeted Upgraded System Prompt (Ambiguity & Missing Info Protocol)
to specific candidate cases in the harmonized SFT dataset.

Rules:
1. ONLY apply the upgraded Ambiguity Protocol to samples where ambiguity or missing info is present:
   - Gold has sys.ask_missing_info (210 samples)
   - History turns involve sys.ask_missing_info (23 samples)
   - Other sys.ask_missing_info invocations (1 sample)
   Total: 233 samples.
2. For the 42 samples among these that were missing sys.ask_missing_info in current turn,
   harmonize their trajectory so sys.ask_missing_info is called, thoughts are aligned,
   and observations are authentically simulated via sandbox.
3. Keep the baseline generic system prompt for the remaining 1,293 samples as requested by user.
4. Verify 100% State Match across all 1,526 samples.
5. Update:
   - training/llm/data/distillation/harmonized_sft_1526_samples.json
   - training/llm/data/distillation/harmonized_sft_1526_samples.jsonl
   - training/llm/data/sft/harmonized_train_chatml.jsonl
   - training/llm/data/sft/harmonized_val_chatml.jsonl
   - training/llm/data/sft/harmonized_dataset_info.json
"""

import sys
import shutil
import json
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

from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory
from lifemate.evaluation.metrics import compute_sample_score

enc = tiktoken.get_encoding("cl100k_base")

UPGRADED_RULE = """4. AMBIGUITY & MISSING INFO PROTOCOL (STRICT MANDATORY TOOL CALL):
   - Whenever an entity search returns MULTIPLE candidates (e.g. 2 contacts named "Minh", 2 chats named "Lan"), or when a required parameter cannot be determined with certainty:
   - You MUST EXPLICITLY INVOKE `sys.ask_missing_info` with the intended write tool name and the ambiguous/missing parameter(s):
     {"name": "sys.ask_missing_info", "arguments": {"tool_name": "<intended_tool_name>", "params": ["<param_name>"]}}
   - NEVER ask the user only via natural language (final_response) without invoking `sys.ask_missing_info`! Calling `sys.ask_missing_info` is the obligatory system event to trigger user clarification."""

OLD_RULE = "4. MISSING INFO: If mandatory information is missing and cannot be retrieved via read tools, call `sys.ask_missing_info`."

def get_current_turn_calls(messages):
    prompt_idx = None
    for i, m in enumerate(messages):
        if m["role"] == "user" and "User Voice Request:" in m["content"]:
            prompt_idx = i
            break
    if prompt_idx is None:
        prompt_idx = [i for i, m in enumerate(messages) if m["role"] == "user"][-1]
    
    calls = []
    for m in messages[prompt_idx:]:
        if m["role"] == "assistant":
            try:
                data = json.loads(m["content"])
                tcs = data.get("tool_calls", [])
                calls.extend(tcs)
            except:
                pass
    return calls

def main():
    print("=" * 80)
    print("🚀 APPLYING TARGETED AMBIGUITY SYSTEM PROMPT & TRAJECTORY HARMONIZATION")
    print("=" * 80)

    json_path = DISTILLATION_DIR / "harmonized_sft_1526_samples.json"
    bak_path = DISTILLATION_DIR / "harmonized_sft_1526_samples.json.bak"
    if not bak_path.exists():
        shutil.copyfile(json_path, bak_path)
        print(f"📦 Created backup: {bak_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    with open(DATA_ROOT / "train" / "train.jsonl", "r", encoding="utf-8") as f:
        train_samples = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

    factory = ZaicSandboxEnvFactory(DATA_ROOT)

    # 1. Identify Target Samples for Upgraded Prompt
    target_ids = set()
    for s in samples:
        sid = s["id"]
        # Gold check
        if sid in train_samples:
            gold_calls = train_samples[sid].get("label", {}).get("tool_calls", [])
            if any(c["name"] == "sys.ask_missing_info" for c in gold_calls):
                target_ids.add(sid)
        # Message check
        for m in s["messages"]:
            if m["role"] == "assistant":
                try:
                    c = json.loads(m["content"])
                    if any(tc.get("name") == "sys.ask_missing_info" for tc in c.get("tool_calls", [])):
                        target_ids.add(sid)
                except:
                    pass

    print(f"📊 Target cases receiving UPGRADED prompt: {len(target_ids)} / {len(samples)}")
    print(f"📊 Standard cases retaining BASELINE prompt:  {len(samples) - len(target_ids)} / {len(samples)}")

    # 2. Identify the 42 samples missing sys.ask_missing_info in current turn
    missing_ids = []
    for sid in target_ids:
        s_gold = train_samples.get(sid)
        if not s_gold:
            continue
        g_calls = s_gold.get("label", {}).get("tool_calls", [])
        if any(c["name"] == "sys.ask_missing_info" for c in g_calls):
            # find sample
            s_obj = next(x for x in samples if x["id"] == sid)
            c_calls = get_current_turn_calls(s_obj["messages"])
            if not any(c["name"] == "sys.ask_missing_info" for c in c_calls):
                missing_ids.append(sid)

    print(f"🔧 Target cases requiring trajectory harmonization: {len(missing_ids)} samples")

    # 3. Apply updates
    updated_count_prompt = 0
    updated_count_trajectory = 0

    samples_dict = {s["id"]: s for s in samples}

    for sid in target_ids:
        s = samples_dict[sid]
        msgs = s["messages"]
        # Update system prompt
        if OLD_RULE in msgs[0]["content"]:
            msgs[0]["content"] = msgs[0]["content"].replace(OLD_RULE, UPGRADED_RULE)
            updated_count_prompt += 1

        # If in missing_ids, harmonize trajectory
        if sid in missing_ids:
            s_gold = train_samples[sid]
            gold_calls = s_gold.get("label", {}).get("tool_calls", [])
            gold_ask = [c for c in gold_calls if c["name"] == "sys.ask_missing_info"][0]
            gold_non_ask = [c for c in gold_calls if c["name"] != "sys.ask_missing_info"]

            prompt_idx = None
            for i, m in enumerate(msgs):
                if m["role"] == "user" and "User Voice Request:" in m["content"]:
                    prompt_idx = i
                    break
            if prompt_idx is None:
                prompt_idx = [i for i, m in enumerate(msgs) if m["role"] == "user"][-1]

            # Get final response text
            last_asst = [m for m in msgs[prompt_idx:] if m["role"] == "assistant"][-1]
            last_data = json.loads(last_asst["content"])
            final_text = last_data.get("final_response") or "Tôi cần thêm thông tin để thực hiện yêu cầu của bạn."

            session = factory.create_session("train", sid)
            new_curr = [msgs[prompt_idx]]

            if gold_non_ask:
                step1_obs = []
                for c in gold_non_ask:
                    res = session.call(c["name"], c.get("arguments", {}))
                    step1_obs.append({"tool_name": c["name"], "ok": res.ok, "result": res.result})

                thought_1 = f"Người dùng yêu cầu thực hiện tác vụ. Tuân thủ READ-BEFORE-WRITE và quy tắc tối giản tham số, thực hiện các thao tác cần thiết: {', '.join(c['name'] for c in gold_non_ask)}."
                new_curr.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "thought": thought_1,
                        "tool_calls": gold_non_ask,
                        "is_finished": False,
                        "final_response": None
                    }, ensure_ascii=False)
                })
                new_curr.append({
                    "role": "user",
                    "content": f"OBSERVATIONS FROM TOOL EXECUTION:\n{json.dumps(step1_obs, ensure_ascii=False, indent=2)}\n\nPlease provide your next action or final response."
                })

            thought_ask = "Phát hiện thông tin còn mơ hồ hoặc tìm thấy nhiều ứng viên / thiếu tham số bắt buộc. Tuân thủ nghiêm ngặt AMBIGUITY & MISSING INFO PROTOCOL, thực hiện gọi sys.ask_missing_info để kích hoạt giao thức hỏi lại người dùng trước khi thực thi."
            new_curr.append({
                "role": "assistant",
                "content": json.dumps({
                    "thought": thought_ask,
                    "tool_calls": [gold_ask],
                    "is_finished": False,
                    "final_response": None
                }, ensure_ascii=False)
            })

            session.call(gold_ask["name"], gold_ask.get("arguments", {}))
            new_curr.append({
                "role": "user",
                "content": f"OBSERVATIONS FROM TOOL EXECUTION:\n{json.dumps([{'tool_name': 'sys.ask_missing_info', 'ok': True, 'result': {'ok': True, 'waiting_for_user': True}}], ensure_ascii=False, indent=2)}\n\nPlease provide your next action or final response."
            })

            thought_final = "Đã phát lệnh sys.ask_missing_info và ghi nhận trạng thái chờ làm rõ. Cung cấp câu hỏi làm rõ tự nhiên cho người dùng."
            new_curr.append({
                "role": "assistant",
                "content": json.dumps({
                    "thought": thought_final,
                    "tool_calls": [],
                    "is_finished": True,
                    "final_response": final_text
                }, ensure_ascii=False)
            })

            msgs = msgs[:prompt_idx] + new_curr
            s["messages"] = msgs

            # Evaluate sample
            pred_hash = session.get_state_hash()
            gold_session = factory.create_session("train", sid)
            for c in gold_calls:
                gold_session.call(c["name"], c.get("arguments", {}))
            gold_hash = gold_session.get_state_hash()

            executed_calls = gold_non_ask + [gold_ask]
            eval_res = compute_sample_score(executed_calls, gold_calls, pred_hash, gold_hash)

            s["score"] = eval_res["score"]
            s["f1"] = eval_res["f1"]
            s["state_match"] = eval_res["state_match"]
            s["pred_hash"] = pred_hash
            s["gold_hash"] = gold_hash
            updated_count_trajectory += 1

    print(f"✅ System prompts updated:      {updated_count_prompt} samples")
    print(f"✅ Trajectories harmonized:     {updated_count_trajectory} samples")

    # 4. Re-evaluate and compute whole dataset metrics
    scores = []
    f1s = []
    state_matches = []
    token_lengths = []

    for s in samples:
        scores.append(s.get("score", 0.0))
        f1s.append(s.get("f1", 0.0))
        state_matches.append(s.get("state_match", False))
        txt = " ".join([m["content"] for m in s["messages"]])
        token_lengths.append(len(enc.encode(txt)))

    avg_score = float(np.mean(scores))
    avg_f1 = float(np.mean(f1s))
    perfect_count = sum(1 for sc in scores if sc == 1.0)
    match_rate = sum(1 for m in state_matches if m) / len(state_matches)

    print("\n" + "=" * 80)
    print("📈 FINAL DATASET EVALUATION METRICS:")
    print(f"  Total samples:            {len(samples)}")
    print(f"  State Match Rate:         {match_rate * 100:.2f}% ({sum(state_matches)} / {len(samples)})")
    print(f"  Average Score:            {avg_score * 100:.2f}%")
    print(f"  Average F1:               {avg_f1 * 100:.2f}%")
    print(f"  Perfect 1.00 Count:       {perfect_count} / {len(samples)} ({perfect_count / len(samples) * 100:.2f}%)")
    print(f"  Avg Tokens per Sample:    {float(np.mean(token_lengths)):.1f} tokens")
    print("=" * 80)

    assert match_rate == 1.0, f"Error: State match is not 100% ({match_rate * 100:.2f}%)"

    # 5. Save master JSON & JSONL
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved updated master JSON:  {json_path}")

    jsonl_path = DISTILLATION_DIR / "harmonized_sft_1526_samples.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"💾 Saved updated master JSONL: {jsonl_path}")

    # 6. Save stratified splits (preserving exact IDs from existing split files)
    sft_dir = SFT_DIR
    train_path = sft_dir / "harmonized_train_chatml.jsonl"
    val_path = sft_dir / "harmonized_val_chatml.jsonl"

    # Read existing train/val IDs
    with open(train_path, "r", encoding="utf-8") as f:
        train_ids = [json.loads(line)["id"] for line in f if line.strip()]
    with open(val_path, "r", encoding="utf-8") as f:
        val_ids = [json.loads(line)["id"] for line in f if line.strip()]

    # Write updated train split
    with open(train_path, "w", encoding="utf-8") as f:
        for sid in train_ids:
            item = samples_dict[sid]
            f.write(json.dumps({"id": item["id"], "messages": item["messages"]}, ensure_ascii=False) + "\n")
    print(f"💾 Saved updated train split ({len(train_ids)} samples): {train_path}")

    # Write updated val split
    with open(val_path, "w", encoding="utf-8") as f:
        for sid in val_ids:
            item = samples_dict[sid]
            f.write(json.dumps({"id": item["id"], "messages": item["messages"]}, ensure_ascii=False) + "\n")
    print(f"💾 Saved updated val split ({len(val_ids)} samples):   {val_path}")

    # Update info JSON
    info_path = sft_dir / "harmonized_dataset_info.json"
    info = {
        "dataset_name": "LifeMate-Harmonized-SFT-1526",
        "description": "Harmonized 3-tier SFT dataset with selective Ambiguity Protocol and 100% State Match.",
        "total_samples": len(samples),
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "prompt_breakdown": {
            "upgraded_ambiguity_protocol_samples": len(target_ids),
            "baseline_minimal_argument_samples": len(samples) - len(target_ids)
        },
        "metrics": {
            "state_match_rate": 1.0,
            "average_score": avg_score,
            "average_f1": avg_f1,
            "perfect_score_count": perfect_count
        },
        "token_statistics": {
            "mean": float(np.mean(token_lengths)),
            "max": int(np.max(token_lengths)),
            "min": int(np.min(token_lengths))
        },
        "files": {
            "master_json": str(json_path.name),
            "master_jsonl": str(jsonl_path.name),
            "train_chatml": str(train_path.name),
            "val_chatml": str(val_path.name)
        }
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved updated metadata:     {info_path}")
    print("=" * 80)
    print("🎉 ALL DATASET FILES SUCCESSFULLY UPDATED AND SYNCHRONIZED!")

if __name__ == "__main__":
    main()

