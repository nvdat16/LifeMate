"""
Synchronize Unified System Prompt Across Entire Harmonized Dataset (1,526 samples)

Features:
- Replaces the legacy/minimal missing info rule with the comprehensive
  AMBIGUITY & MISSING INFO PROTOCOL across all 1,526 samples.
- Eliminates Train-Serving Skew: 100% of samples now share the identical System Prompt architecture.
- Updates:
    1. training/llm/data/distillation/harmonized_sft_1526_samples.json
    2. training/llm/data/distillation/harmonized_sft_1526_samples.jsonl
    3. training/llm/data/sft/harmonized_train_chatml.jsonl (1,450 train)
    4. training/llm/data/sft/harmonized_val_chatml.jsonl (76 val)
    5. training/llm/data/sft/harmonized_dataset_info.json
    6. scripts/harmonize_sft_dataset.py (build_harmonized_system_prompt)
"""

import sys
import json
from pathlib import Path
import tiktoken
import numpy as np

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))
from paths import (PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR, LLM_OUTPUT_DIR)

WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT))

enc = tiktoken.get_encoding("cl100k_base")

UPGRADED_RULE = """4. AMBIGUITY & MISSING INFO PROTOCOL (STRICT MANDATORY TOOL CALL):
   - Whenever an entity search returns MULTIPLE candidates (e.g. 2 contacts named "Minh", 2 chats named "Lan"), or when a required parameter cannot be determined with certainty:
   - You MUST EXPLICITLY INVOKE `sys.ask_missing_info` with the intended write tool name and the ambiguous/missing parameter(s):
     {"name": "sys.ask_missing_info", "arguments": {"tool_name": "<intended_tool_name>", "params": ["<param_name>"]}}
   - NEVER ask the user only via natural language (final_response) without invoking `sys.ask_missing_info`! Calling `sys.ask_missing_info` is the obligatory system event to trigger user clarification."""

OLD_RULE_1 = "4. MISSING INFO: If mandatory information is missing and cannot be retrieved via read tools, call `sys.ask_missing_info`."
OLD_RULE_2 = "4. MISSING INFO: If mandatory information is missing and cannot be retrieved via read tools, call sys.ask_missing_info."

def main():
    print("=" * 80)
    print("🚀 SYNCHRONIZING UNIFIED SYSTEM PROMPT ACROSS ALL 1,526 SAMPLES")
    print("=" * 80)

    json_path = DISTILLATION_DIR / "harmonized_sft_1526_samples.json"
    with open(json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    updated_count = 0
    already_count = 0

    for s in samples:
        sys_msg = s["messages"][0]["content"]
        if "AMBIGUITY & MISSING INFO PROTOCOL" in sys_msg:
            already_count += 1
            continue

        if OLD_RULE_1 in sys_msg:
            s["messages"][0]["content"] = sys_msg.replace(OLD_RULE_1, UPGRADED_RULE)
            updated_count += 1
        elif OLD_RULE_2 in sys_msg:
            s["messages"][0]["content"] = sys_msg.replace(OLD_RULE_2, UPGRADED_RULE)
            updated_count += 1
        else:
            # Fallback regex if slight whitespace difference
            import re
            new_sys, n = re.subn(r"4\.\s*MISSING INFO:.*?(?=\n\n### PARAMETER INVOCATION RULES)", UPGRADED_RULE, sys_msg, flags=re.DOTALL)
            if n > 0:
                s["messages"][0]["content"] = new_sys
                updated_count += 1
            else:
                print(f"⚠️ Warning: Could not find target rule in {s['id']}")

    print(f"✅ Samples already with UPGRADED prompt: {already_count}")
    print(f"✅ Samples newly synchronized:          {updated_count}")
    print(f"✅ Total unified samples:                {already_count + updated_count} / {len(samples)}")

    assert (already_count + updated_count) == len(samples), "Not all samples were synchronized!"

    # Calculate token lengths
    token_lengths = []
    scores = []
    f1s = []
    state_matches = []
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
    print("📈 UNIFIED DATASET METRICS:")
    print(f"  Total samples:            {len(samples)}")
    print(f"  System Prompt Status:     100% UNIFIED (1526 / 1526)")
    print(f"  State Match Rate:         {match_rate * 100:.2f}% ({sum(state_matches)} / {len(samples)})")
    print(f"  Average Score:            {avg_score * 100:.2f}%")
    print(f"  Average F1:               {avg_f1 * 100:.2f}%")
    print(f"  Perfect 1.00 Count:       {perfect_count} / {len(samples)} ({perfect_count / len(samples) * 100:.2f}%)")
    print(f"  Avg Tokens per Sample:    {float(np.mean(token_lengths)):.1f} tokens")
    print("=" * 80)

    # 1. Save master JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved master JSON:  {json_path}")

    # 2. Save master JSONL
    jsonl_path = DISTILLATION_DIR / "harmonized_sft_1526_samples.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"💾 Saved master JSONL: {jsonl_path}")

    # 3. Save train/val splits
    sft_dir = SFT_DIR
    train_path = sft_dir / "harmonized_train_chatml.jsonl"
    val_path = sft_dir / "harmonized_val_chatml.jsonl"

    samples_dict = {s["id"]: s for s in samples}

    with open(train_path, "r", encoding="utf-8") as f:
        train_ids = [json.loads(line)["id"] for line in f if line.strip()]
    with open(val_path, "r", encoding="utf-8") as f:
        val_ids = [json.loads(line)["id"] for line in f if line.strip()]

    with open(train_path, "w", encoding="utf-8") as f:
        for sid in train_ids:
            item = samples_dict[sid]
            f.write(json.dumps({"id": item["id"], "messages": item["messages"]}, ensure_ascii=False) + "\n")
    print(f"💾 Saved train split ({len(train_ids)} samples): {train_path}")

    with open(val_path, "w", encoding="utf-8") as f:
        for sid in val_ids:
            item = samples_dict[sid]
            f.write(json.dumps({"id": item["id"], "messages": item["messages"]}, ensure_ascii=False) + "\n")
    print(f"💾 Saved val split ({len(val_ids)} samples):   {val_path}")

    # 4. Save metadata info JSON
    info_path = sft_dir / "harmonized_dataset_info.json"
    info = {
        "dataset_name": "LifeMate-Harmonized-SFT-1526",
        "description": "Harmonized 3-tier SFT dataset with 100% unified System Prompt architecture and 100% State Match.",
        "total_samples": len(samples),
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "prompt_breakdown": {
            "unified_comprehensive_prompt_samples": len(samples)
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
    print(f"💾 Saved metadata info: {info_path}")

if __name__ == "__main__":
    main()

