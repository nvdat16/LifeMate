"""
Prepare and Format Dataset for SFT (Supervised Fine-Tuning) with Unsloth / HuggingFace.

Outputs:
1. training/llm/data/sft/train_chatml.jsonl (Train split, standard messages format)
2. training/llm/data/sft/val_chatml.jsonl (Validation split, standard messages format)
3. training/llm/data/sft/train_sharegpt.jsonl (ShareGPT conversations format for LLaMA-Factory / Axolotl)
4. training/llm/data/sft/val_sharegpt.jsonl (ShareGPT conversations format)
5. training/llm/data/sft/dataset_info.json (Summary metadata, splits, token stats)
"""

import os
import sys
import json
import random
from pathlib import Path
import tiktoken
import numpy as np

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))
from paths import (PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR, LLM_OUTPUT_DIR)

WORKSPACE_ROOT = PROJECT_ROOT
OUTPUT_DIR = SFT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_JSON = DISTILLATION_DIR / "sft_train_1526_samples.json"

enc = tiktoken.get_encoding("cl100k_base")

def main():
    print("=" * 80)
    print("🚀 PREPARING SFT DATASET (CHATML & SHAREGPT FORMATS)")
    print(f"Input file:  {INPUT_JSON}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print("=" * 80)

    if not INPUT_JSON.is_file():
        print(f"Error: {INPUT_JSON} not found! Please run export_sft_dataset.py first.")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_samples = len(data)
    print(f"Loaded {total_samples} verified samples.")

    # 1. Validation checks
    valid_samples = []
    token_lengths = []

    for item in data:
        sid = item["id"]
        messages = item["messages"]
        if not messages or len(messages) < 3:
            print(f"Warning: sample {sid} has too few messages ({len(messages)}). Skipping.")
            continue

        # Check that all assistant responses are valid JSON
        all_json_ok = True
        for m in messages:
            if m["role"] == "assistant":
                try:
                    parsed = json.loads(m["content"])
                    if "thought" not in parsed or "tool_calls" not in parsed:
                        all_json_ok = False
                except Exception:
                    all_json_ok = False
                    break

        if not all_json_ok:
            print(f"Warning: sample {sid} contains invalid assistant JSON. Skipping.")
            continue

        # Measure token length
        full_text = "".join(m["content"] for m in messages)
        n_tok = len(enc.encode(full_text))
        token_lengths.append(n_tok)

        valid_samples.append((item, n_tok))

    print(f"Valid samples after verification: {len(valid_samples)}/{total_samples}")
    print(f"Token length stats -> Mean: {np.mean(token_lengths):.1f}, Max: {np.max(token_lengths)}, Min: {np.min(token_lengths)}")

    # 2. Train / Validation Split (Stratified by teacher)
    # Stratify by teacher so both splits have Luna, Terra, and Sol representations
    teachers = {}
    for item, n_tok in valid_samples:
        t = item.get("teacher", "unknown")
        teachers.setdefault(t, []).append((item, n_tok))

    random.seed(42)
    train_items = []
    val_items = []

    # 95% train / 5% validation split (~76 validation samples for fast evaluation)
    val_ratio = 0.05
    for t_name, group in teachers.items():
        random.shuffle(group)
        n_val = max(1, int(len(group) * val_ratio))
        val_items.extend(group[:n_val])
        train_items.extend(group[n_val:])

    print(f"\nSplit distribution:")
    print(f"  Train set: {len(train_items)} samples ({len(train_items)/len(valid_samples)*100:.1f}%)")
    print(f"  Val set:   {len(val_items)} samples ({len(val_items)/len(valid_samples)*100:.1f}%)")

    # 3. Export Formats
    # A. ChatML format (Standard messages: [{"role": ..., "content": ...}])
    train_chatml_path = OUTPUT_DIR / "train_chatml.jsonl"
    with open(train_chatml_path, "w", encoding="utf-8") as f:
        for item, _ in train_items:
            f.write(json.dumps({"id": item["id"], "messages": item["messages"]}, ensure_ascii=False) + "\n")

    val_chatml_path = OUTPUT_DIR / "val_chatml.jsonl"
    with open(val_chatml_path, "w", encoding="utf-8") as f:
        for item, _ in val_items:
            f.write(json.dumps({"id": item["id"], "messages": item["messages"]}, ensure_ascii=False) + "\n")

    # B. ShareGPT format (Standard conversations: [{"from": ..., "value": ...}])
    role_map = {"system": "system", "user": "human", "assistant": "gpt"}
    def to_sharegpt(messages):
        convs = []
        for m in messages:
            convs.append({
                "from": role_map.get(m["role"], m["role"]),
                "value": m["content"]
            })
        return convs

    train_sharegpt_path = OUTPUT_DIR / "train_sharegpt.jsonl"
    with open(train_sharegpt_path, "w", encoding="utf-8") as f:
        for item, _ in train_items:
            f.write(json.dumps({
                "id": item["id"],
                "conversations": to_sharegpt(item["messages"])
            }, ensure_ascii=False) + "\n")

    val_sharegpt_path = OUTPUT_DIR / "val_sharegpt.jsonl"
    with open(val_sharegpt_path, "w", encoding="utf-8") as f:
        for item, _ in val_items:
            f.write(json.dumps({
                "id": item["id"],
                "conversations": to_sharegpt(item["messages"])
            }, ensure_ascii=False) + "\n")

    # C. Metadata summary
    info = {
        "dataset_name": "LifeMate-Agentic-SFT-1526",
        "description": "Golden ReAct trajectories verified via Zaic Sandbox Environment state matching.",
        "total_samples": len(valid_samples),
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "token_statistics": {
            "mean": float(np.mean(token_lengths)),
            "median": float(np.median(token_lengths)),
            "min": int(np.min(token_lengths)),
            "max": int(np.max(token_lengths)),
            "p95": float(np.percentile(token_lengths, 95)),
            "p99": float(np.percentile(token_lengths, 99))
        },
        "recommended_max_seq_length": 10240,
        "files": {
            "train_chatml": str(train_chatml_path.name),
            "val_chatml": str(val_chatml_path.name),
            "train_sharegpt": str(train_sharegpt_path.name),
            "val_sharegpt": str(val_sharegpt_path.name)
        }
    }
    with open(OUTPUT_DIR / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("✅ DATASET PREPARATION COMPLETE!")
    print(f"📁 Files saved in: {OUTPUT_DIR}")
    print(f"  1. {train_chatml_path.name} ({len(train_items)} samples)")
    print(f"  2. {val_chatml_path.name} ({len(val_items)} samples)")
    print(f"  3. {train_sharegpt_path.name}")
    print(f"  4. {val_sharegpt_path.name}")
    print(f"  5. dataset_info.json")
    print("=" * 80)

if __name__ == "__main__":
    main()

